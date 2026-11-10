"""TP2 BF16 NZ shared-weight GMM/MegaMoe prepare/kernel/finalize regression.

Requires two idle A2 cards, CANN MegaMoe, and the installed editable checkout:
    ASCEND_RT_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc-per-node=2 \
        tests/e2e/nightly/single_node/ops/test_mega_moe_bf16_nz.py

Identity-projection experts isolate padding, token order and active-mask errors.
This is an operator integration test, not full-model accuracy or latency data.
"""

import json
import os
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu

from vllm_ascend.ops.fused_moe import prepare_finalize as pf
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights, build_fused_experts_input
from vllm_ascend.ops.fused_moe.megamoe_bf16_nz import make_bf16_nz_expert_weights
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl
from vllm_ascend.ops.fused_moe.moe_utils import load_cann_mega_moe_ops
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
from vllm_ascend.quantization.quant_type import QuantType

HIDDEN = 2048
INTERMEDIATE = 512
EXPERTS = 256
TOPK = 8
CAPACITY = 16384
TOKEN_COUNTS = (65, 1055, 15361, 15376, CAPACITY)


def main():
    assert int(os.environ["WORLD_SIZE"]) == 2, "This test requires exactly two ranks"
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    group = dist.group.WORLD
    dist.all_reduce(torch.ones(1, device="npu"))
    allocate, mega = load_cann_mega_moe_ops()
    buffer = allocate(
        group,
        EXPERTS,
        CAPACITY // 2,
        TOPK,
        hidden=HIDDEN,
        intermediate_hidden=INTERMEDIATE,
        max_recv_token_num=CAPACITY * TOPK,
        dispatch_quant_mode=0,
    )
    w1 = torch.zeros((HIDDEN, 2 * INTERMEDIATE), device="npu", dtype=torch.bfloat16)
    w2 = torch.zeros((INTERMEDIATE, HIDDEN), device="npu", dtype=torch.bfloat16)
    diagonal = torch.arange(INTERMEDIATE, device="npu")
    w1[diagonal, diagonal] = 1
    w1[diagonal, diagonal + INTERMEDIATE] = 1
    w2[diagonal, diagonal] = 1
    torch.npu.config.allow_internal_format = True
    nz1, nz2 = make_bf16_nz_expert_weights(
        w1.unsqueeze(0).repeat(EXPERTS // 2, 1, 1), w2.unsqueeze(0).repeat(EXPERTS // 2, 1, 1)
    )
    weights = MoEWeights(w1=nz1, w2=nz2)
    assert len(nz1) == len(nz2) == 1 and nz1[0].ndim == nz2[0].ndim == 3
    # Ordinary GMM must consume the identical persistent 3D NZ allocations.
    gmm_x = torch.full((EXPERTS // 2, HIDDEN), 0.25, device="npu", dtype=torch.bfloat16)
    groups = torch.ones(EXPERTS // 2, device="npu", dtype=torch.int64)
    gate_up = torch_npu.npu_grouped_matmul(
        x=[gmm_x], weight=nz1, group_list=groups, group_list_type=1, split_item=2, group_type=0
    )[0]
    act = torch_npu.npu_swiglu(gate_up)
    gmm_y = torch_npu.npu_grouped_matmul(
        x=[act], weight=nz2, group_list=groups, group_list_type=1, split_item=2, group_type=0
    )[0]
    reference = torch.zeros_like(gmm_x)
    reference[:, :INTERMEDIATE] = torch_npu.npu_swiglu(torch.cat((gmm_x[:, :INTERMEDIATE],) * 2, -1))
    torch.testing.assert_close(gmm_y.cpu(), reference.cpu(), rtol=0, atol=0)
    addresses = [w.data_ptr() for w in nz1 + nz2]
    comm = object.__new__(FusedMC2CommImpl)
    comm.mega_moe, comm.mega_moe_symm_buffer = mega, buffer
    comm.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
    comm.token_dispatcher.max_num_tokens_per_rank = CAPACITY // 2
    comm.token_dispatcher.global_bs = 0
    layer = torch.nn.Module()
    layer.mega_moe_activation_kwargs = {}
    prepare = object.__new__(pf.PrepareAndFinalizeWithMC2)
    prepare.tp_size, prepare.tp_rank = 2, rank
    prepare.moe_config = SimpleNamespace(tp_group=SimpleNamespace(device_group=group))
    for tokens in TOKEN_COUNTS:
        padded = ((tokens + 1) // 2) * 2
        rows = torch.arange(tokens, device="npu")
        x = ((rows % 4 + 1).to(torch.bfloat16) / 8).unsqueeze(1).expand(-1, HIDDEN).contiguous()
        ids = ((rows.unsqueeze(1) + torch.arange(TOPK, device="npu")) % EXPERTS).to(torch.int32)
        probs = torch.full((tokens, TOPK), 1 / TOPK, device="npu", dtype=torch.float32)
        context = SimpleNamespace(mc2_mask=torch.arange(padded, device="npu") < tokens, padded_num_tokens=padded)
        with patch.object(pf, "_EXTRA_CTX", context):
            prepared = prepare.prepare(x, probs)
        local_ids = torch.nn.functional.pad(ids, (0, 0, 0, padded - tokens)).chunk(2)[rank].contiguous()
        inp = build_fused_experts_input(
            hidden_states=prepared.hidden_states,
            topk_ids=local_ids,
            topk_weights=prepared.router_logits,
            mc2_mask=prepared.mc2_mask,
            quant_type=QuantType.NONE,
            dynamic_eplb=False,
            layer=layer,
        )
        output, expert_tokens = comm._apply_cann_mega_moe(inp, weights, is_decode_only_node=False)
        actual = prepare.finalize(output, False, prepared.padded_hidden_states_shape)
        golden = torch.zeros_like(x)
        gate_up = torch.cat((x[:, :INTERMEDIATE], x[:, :INTERMEDIATE]), dim=-1)
        golden[:, :INTERMEDIATE] = torch_npu.npu_swiglu(gate_up)
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), golden.cpu(), rtol=0, atol=0)
        expected_counts = torch.bincount(ids.flatten().to(torch.int64), minlength=EXPERTS)
        expected_counts = expected_counts.chunk(2)[rank].cpu()
        torch.testing.assert_close(expert_tokens.cpu().flatten().to(torch.int64), expected_counts, rtol=0, atol=0)
        print(json.dumps({"rank": rank, "tokens": tokens, "padded": padded, "exact_match": True}), flush=True)
    assert addresses == [w.data_ptr() for w in nz1 + nz2]
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
