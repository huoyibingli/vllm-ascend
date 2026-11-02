# SPDX-License-Identifier: Apache-2.0
"""Check ND GMM and MegaMoe storage sharing on two idle A2 cards.

Direct invocation:
    torchrun --standalone --nproc-per-node=2 FILE --output-dir /tmp/nd-views
"""

import argparse
import datetime
import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

NUM_RANKS = 2
NUM_EXPERTS = 256
HIDDEN = 2048
INTERMEDIATE = 512
TOP_K = 8
MAX_TOKENS = 512


def _run_rank(output: Path):
    import torch_npu
    from cann_ops_transformer import get_symm_buffer_for_mega_moe, mega_moe

    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", timeout=datetime.timedelta(seconds=120))
    assert dist.get_world_size() == NUM_RANKS
    dist.all_reduce(torch.ones(1, device="npu"))
    torch.npu.config.allow_internal_format = True
    torch.manual_seed(811 + rank)
    torch.npu.manual_seed(811 + rank)
    with torch.inference_mode():
        w13 = torch.randn((NUM_EXPERTS // NUM_RANKS, HIDDEN, 2 * INTERMEDIATE), device="npu", dtype=torch.bfloat16)
        w2 = torch.randn((NUM_EXPERTS // NUM_RANKS, INTERMEDIATE, HIDDEN), device="npu", dtype=torch.bfloat16)
        w13.mul_(0.02)
        w2.mul_(0.02)
        views1, views2 = list(w13.unbind(0)), list(w2.unbind(0))
        clones1, clones2 = [w.clone() for w in views1], [w.clone() for w in views2]
        for stacked, views in ((w13, views1), (w2, views2)):
            assert all(v.untyped_storage().data_ptr() == stacked.untyped_storage().data_ptr() for v in views)
            assert views[-1].storage_offset() == (len(views) - 1) * stacked[0].numel()
        gmm_results = []
        for name, stacked, clones in (("w13", w13, clones1), ("w2", w2, clones2)):
            # Include empty groups and both the first and last expert offsets.
            groups = (torch.arange(NUM_EXPERTS // NUM_RANKS, device="npu", dtype=torch.int64) + rank) % 5
            rows = int(groups.sum().item())
            hidden = torch.randn((rows, stacked.shape[1]), device="npu", dtype=torch.bfloat16).mul_(0.1)
            kwargs = dict(x=[hidden], split_item=2, group_type=0, group_list_type=1, group_list=groups)
            reference = torch_npu.npu_grouped_matmul(weight=clones, **kwargs)[0]
            actual = torch_npu.npu_grouped_matmul(weight=[stacked], **kwargs)[0]
            torch.npu.synchronize()
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            gmm_results.append(dict(projection=name, exact=True))
        buffer = get_symm_buffer_for_mega_moe(
            dist.group.WORLD,
            NUM_EXPERTS,
            MAX_TOKENS,
            TOP_K,
            HIDDEN,
            2 * INTERMEDIATE,
            max_recv_token_num=MAX_TOKENS * NUM_RANKS * TOP_K,
        )
        results = []
        for tokens, masked in ((32, False), (65, False), (MAX_TOKENS, False), (32, True)):
            x = torch.randn((tokens, HIDDEN), device="npu", dtype=torch.bfloat16).mul_(0.1)
            ids = ((torch.arange(tokens * TOP_K, device="npu").reshape(tokens, TOP_K) + 97 * rank) % NUM_EXPERTS).to(
                torch.int32
            )
            probs = torch.softmax(torch.randn((tokens, TOP_K), device="npu"), -1)
            mask = torch.ones(tokens, device="npu", dtype=torch.int8)
            if masked:
                mask[-5:] = 0
            reference, reference_counts = mega_moe(x, ids, probs, clones1, clones2, buffer, x_active_mask=mask)
            torch.npu.synchronize()
            dist.barrier()
            actual, counts = mega_moe(x, ids, probs, views1, views2, buffer, x_active_mask=mask)
            torch.npu.synchronize()
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            torch.testing.assert_close(counts, reference_counts, atol=0, rtol=0)
            results.append(dict(tokens=tokens, masked=masked, exact=True))
            dist.barrier()
        output.mkdir(parents=True, exist_ok=True)
        (output / f"rank{rank}.json").write_text(json.dumps(dict(gmm=gmm_results, megamoe=results), indent=2) + "\n")
        dist.barrier()
        del buffer
        dist.destroy_process_group()


def test_megamoe_and_gmm_share_nd_storage(tmp_path):
    torch_npu = pytest.importorskip("torch_npu")
    if not torch_npu.npu.is_available() or torch_npu.npu.device_count() < NUM_RANKS:
        pytest.skip("Requires two idle A2 cards")
    if "910B" not in torch.npu.get_device_name(0):
        pytest.skip("This BF16 MegaMoe case targets A2")
    if importlib.util.find_spec("cann_ops_transformer") is None:
        pytest.skip("Requires the CANN MegaMoe extension")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={NUM_RANKS}",
        str(Path(__file__).resolve()),
        "--output-dir",
        str(tmp_path),
    ]
    with (tmp_path / "workers.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            assert process.wait(timeout=360) == 0, (tmp_path / "workers.log").read_text()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    for rank in range(NUM_RANKS):
        results = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert len(results["gmm"]) == 2 and len(results["megamoe"]) == 4
        assert all(case["exact"] for case in results["gmm"] + results["megamoe"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    _run_rank(parser.parse_args().output_dir)
