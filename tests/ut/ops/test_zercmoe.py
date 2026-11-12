# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import torch
from vllm.config import CompilationMode, CUDAGraphMode

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.device.hardware_profile import MoECommPolicy
from vllm_ascend.ops.fused_moe import zercmoe


class TestZercMoE(TestCase):
    def test_prefill_and_mixed_modes(self):
        def metadata(*prefilling):
            requests = [SimpleNamespace(is_prefill=phase) for phase in prefilling]
            return {"gdn": object(), "qsa": SimpleNamespace(qsa_request_plan=SimpleNamespace(requests=requests))}

        with patch.object(afc, "get_ascend_config") as get_config:
            config = SimpleNamespace(enable_zerc_moe=1, zercmoe_batch_mode="prefill")
            get_config.return_value = config
            self.assertTrue(afc.should_try_zercmoe(metadata(True), CUDAGraphMode.NONE, False))
            self.assertFalse(afc.should_try_zercmoe(metadata(True, False), CUDAGraphMode.NONE, False))
            config.zercmoe_batch_mode = "prefill_and_mixed"
            self.assertTrue(afc.should_try_zercmoe(metadata(True, False), CUDAGraphMode.NONE, False))
            self.assertFalse(afc.should_try_zercmoe(metadata(False), CUDAGraphMode.NONE, False))
            self.assertFalse(afc.should_try_zercmoe(metadata(True, False), CUDAGraphMode.FULL, False))
            self.assertFalse(afc.should_try_zercmoe(None, CUDAGraphMode.NONE, False))
            self.assertTrue(afc.should_try_zercmoe(None, CUDAGraphMode.NONE, True))

    def test_all_mode_accepts_every_eager_phase(self):
        config = SimpleNamespace(enable_zerc_moe=1, zercmoe_batch_mode="all")
        with patch.object(afc, "get_ascend_config", return_value=config):
            for metadata in (
                SimpleNamespace(num_prefills=1, num_decodes=0),
                SimpleNamespace(num_prefills=1, num_decodes=1),
                SimpleNamespace(num_prefills=0, num_decodes=1),
                None,
            ):
                with self.subTest(metadata=metadata):
                    self.assertTrue(afc.should_try_zercmoe(metadata, CUDAGraphMode.NONE, False))
                    for graph_mode in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL):
                        self.assertFalse(afc.should_try_zercmoe(metadata, graph_mode, False))
            config.enable_zerc_moe = 0
            self.assertFalse(afc.should_try_zercmoe(None, CUDAGraphMode.NONE, True))

    def test_all_mode_disables_compilation_and_decode_graphs(self):
        from vllm_ascend.platform import _update_compilation_modes

        for batch_mode in ("prefill", "prefill_and_mixed", "all"):
            for graph_mode in (CUDAGraphMode.NONE, CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY):
                for enabled, supported in ((1, True), (0, True), (1, False)):
                    with self.subTest(mode=batch_mode, graph=graph_mode, enabled=enabled, supported=supported):
                        config = SimpleNamespace(
                            model_config=SimpleNamespace(enforce_eager=False, is_encoder_decoder=False),
                            compilation_config=SimpleNamespace(
                                mode=CompilationMode.VLLM_COMPILE, cudagraph_mode=graph_mode, splitting_ops=None
                            ),
                            additional_config={},
                        )
                        ascend_config = SimpleNamespace(
                            ascend_compilation_config={},
                            xlite_graph_config=SimpleNamespace(enabled=False),
                            enable_zerc_moe=enabled,
                            zercmoe_batch_mode=batch_mode,
                        )
                        with patch.object(afc, "is_zercmoe_supported_by_config", return_value=supported):
                            _update_compilation_modes(config, ascend_config)
                        all_enabled = batch_mode == "all" and enabled == 1 and supported
                        self.assertEqual(config.model_config.enforce_eager, all_enabled)
                        self.assertEqual(
                            config.compilation_config.mode,
                            CompilationMode.NONE if all_enabled else CompilationMode.VLLM_COMPILE,
                        )
                        expected_graph = graph_mode
                        if all_enabled:
                            expected_graph = CUDAGraphMode.NONE
                        elif enabled and supported and graph_mode == CUDAGraphMode.FULL:
                            expected_graph = CUDAGraphMode.FULL_DECODE_ONLY
                        self.assertEqual(config.compilation_config.cudagraph_mode, expected_graph)

    def test_selector_checks_capacity_and_draft(self):
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(enable_expert_parallel=True, tensor_parallel_size=8),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=4095),
        )
        with (
            patch.object(afc, "get_ascend_config", return_value=SimpleNamespace(deepgemm_mega_moe=False)),
            patch.object(afc, "is_moe_model", return_value=True),
            patch.object(afc, "get_ep_group", return_value=SimpleNamespace(world_size=8)),
            patch.object(
                afc,
                "get_current_hardware_profile",
                return_value=SimpleNamespace(moe_comm_policy=MoECommPolicy.ALLGATHER),
            ),
            patch.object(afc, "use_cann_zercmoe", return_value=True) as wheel,
        ):
            self.assertEqual(afc.get_zercmoe_tokens_capacity(config), 4096)
            for tokens, phase, draft, expected in (
                (64, True, False, afc.MoECommType.ALLGATHER),
                (65, True, False, afc.MoECommType.ZERC_MOE),
                (4096, True, False, afc.MoECommType.ZERC_MOE),
                (4097, True, False, afc.MoECommType.ALLGATHER),
                (1024, False, False, afc.MoECommType.ALLGATHER),
                (1024, True, True, afc.MoECommType.ALLGATHER),
            ):
                self.assertEqual(
                    afc.select_moe_comm_method(tokens, config, is_draft_model=draft, zercmoe_prefill=phase), expected
                )
            wheel.return_value = False
            self.assertEqual(afc.select_moe_comm_method(1024, config, zercmoe_prefill=True), afc.MoECommType.ALLGATHER)

    def test_all_mode_routes_small_decode_batches_with_capacity_guard(self):
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(enable_expert_parallel=True, tensor_parallel_size=8),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=4095),
        )
        ascend_config = SimpleNamespace(deepgemm_mega_moe=False, enable_zerc_moe=1, zercmoe_batch_mode="all")
        with (
            patch.object(afc, "get_ascend_config", return_value=ascend_config),
            patch.object(afc, "is_moe_model", return_value=True),
            patch.object(afc, "get_ep_group", return_value=SimpleNamespace(world_size=8)),
            patch.object(
                afc,
                "get_current_hardware_profile",
                return_value=SimpleNamespace(moe_comm_policy=MoECommPolicy.ALLGATHER),
            ),
            patch.object(afc, "use_cann_zercmoe", return_value=True) as wheel,
        ):
            decode = SimpleNamespace(num_prefills=0, num_decodes=1)
            eligible = afc.should_try_zercmoe(decode, CUDAGraphMode.NONE, False)
            for tokens in (1, 7, 8, 64, 65, 4096):
                with self.subTest(tokens=tokens):
                    self.assertEqual(
                        afc.select_moe_comm_method(tokens, config, zercmoe_prefill=eligible), afc.MoECommType.ZERC_MOE
                    )
            for tokens, phase, draft in ((0, True, False), (4097, True, False), (1, False, False), (1, True, True)):
                self.assertEqual(
                    afc.select_moe_comm_method(tokens, config, zercmoe_prefill=phase, is_draft_model=draft),
                    afc.MoECommType.ALLGATHER,
                )
            wheel.return_value = False
            self.assertEqual(afc.select_moe_comm_method(1, config, zercmoe_prefill=True), afc.MoECommType.ALLGATHER)

    def test_w8a8_payload_shares_weights_and_passes_scales(self):
        layer = SimpleNamespace(
            moe_config=SimpleNamespace(num_local_experts=2, hidden_dim=32, intermediate_size_per_partition=16),
            w13_weight=torch.zeros(2, 32, 32, dtype=torch.int8),
            w2_weight=torch.zeros(2, 16, 32, dtype=torch.int8),
            w13_weight_scale=torch.full((2, 32), 0.25),
            w2_weight_scale=torch.full((2, 32), 0.5),
            w13_weight_offset=torch.zeros(2, 32),
            w2_weight_offset=torch.zeros(2, 32),
        )
        zercmoe.validate_w8a8_zercmoe_weights(layer)
        encoder = Mock(side_effect=lambda scale: scale.view(torch.int32).to(torch.int64))
        weights = zercmoe.build_w8a8_zercmoe_weights(layer, encoder)
        self.assertEqual(encoder.call_count, 2)
        self.assertEqual(weights.w1[0].data_ptr(), layer.w13_weight[0].data_ptr())
        self.assertEqual(weights.w2[1].data_ptr(), layer.w2_weight[1].data_ptr())

        inputs = SimpleNamespace(
            hidden_states=torch.empty(2, 32, dtype=torch.bfloat16),
            topk_ids=torch.zeros((2, 1), dtype=torch.int64),
            topk_weights=torch.ones((2, 1), dtype=torch.bfloat16),
            routing=SimpleNamespace(mc2_mask=torch.tensor([True, False])),
        )
        buffer = SimpleNamespace(moe_expert_num=512, ep_world_size=8, ccl_buffer_size=128)
        with patch.object(torch.ops.npu, "zerc_moe_int8", create=True) as kernel:
            zercmoe.run_zercmoe(inputs, weights, buffer, w8a8=True)
        kwargs = kernel.call_args.kwargs
        self.assertIs(kwargs["x"], inputs.hidden_states)
        self.assertIs(kwargs["weight1"], weights.w1)
        self.assertIs(kwargs["weight2"], weights.w2)
        self.assertIs(kwargs["scale1"], weights.w1_scale)
        self.assertIs(kwargs["scale2"], weights.w2_scale)
        self.assertEqual(kwargs["expert_idx"].dtype, torch.int32)
        self.assertEqual(kwargs["probs"].dtype, torch.float32)
        torch.testing.assert_close(kwargs["x_active_mask"], torch.tensor([1, 0], dtype=torch.int8))
        self.assertEqual(kwargs["return_expert_tokens"], 0)

        layer.w2_weight_offset[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "zero offsets"):
            zercmoe.validate_w8a8_zercmoe_weights(layer)

    def test_bf16_packs_into_reusable_scratch(self):
        layer = SimpleNamespace(
            w13_weight=torch.arange(2 * 16 * 32).reshape(2, 16, 32).to(torch.bfloat16),
            w2_weight=torch.arange(2 * 16 * 16).reshape(2, 16, 16).to(torch.bfloat16),
        )
        scratch = zercmoe.pack_bf16_zercmoe_weights(layer, None)
        expected = torch.cat([part.flatten() for expert in layer.w13_weight for part in expert.split(16, dim=1)])
        torch.testing.assert_close(scratch.w1, expected, atol=0, rtol=0)
        layer.w13_weight.zero_()
        self.assertIs(zercmoe.pack_bf16_zercmoe_weights(layer, scratch), scratch)
        self.assertEqual(torch.count_nonzero(scratch.w1).item(), 0)
