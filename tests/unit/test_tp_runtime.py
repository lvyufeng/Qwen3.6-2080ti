from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from checkpoint import TensorInfo, build_manifest
from reference_ops import LinearDispatchStats, ReferenceWeights, decoder_layer, language_model
from runtime_config import parse_runtime_config
from tensor_parallel import TensorParallel
from decode_state import DecodeState
from tp_runtime import (
    RuntimeProfileConfig,
    TpLaunchConfig,
    TpRuntime,
    TpRuntimeError,
    mapped_tensor_bytes,
    tp_decode_step,
    tp_decoder_layer,
    tp_greedy_next_token,
    tp_greedy_next_tokens,
    tp_language_model,
    tp_moe,
    _record_paged_attention_dispatch,
    _recurrent_gated_delta_rule,
)
from loader import TensorLoader
from weight_mapping import (
    ExpertMapping,
    FullAttentionMapping,
    LanguageModelMapping,
    LayerMapping,
    LinearTensor,
    MoEMapping,
    ShardedTensor,
    TensorShard,
    build_language_model_mapping,
)


def test_tp_launch_config_validates_rank() -> None:
    with pytest.raises(TpRuntimeError, match="out of range"):
        TpLaunchConfig(world_size=2, rank=2)


def test_single_rank_runtime_all_reduce_is_noop_cpu() -> None:
    tensor = torch.tensor([1.0, 2.0])

    with TpRuntime(TpLaunchConfig(backend="gloo", device="cpu")) as runtime:
        out = runtime.all_reduce_sum(tensor)

    assert out is tensor
    torch.testing.assert_close(tensor, torch.tensor([1.0, 2.0]))


def test_runtime_profile_disabled_is_noop() -> None:
    with TpRuntime(TpLaunchConfig(backend="gloo", device="cpu")) as runtime:
        with runtime.profile_scope("example"):
            pass

    assert runtime.profile_stats.enabled is False
    assert runtime.profile_stats.scopes == {}


def test_runtime_profile_records_scope_cpu() -> None:
    with TpRuntime(TpLaunchConfig(backend="gloo", device="cpu")) as runtime:
        runtime.configure_profiling(RuntimeProfileConfig(enabled=True))
        with runtime.profile_scope("example", input_tokens=2, bytes=4):
            torch.tensor([1.0, 2.0]).sum()

    stats = runtime.profile_stats.scopes["example"]
    assert stats.calls == 1
    assert stats.total_seconds >= 0
    assert stats.max_seconds >= 0
    assert stats.input_tokens == 2
    assert stats.bytes == 4


def test_runtime_profile_records_single_rank_collective_bytes() -> None:
    tensor = torch.tensor([1.0, 2.0])

    with TpRuntime(TpLaunchConfig(backend="gloo", device="cpu")) as runtime:
        runtime.configure_profiling(RuntimeProfileConfig(enabled=True))
        runtime.all_reduce_sum(tensor)

    stats = runtime.profile_stats.scopes["collective.all_reduce_sum"]
    assert stats.calls == 1
    assert stats.bytes == tensor.numel() * tensor.element_size()


def test_recurrent_gated_delta_rule_cpu_fallback_preserves_shape_and_state() -> None:
    torch.manual_seed(0)
    query = torch.randn((2, 3, 4, 5), dtype=torch.float32)
    key = torch.randn((2, 3, 4, 5), dtype=torch.float32)
    value = torch.randn((2, 3, 4, 6), dtype=torch.float32)
    g = -torch.rand((2, 3, 4), dtype=torch.float32)
    beta = torch.rand((2, 3, 4), dtype=torch.float32)
    initial_state = torch.randn((2, 4, 5, 6), dtype=torch.float32)

    out, state = _recurrent_gated_delta_rule(query, key, value, g, beta, initial_state=initial_state, return_state=True)

    assert out.shape == (2, 3, 4, 6)
    assert out.dtype == query.dtype
    assert state.shape == (2, 4, 5, 6)
    assert state.dtype == torch.float32


def test_mapped_tensor_bytes_counts_local_expert_shard() -> None:
    mapping = LanguageModelMapping(
        model_dir=__import__("pathlib").Path("."),
        embed_tokens=_info("embed", (4, 2), nbytes=16),
        final_norm=_info("norm", (2,), nbytes=4),
        lm_head=_info("head", (4, 2), nbytes=16),
        layers=(),
        mapped_tensor_names=frozenset(),
        ignored_tensor_names=frozenset(),
        unmapped_language_tensor_names=(),
    )

    assert mapped_tensor_bytes(mapping) == 36


def test_packed_tp_moe_matches_loop_and_records_stats() -> None:
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    loader = _FakeLoader(_moe_test_tensors())
    mapping = _moe_mapping((0, 1), TensorParallel(world_size=1, rank=0))
    packed_config = _config_with_experts_per_token(2, packed=True)
    loop_config = _config_with_experts_per_token(2, packed=False)

    with TpRuntime(TpLaunchConfig(backend="gloo", device="cpu")) as runtime:
        expected = tp_moe(hidden, mapping, loop_config, ReferenceWeights(loader), runtime)
        weights = ReferenceWeights(loader)
        weights.dispatch_stats = LinearDispatchStats()
        actual = tp_moe(hidden, mapping, packed_config, weights, runtime)

    torch.testing.assert_close(actual, expected)
    stats = weights.dispatch_stats
    assert stats.moe_calls == 1
    assert stats.moe_packed_calls == 1
    assert stats.moe_loop_calls == 0
    assert stats.moe_assignments == 6
    assert stats.moe_local_assignments == 6
    assert stats.moe_active_expert_groups == 2
    assert stats.moe_empty_local_dispatches == 0
    assert stats.moe_max_group_tokens == 3
    assert stats.moe_packed_index_add_calls == 1
    assert stats.moe_packed_single_scatter_calls == 1
    assert stats.moe_group_size_1 == 0
    assert stats.moe_group_size_2_to_4 == 2
    assert stats.moe_group_size_5_to_8 == 0
    assert stats.moe_native_assignment_calls == 1
    assert stats.moe_native_assignment_hits == 0
    assert stats.moe_native_assignment_fallbacks == 1
    assert stats.moe_native_assignment_fallback_device == 1
    assert stats.moe_native_expert_calls == 2
    assert stats.moe_native_expert_hits == 0
    assert stats.moe_native_expert_fallbacks == 2
    assert stats.moe_native_expert_fallback_device == 2
    assert stats.moe_native_expert_max_group_tokens == 3


def test_packed_tp_moe_records_native_disabled_fallback() -> None:
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    loader = _FakeLoader(_moe_test_tensors())
    mapping = _moe_mapping((0, 1), TensorParallel(world_size=1, rank=0))
    packed_config = _config_with_experts_per_token(2, packed=True, native=False)
    loop_config = _config_with_experts_per_token(2, packed=False)
    weights = ReferenceWeights(loader)
    weights.dispatch_stats = LinearDispatchStats()

    with TpRuntime(TpLaunchConfig(backend="gloo", device="cpu")) as runtime:
        expected = tp_moe(hidden, mapping, loop_config, ReferenceWeights(loader), runtime)
        actual = tp_moe(hidden, mapping, packed_config, weights, runtime)

    torch.testing.assert_close(actual, expected)
    assert weights.dispatch_stats.moe_packed_index_add_calls == 1
    assert weights.dispatch_stats.moe_packed_single_scatter_calls == 1
    assert weights.dispatch_stats.moe_native_assignment_calls == 1
    assert weights.dispatch_stats.moe_native_assignment_hits == 0
    assert weights.dispatch_stats.moe_native_assignment_fallback_device == 1
    assert weights.dispatch_stats.moe_native_expert_calls == 2
    assert weights.dispatch_stats.moe_native_expert_hits == 0
    assert weights.dispatch_stats.moe_native_expert_fallbacks == 2
    assert weights.dispatch_stats.moe_native_expert_fallback_disabled == 2


def test_loop_tp_moe_records_loop_stats() -> None:
    hidden = torch.tensor([[[1.0, 0.0]]])
    loader = _FakeLoader(_moe_test_tensors())
    mapping = _moe_mapping((0, 1), TensorParallel(world_size=1, rank=0))
    config = _config_with_experts_per_token(2, packed=False)
    weights = ReferenceWeights(loader)
    weights.dispatch_stats = LinearDispatchStats()

    with TpRuntime(TpLaunchConfig(backend="gloo", device="cpu")) as runtime:
        tp_moe(hidden, mapping, config, weights, runtime)

    assert weights.dispatch_stats.moe_calls == 1
    assert weights.dispatch_stats.moe_packed_calls == 0
    assert weights.dispatch_stats.moe_loop_calls == 1
    assert weights.dispatch_stats.moe_assignments == 2
    assert weights.dispatch_stats.moe_local_assignments == 0


def test_packed_tp_moe_accumulates_duplicate_token_assignments_with_single_scatter() -> None:
    hidden = torch.tensor([[[1.0, 0.0]]])
    loader = _FakeLoader(_moe_test_tensors())
    mapping = _moe_mapping((0, 1), TensorParallel(world_size=1, rank=0))
    packed_config = _config_with_experts_per_token(2, packed=True)
    loop_config = _config_with_experts_per_token(2, packed=False)
    weights = ReferenceWeights(loader)
    weights.dispatch_stats = LinearDispatchStats()

    with TpRuntime(TpLaunchConfig(backend="gloo", device="cpu")) as runtime:
        expected = tp_moe(hidden, mapping, loop_config, ReferenceWeights(loader), runtime)
        actual = tp_moe(hidden, mapping, packed_config, weights, runtime)

    torch.testing.assert_close(actual, expected)
    assert weights.dispatch_stats.moe_local_assignments == 2
    assert weights.dispatch_stats.moe_active_expert_groups == 2
    assert weights.dispatch_stats.moe_packed_index_add_calls == 1
    assert weights.dispatch_stats.moe_packed_single_scatter_calls == 1
    assert weights.dispatch_stats.moe_group_size_1 == 2


def test_two_rank_packed_tp_moe_matches_loop(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(
        _tp_moe_worker,
        args=(tmp_path,),
        nprocs=2,
        join=True,
    )


def test_two_rank_tp_decoder_layer_matches_dense_reference(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(
        _tp_decoder_layer_worker,
        args=(tmp_path,),
        nprocs=2,
        join=True,
    )


def test_two_rank_safetensors_tp_language_model_matches_dense(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(
        _tp_language_model_worker,
        args=(tmp_path,),
        nprocs=2,
        join=True,
    )



def test_two_rank_safetensors_tp_decode_matches_prefill(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(
        _tp_decode_worker,
        args=(tmp_path,),
        nprocs=2,
        join=True,
    )


def test_paged_attention_dispatch_records_disabled_fallback() -> None:
    config = _config_with_experts_per_token(1, packed=True)
    with TpRuntime(TpLaunchConfig(backend="gloo", device="cpu")) as runtime:
        _record_paged_attention_dispatch(
            runtime,
            config,
            torch.zeros((1, 1, 1, 1)),
            torch.zeros((1, 1, 1, 1)),
            torch.zeros((1, 1, 1, 1)),
            seq_len=1,
            batched=False,
        )

    stats = runtime.paged_attention_stats
    assert stats.calls == 1
    assert stats.dense_fallbacks == 1
    assert stats.fallback_disabled == 1
    assert stats.eligible == 0
    assert stats.native_hits == 0


def test_paged_attention_dispatch_records_cpu_fallback_when_enabled() -> None:
    config = parse_runtime_config(_config())
    config = replace(config, full_attention=replace(config.full_attention, paged_kv_metadata=True, native_paged_attention=True))
    with TpRuntime(TpLaunchConfig(backend="gloo", device="cpu")) as runtime:
        _record_paged_attention_dispatch(
            runtime,
            config,
            torch.zeros((1, 1, 1, 1)),
            torch.zeros((1, 1, 1, 1)),
            torch.zeros((1, 1, 1, 1)),
            seq_len=1,
            batched=False,
        )

    stats = runtime.paged_attention_stats
    assert stats.calls == 1
    assert stats.dense_fallbacks == 1
    assert stats.fallback_cpu == 1
    assert stats.eligible == 0
    assert stats.native_hits == 0


def test_paged_attention_dispatch_records_prefill_or_multitoken_fallback_when_enabled() -> None:
    config = parse_runtime_config(_config())
    config = replace(config, full_attention=replace(config.full_attention, paged_kv_metadata=True, native_paged_attention=True))
    with TpRuntime(TpLaunchConfig(backend="gloo", device="cpu")) as runtime:
        _record_paged_attention_dispatch(
            runtime,
            config,
            _fake_cuda_tensor((1, 1, 2, 1)),
            _fake_cuda_tensor((1, 1, 2, 1)),
            _fake_cuda_tensor((1, 1, 2, 1)),
            seq_len=2,
            batched=False,
        )

    stats = runtime.paged_attention_stats
    assert stats.calls == 1
    assert stats.dense_fallbacks == 1
    assert stats.fallback_prefill_or_multitoken == 1
    assert stats.eligible == 0
    assert stats.native_hits == 0


def test_paged_attention_dispatch_records_per_request_pool_fallback_for_batches() -> None:
    config = parse_runtime_config(_config())
    config = replace(config, full_attention=replace(config.full_attention, paged_kv_metadata=True, native_paged_attention=True))
    with TpRuntime(TpLaunchConfig(backend="gloo", device="cpu")) as runtime:
        _record_paged_attention_dispatch(
            runtime,
            config,
            _fake_cuda_tensor((2, 1, 1, 1)),
            _fake_cuda_tensor((2, 1, 1, 1)),
            _fake_cuda_tensor((2, 1, 1, 1)),
            seq_len=1,
            batched=True,
        )

    stats = runtime.paged_attention_stats
    assert stats.calls == 1
    assert stats.dense_fallbacks == 1
    assert stats.fallback_per_request_pools == 1
    assert stats.eligible == 0
    assert stats.native_hits == 0

def test_tp_greedy_next_tokens_single_rank_batch_matches_argmax() -> None:
    runtime = TpRuntime(TpLaunchConfig(world_size=1, rank=0, local_rank=0, backend="gloo", device="cpu"))
    tp = TensorParallel(world_size=1, rank=0)
    lm_head = ShardedTensor(_info("head", (4, 2), nbytes=16), TensorShard.dim_shard("parallel_head", (4, 2), dim=0, tp=tp))
    logits = torch.tensor(
        [
            [[0.0, 1.0, 4.0, 3.0], [2.0, 0.0, 1.0, 3.0]],
            [[5.0, 1.0, 0.0, 2.0], [0.0, 8.0, 2.0, 1.0]],
        ]
    )

    actual = tp_greedy_next_tokens(logits, lm_head, runtime)
    expected = torch.argmax(logits[:, -1], dim=-1)

    torch.testing.assert_close(actual, expected)


def test_two_rank_tp_greedy_next_token_matches_full_gather_argmax(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(
        _tp_greedy_next_token_worker,
        args=(tmp_path,),
        nprocs=2,
        join=True,
    )


def test_two_rank_tp_greedy_next_tokens_matches_full_gather_argmax(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(
        _tp_greedy_next_tokens_worker,
        args=(tmp_path,),
        nprocs=2,
        join=True,
    )


def _tp_moe_worker(rank: int, tmp_path: Path) -> None:
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    loader = _FakeLoader(_moe_test_tensors())
    packed_config = _config_with_experts_per_token(2, packed=True)
    loop_config = _config_with_experts_per_token(2, packed=False)
    tp_mapping = _moe_mapping((rank,), TensorParallel(world_size=2, rank=rank))
    init_method = f"file://{tmp_path / 'tp-moe-dist-init'}"

    with TpRuntime(TpLaunchConfig(world_size=2, rank=rank, local_rank=rank, backend="gloo", init_method=init_method, device="cpu")) as runtime:
        expected = tp_moe(hidden, tp_mapping, loop_config, ReferenceWeights(loader), runtime)
        weights = ReferenceWeights(loader)
        weights.dispatch_stats = LinearDispatchStats()
        actual = tp_moe(hidden, tp_mapping, packed_config, weights, runtime)

    torch.testing.assert_close(actual, expected)
    assert weights.dispatch_stats.moe_calls == 1
    assert weights.dispatch_stats.moe_packed_calls == 1
    assert weights.dispatch_stats.moe_loop_calls == 0
    assert weights.dispatch_stats.moe_assignments == 6
    assert weights.dispatch_stats.moe_local_assignments == 3
    assert weights.dispatch_stats.moe_active_expert_groups == 1
    assert weights.dispatch_stats.moe_empty_local_dispatches == 0
    assert weights.dispatch_stats.moe_max_group_tokens == 3
    assert weights.dispatch_stats.moe_packed_index_add_calls == 1
    assert weights.dispatch_stats.moe_packed_single_scatter_calls == 1
    assert weights.dispatch_stats.moe_group_size_2_to_4 == 1
    assert weights.dispatch_stats.moe_native_expert_calls == 1
    assert weights.dispatch_stats.moe_native_expert_hits == 0
    assert weights.dispatch_stats.moe_native_expert_fallbacks == 1
    assert weights.dispatch_stats.moe_native_expert_fallback_device == 1


def _tp_greedy_lm_head(rank: int) -> ShardedTensor:
    tp = TensorParallel(world_size=2, rank=rank)
    return ShardedTensor(
        _info("head", (4, 2), nbytes=16),
        TensorShard.dim_shard("parallel_head", (4, 2), dim=0, tp=tp),
    )


def _tp_greedy_next_token_worker(rank: int, tmp_path: Path) -> None:
    init_method = f"file://{tmp_path / 'tp-greedy-dist-init'}"
    with TpRuntime(TpLaunchConfig(world_size=2, rank=rank, local_rank=rank, backend="gloo", init_method=init_method, device="cpu")) as runtime:
        local_logits = (
            torch.tensor([[[1.0, 4.0], [0.0, 3.0]]])
            if rank == 0
            else torch.tensor([[[2.0, 3.0], [5.0, 1.0]]])
        )
        full_logits = runtime.all_gather_cat(local_logits, dim=-1)
        expected = torch.argmax(full_logits[:, -1], dim=-1)
        actual = tp_greedy_next_token(local_logits, _tp_greedy_lm_head(rank), runtime)
        torch.testing.assert_close(actual, expected)


def _tp_greedy_next_tokens_worker(rank: int, tmp_path: Path) -> None:
    init_method = f"file://{tmp_path / 'tp-greedy-batch-dist-init'}"
    with TpRuntime(TpLaunchConfig(world_size=2, rank=rank, local_rank=rank, backend="gloo", init_method=init_method, device="cpu")) as runtime:
        local_logits = (
            torch.tensor([
                [[1.0, 4.0], [0.0, 3.0]],
                [[5.0, 1.0], [2.0, 6.0]],
            ])
            if rank == 0
            else torch.tensor([
                [[2.0, 3.0], [5.0, 1.0]],
                [[0.0, 9.0], [8.0, 1.0]],
            ])
        )
        full_logits = runtime.all_gather_cat(local_logits, dim=-1)
        expected = torch.argmax(full_logits[:, -1], dim=-1)
        actual = tp_greedy_next_tokens(local_logits, _tp_greedy_lm_head(rank), runtime)
        torch.testing.assert_close(actual, expected)


def _moe_test_tensors() -> dict[str, torch.Tensor]:
    return {
        "gate": torch.tensor([[5.0, 0.0], [0.0, 5.0]]),
        "shared_gate": torch.tensor([[-100.0, -100.0]]),
        "e0_gate": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "e0_up": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "e0_down": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "e1_gate": torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        "e1_up": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "e1_down": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "shared_gate_proj": torch.zeros((2, 2)),
        "shared_up_proj": torch.zeros((2, 2)),
        "shared_down_proj": torch.zeros((2, 2)),
    }


def _config_with_experts_per_token(experts_per_token: int, *, packed: bool, native: bool = True) -> object:
    config = parse_runtime_config(_config())
    return replace(
        config,
        moe=replace(
            config.moe,
            experts_per_token=experts_per_token,
            packed_expert_dispatch=packed,
            native_fused_expert_dispatch=native,
        ),
    )


def _fake_cuda_tensor(shape: tuple[int, ...]) -> SimpleNamespace:
    return SimpleNamespace(shape=shape, is_cuda=True)

def _tp_decode_worker(rank: int, tmp_path: Path) -> None:
    save_file = pytest.importorskip("safetensors.torch").save_file
    model_dir = tmp_path / "tiny-decode"
    if rank == 0:
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps(_safetensors_config()), encoding="utf-8")
        save_file(_safetensors_tensors(), model_dir / "model.safetensors")
    init_method = f"file://{tmp_path / 'tp-decode-dist-init'}"
    with TpRuntime(TpLaunchConfig(world_size=2, rank=rank, local_rank=rank, backend="gloo", init_method=init_method, device="cpu")) as runtime:
        runtime.barrier()
        manifest = build_manifest(model_dir)
        config = parse_runtime_config(manifest.config)
        tp_mapping = build_language_model_mapping(manifest, strict=True, tensor_parallel=TensorParallel(world_size=2, rank=rank))
        input_ids = torch.tensor([[0, 1, 2, 3]])
        with TensorLoader(manifest) as loader:
            weights = ReferenceWeights(loader)
            prefill = tp_language_model(input_ids, tp_mapping, config, weights, runtime)
            state = DecodeState.empty(tp_mapping, config)
            step_logits = [
                tp_decode_step(input_ids[:, :3], tp_mapping, config, weights, runtime, state),
                tp_decode_step(input_ids[:, 3:4], tp_mapping, config, weights, runtime, state),
            ]
            decoded = torch.cat(step_logits, dim=1)
        torch.testing.assert_close(decoded, prefill, atol=1e-5, rtol=1e-5)


def _tp_language_model_worker(rank: int, tmp_path: Path) -> None:
    save_file = pytest.importorskip("safetensors.torch").save_file
    model_dir = tmp_path / "tiny-full"
    if rank == 0:
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps(_safetensors_config()), encoding="utf-8")
        save_file(_safetensors_tensors(), model_dir / "model.safetensors")
    init_method = f"file://{tmp_path / 'safetensors-dist-init'}"
    with TpRuntime(TpLaunchConfig(world_size=2, rank=rank, local_rank=rank, backend="gloo", init_method=init_method, device="cpu")) as runtime:
        runtime.barrier()
        manifest = build_manifest(model_dir)
        config = parse_runtime_config(manifest.config)
        dense_mapping = build_language_model_mapping(manifest, strict=True)
        tp_mapping = build_language_model_mapping(manifest, strict=True, tensor_parallel=TensorParallel(world_size=2, rank=rank))
        input_ids = torch.tensor([[0, 3]])
        with TensorLoader(manifest) as loader:
            expected = language_model(input_ids, dense_mapping, config, ReferenceWeights(loader))
            actual = tp_language_model(input_ids, tp_mapping, config, ReferenceWeights(loader), runtime)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)



def _tp_decoder_layer_worker(rank: int, tmp_path: Path) -> None:
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    q = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    o = torch.eye(2)
    loader = _FakeLoader(
        {
            "input_norm": torch.zeros(2),
            "q": q,
            "q_r0": q[:2].contiguous(),
            "q_r1": q[2:].contiguous(),
            "k": torch.tensor([[1.0, 0.0]]),
            "v": torch.tensor([[0.0, 1.0]]),
            "o": o,
            "o_r0": o[:, :1].contiguous(),
            "o_r1": o[:, 1:].contiguous(),
            "q_norm": torch.zeros(1),
            "k_norm": torch.zeros(1),
            "post_norm": torch.zeros(2),
            "gate": torch.tensor([[5.0, 0.0], [0.0, 5.0]]),
            "shared_gate": torch.tensor([[-100.0, -100.0]]),
            "e0_gate": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "e0_up": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "e0_down": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "e1_gate": torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            "e1_up": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "e1_down": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "shared_gate_proj": torch.zeros((2, 2)),
            "shared_up_proj": torch.zeros((2, 2)),
            "shared_down_proj": torch.zeros((2, 2)),
        }
    )
    config = parse_runtime_config(_config())
    init_method = f"file://{tmp_path / 'dist-init'}"
    dense_mapping = _layer_mapping((0, 1), TensorParallel(world_size=1, rank=0))
    tp_mapping = _layer_mapping((rank,), TensorParallel(world_size=2, rank=rank))

    with TpRuntime(TpLaunchConfig(world_size=2, rank=rank, local_rank=rank, backend="gloo", init_method=init_method, device="cpu")) as runtime:
        tp_out = tp_decoder_layer(hidden, tp_mapping, config, ReferenceWeights(loader), runtime)
        expected = decoder_layer(hidden, dense_mapping, config, ReferenceWeights(loader))
        torch.testing.assert_close(tp_out, expected)


def _layer_mapping(local_experts: tuple[int, ...], tp: TensorParallel) -> LayerMapping:
    return LayerMapping(
        index=0,
        layer_type="full_attention",
        input_layernorm=_info("input_norm", (2,), nbytes=4),
        attention=FullAttentionMapping(
            q_proj=_linear_shape("q" if tp.world_size == 1 else f"q_r{tp.rank}", (4 // tp.world_size, 2) if tp.world_size > 1 else (4, 2)),
            k_proj=_linear_shape("k", (1, 2)),
            v_proj=_linear_shape("v", (1, 2)),
            o_proj=_linear_shape("o" if tp.world_size == 1 else f"o_r{tp.rank}", (2, 2 // tp.world_size) if tp.world_size > 1 else (2, 2)),
            q_norm=_info("q_norm", (1,), nbytes=2),
            k_norm=_info("k_norm", (1,), nbytes=2),
        ),
        post_attention_layernorm=_info("post_norm", (2,), nbytes=4),
        mlp=_moe_mapping(local_experts, tp),
    )


def _moe_mapping(local_experts: tuple[int, ...], tp: TensorParallel) -> MoEMapping:
    names = {0: "e0", 1: "e1"}
    return MoEMapping(
        gate=_info("gate", (2, 2), nbytes=8),
        experts=tuple(
            ExpertMapping(i, _linear(f"{names[i]}_gate"), _linear(f"{names[i]}_up"), _linear(f"{names[i]}_down"))
            for i in local_experts
        ),
        shared_expert=ExpertMapping(-1, _linear("shared_gate_proj"), _linear("shared_up_proj"), _linear("shared_down_proj")),
        shared_expert_gate=_info("shared_gate", (1, 2), nbytes=4),
        expert_start=local_experts[0],
        expert_end=local_experts[-1] + 1,
        num_experts=2,
        tp=tp,
    )


def _linear(name: str) -> LinearTensor:
    return _linear_shape(name, (2, 2))


def _linear_shape(name: str, shape: tuple[int, int]) -> LinearTensor:
    return LinearTensor(weight=_info(name, shape, nbytes=8), scale=None)


def _info(name: str, shape: tuple[int, ...], *, nbytes: int) -> TensorInfo:
    return TensorInfo(name=name, dtype="BF16", shape=shape, shard="model.safetensors", begin=0, end=nbytes, data_start=0)


class _FakeLoader:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def tensor(self, name: str, *, device: str | None = None) -> torch.Tensor:
        tensor = self.tensors[name]
        return tensor if device is None else tensor.to(device)


def _config() -> dict[str, object]:
    return {
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2,
            "vocab_size": 4,
            "num_hidden_layers": 1,
            "layer_types": ["full_attention"],
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 1,
            "linear_value_head_dim": 1,
            "linear_conv_kernel_dim": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 1,
            "attn_output_gate": True,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 2,
            "shared_expert_intermediate_size": 2,
            "max_position_embeddings": 8,
            "rms_norm_eps": 1e-6,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {"rope_theta": 10000},
        }
    }



def _safetensors_config() -> dict[str, object]:
    return {
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 4,
            "vocab_size": 4,
            "num_hidden_layers": 1,
            "layer_types": ["full_attention"],
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 2,
            "linear_value_head_dim": 2,
            "linear_conv_kernel_dim": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "attn_output_gate": True,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 4,
            "shared_expert_intermediate_size": 4,
            "max_position_embeddings": 8,
            "rms_norm_eps": 1e-6,
            "partial_rotary_factor": 1.0,
            "rope_parameters": {"rope_theta": 10000},
        }
    }


def _safetensors_tensors() -> dict[str, torch.Tensor]:
    p = "model.language_model.layers.0."
    q_proj = torch.zeros((8, 4), dtype=torch.bfloat16)
    q_proj[0, 0] = 1.0
    q_proj[1, 1] = 1.0
    q_proj[4, 2] = 1.0
    q_proj[5, 3] = 1.0
    k_proj = torch.zeros((2, 4), dtype=torch.bfloat16)
    k_proj[0, 0] = 1.0
    k_proj[1, 1] = 1.0
    v_proj = torch.zeros((2, 4), dtype=torch.bfloat16)
    v_proj[0, 2] = 1.0
    v_proj[1, 3] = 1.0
    tensors: dict[str, torch.Tensor] = {
        "model.language_model.embed_tokens.weight": torch.tensor(
            [[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5], [1.0, 1.0, 0.0, 0.0], [-1.0, 0.5, 1.0, 0.0]],
            dtype=torch.bfloat16,
        ),
        "model.language_model.norm.weight": torch.zeros(4, dtype=torch.bfloat16),
        "lm_head.weight": torch.tensor(
            [[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5], [1.0, 1.0, 0.0, 0.0], [-1.0, 0.5, 1.0, 0.0]],
            dtype=torch.bfloat16,
        ),
        p + "input_layernorm.weight": torch.zeros(4, dtype=torch.bfloat16),
        p + "post_attention_layernorm.weight": torch.zeros(4, dtype=torch.bfloat16),
        p + "self_attn.q_proj.weight": q_proj,
        p + "self_attn.k_proj.weight": k_proj,
        p + "self_attn.v_proj.weight": v_proj,
        p + "self_attn.o_proj.weight": torch.eye(4, dtype=torch.bfloat16),
        p + "self_attn.q_norm.weight": torch.zeros(2, dtype=torch.bfloat16),
        p + "self_attn.k_norm.weight": torch.zeros(2, dtype=torch.bfloat16),
        p + "mlp.gate.weight": torch.tensor([[5.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0]], dtype=torch.bfloat16),
        p + "mlp.shared_expert_gate.weight": torch.full((1, 4), -100.0, dtype=torch.bfloat16),
        p + "mlp.shared_expert.gate_proj.weight": torch.zeros((4, 4), dtype=torch.bfloat16),
        p + "mlp.shared_expert.up_proj.weight": torch.zeros((4, 4), dtype=torch.bfloat16),
        p + "mlp.shared_expert.down_proj.weight": torch.zeros((4, 4), dtype=torch.bfloat16),
    }
    for expert, scale in ((0, 1.0), (1, 2.0)):
        prefix = p + f"mlp.experts.{expert}."
        tensors[prefix + "gate_proj.weight"] = torch.eye(4, dtype=torch.bfloat16) * scale
        tensors[prefix + "up_proj.weight"] = torch.eye(4, dtype=torch.bfloat16)
        tensors[prefix + "down_proj.weight"] = torch.eye(4, dtype=torch.bfloat16)
    fp8_names = [
        p + "self_attn.q_proj.weight",
        p + "self_attn.k_proj.weight",
        p + "self_attn.v_proj.weight",
        p + "self_attn.o_proj.weight",
        p + "mlp.shared_expert.gate_proj.weight",
        p + "mlp.shared_expert.up_proj.weight",
        p + "mlp.shared_expert.down_proj.weight",
    ]
    fp8_names.extend(
        p + f"mlp.experts.{expert}.{suffix}.weight"
        for expert in (0, 1)
        for suffix in ("gate_proj", "up_proj", "down_proj")
    )
    for name in fp8_names:
        tensors[name] = tensors[name].to(torch.float8_e4m3fn)
        rows, cols = tensors[name].shape
        tensors[name + "_scale_inv"] = torch.ones(((rows + 127) // 128, (cols + 127) // 128), dtype=torch.bfloat16)
    return tensors
