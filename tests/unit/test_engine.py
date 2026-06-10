from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from checkpoint import build_manifest
from decode_state import FullAttentionCache
from engine import EngineError, TpModelRunner, TpModelSession
import engine
from runtime_config import parse_runtime_config
from tp_runtime import RuntimeProfileConfig, TpLaunchConfig
from test_cli import _runtime_config
from test_weight_mapping import add_full_attention_layer, add_moe, write_safetensors


def test_tp_model_runner_generate_single_rank_cpu_gloo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    manifest = build_manifest(tmp_path)
    runner = TpModelRunner(manifest, parse_runtime_config(manifest.config), TpLaunchConfig(backend="gloo", device="cpu"))

    result = runner.generate("hello", max_new_tokens=2)

    assert result.world_size == 1
    assert result.rank == 0
    assert result.local_rank == 0
    assert str(result.device) == "cpu"
    assert result.prompt_tokens == 2
    assert result.max_new_tokens == 2
    assert result.layers == 1
    assert result.mapped_tensors > 0
    assert result.mapped_bytes > 0
    assert result.load_stats.tensor_count > 0
    assert result.load_stats.bytes > 0
    assert result.load_seconds >= 0
    assert result.prefill_seconds >= 0
    assert result.decode_seconds >= 0
    assert result.total_seconds >= 0
    assert result.decode_tokens_per_second >= 0
    assert result.total_tokens_per_second >= 0
    assert result.dispatch_stats.calls > 0
    assert result.fp8_native_stats["dense_linear_calls"] == 0
    assert "moe_expert_tensor_core_hits" in result.fp8_native_stats
    assert result.all_finite is True
    assert len(result.generated_token_ids) == 2
    assert result.text == "decoded:" + ",".join(str(token) for token in result.generated_token_ids)
    assert result.cuda_memory.available is False
    assert result.profile.enabled is False
    assert result.profile.scopes == {}
    assert result.kv_cache.full_attention_layers == 1
    assert result.kv_cache.valid_tokens_total > 0
    assert result.kv_cache.capacity_tokens_total >= result.kv_cache.valid_tokens_total
    assert result.cuda_graph_probe.enabled is False
    assert result.cuda_graph_probe.eligible is False


def test_tp_model_session_cuda_graph_probe_reports_blockers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    manifest = build_manifest(tmp_path)
    runner = TpModelRunner(manifest, parse_runtime_config(manifest.config), TpLaunchConfig(backend="gloo", device="cpu"))

    result = runner.generate("hello", max_new_tokens=2, cuda_graph_probe=True)

    assert result.cuda_graph_probe.enabled is True
    assert result.cuda_graph_probe.eligible is False
    assert "native_paged_attention_required" in result.cuda_graph_probe.reasons
    assert "moe_dynamic_route_dispatch" in result.cuda_graph_probe.reasons
    assert "probe_only_no_cuda_graph_capture_attempted" in result.cuda_graph_probe.notes


def test_tp_model_session_profile_enabled_records_scopes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    manifest = build_manifest(tmp_path)

    with TpModelSession(
        manifest,
        parse_runtime_config(manifest.config),
        TpLaunchConfig(backend="gloo", device="cpu"),
        profile_config=RuntimeProfileConfig(enabled=True),
    ) as session:
        result = session.generate("hello", max_new_tokens=1)

    assert result.profile.enabled is True
    assert "embedding" in result.profile.scopes
    assert "layers_total" in result.profile.scopes
    assert "lm_head" in result.profile.scopes



def test_tp_model_session_reuses_loaded_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    manifest = build_manifest(tmp_path)
    preload_calls = 0
    original_preload = engine.MappedWeights.preload

    def counted_preload(self):
        nonlocal preload_calls
        preload_calls += 1
        return original_preload(self)

    monkeypatch.setattr(engine.MappedWeights, "preload", counted_preload)

    with TpModelSession(manifest, parse_runtime_config(manifest.config), TpLaunchConfig(backend="gloo", device="cpu")) as session:
        first = session.generate("hello", max_new_tokens=1)
        second = session.generate("hello again", max_new_tokens=1)

    assert preload_calls == 1
    assert first.load_stats.tensor_count == second.load_stats.tensor_count
    assert first.load_seconds == second.load_seconds
    assert first.dispatch_stats.calls > 0
    assert second.dispatch_stats.calls > 0


@pytest.mark.parametrize(
    ("input_ids", "message"),
    [
        (torch.empty((1, 0), dtype=torch.long), "requires at least one prompt token"),
        (torch.tensor([[1], [2]]), "supports exactly one prompt"),
    ],
)
def test_tp_model_runner_validates_prompt_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_ids: torch.Tensor,
    message: str,
) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, input_ids)
    manifest = build_manifest(tmp_path)
    runner = TpModelRunner(manifest, parse_runtime_config(manifest.config), TpLaunchConfig(backend="gloo", device="cpu"))

    with pytest.raises(EngineError, match=message):
        runner.generate("hello", max_new_tokens=1)


def test_tp_model_session_steps_generation_to_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    manifest = build_manifest(tmp_path)

    with TpModelSession(manifest, parse_runtime_config(manifest.config), TpLaunchConfig(backend="gloo", device="cpu")) as session:
        state = session.start_generation("hello", max_new_tokens=2)

        assert state.prompt == "hello"
        assert state.max_new_tokens == 2
        assert state.prompt_tokens == 2
        assert state.generated_token_ids == []
        assert state.completed is False
        assert state.prefill_seconds >= 0

        first = session.step_generation(state)
        assert first.index == 0
        assert first.is_complete is False
        assert len(state.generated_token_ids) == 1
        assert first.token_id == state.generated_token_ids[0]
        assert first.prefill_seconds >= 0
        assert first.decode_seconds >= 0
        assert first.total_seconds >= 0

        second = session.step_generation(state)
        assert second.index == 1
        assert second.is_complete is True
        assert state.completed is True
        assert len(state.generated_token_ids) == 2
        assert second.token_id == state.generated_token_ids[1]

        result = session.finish_generation(state)
        full_cache = state.decode_state.layers[0]
        assert isinstance(full_cache, FullAttentionCache)
        assert full_cache.length == 0
        assert full_cache.block_table == []

    assert result.prompt_tokens == 2
    assert result.max_new_tokens == 2
    assert result.generated_token_ids == state.generated_token_ids
    assert result.text == "decoded:" + ",".join(str(token) for token in state.generated_token_ids)
    assert result.all_finite is True
    assert result.prefill_seconds >= 0
    assert result.decode_seconds >= 0
    assert result.total_seconds >= 0
    assert result.decode_tokens_per_second >= 0
    assert result.total_tokens_per_second >= 0


def test_tp_model_session_generate_matches_manual_stepping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    manifest = build_manifest(tmp_path)

    with TpModelSession(manifest, parse_runtime_config(manifest.config), TpLaunchConfig(backend="gloo", device="cpu")) as session:
        state = session.start_generation("hello", max_new_tokens=2)
        while not state.completed:
            session.step_generation(state)
        manual = session.finish_generation(state)
        wrapped = session.generate("hello", max_new_tokens=2)

    assert wrapped.backend == manual.backend
    assert wrapped.world_size == manual.world_size
    assert wrapped.rank == manual.rank
    assert wrapped.local_rank == manual.local_rank
    assert str(wrapped.device) == str(manual.device)
    assert wrapped.prompt_tokens == manual.prompt_tokens
    assert wrapped.max_new_tokens == manual.max_new_tokens
    assert wrapped.layers == manual.layers
    assert wrapped.mapped_tensors == manual.mapped_tensors
    assert wrapped.mapped_bytes == manual.mapped_bytes
    assert wrapped.generated_token_ids == manual.generated_token_ids
    assert wrapped.text == manual.text
    assert wrapped.all_finite == manual.all_finite


def test_tp_model_session_step_after_completion_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    manifest = build_manifest(tmp_path)

    with TpModelSession(manifest, parse_runtime_config(manifest.config), TpLaunchConfig(backend="gloo", device="cpu")) as session:
        state = session.start_generation("hello", max_new_tokens=1)
        step = session.step_generation(state)
        assert step.is_complete is True

        with pytest.raises(EngineError, match="complete"):
            session.step_generation(state)


def test_tp_model_session_finish_before_completion_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    manifest = build_manifest(tmp_path)

    with TpModelSession(manifest, parse_runtime_config(manifest.config), TpLaunchConfig(backend="gloo", device="cpu")) as session:
        state = session.start_generation("hello", max_new_tokens=2)
        session.step_generation(state)

        with pytest.raises(EngineError, match="not complete"):
            session.finish_generation(state)


def test_tp_model_session_finish_twice_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    manifest = build_manifest(tmp_path)

    with TpModelSession(manifest, parse_runtime_config(manifest.config), TpLaunchConfig(backend="gloo", device="cpu")) as session:
        state = session.start_generation("hello", max_new_tokens=1)
        session.step_generation(state)
        session.finish_generation(state)

        with pytest.raises(EngineError, match="finalized"):
            session.finish_generation(state)


def test_tp_model_session_final_token_does_not_decode_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    manifest = build_manifest(tmp_path)
    original_decode = engine.tp_decode_step_local_logits
    shapes: list[tuple[int, ...]] = []

    def counted_decode(input_ids, *args, **kwargs):
        shapes.append(tuple(input_ids.shape))
        return original_decode(input_ids, *args, **kwargs)

    monkeypatch.setattr(engine, "tp_decode_step_local_logits", counted_decode)

    with TpModelSession(manifest, parse_runtime_config(manifest.config), TpLaunchConfig(backend="gloo", device="cpu")) as session:
        session.generate("hello", max_new_tokens=1)
        assert shapes == [(1, 2)]

        shapes.clear()
        session.generate("hello", max_new_tokens=2)
        assert shapes == [(1, 2), (1, 1)]


def test_tp_model_session_zero_token_generate_still_finishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    manifest = build_manifest(tmp_path)

    with TpModelSession(manifest, parse_runtime_config(manifest.config), TpLaunchConfig(backend="gloo", device="cpu")) as session:
        result = session.generate("hello", max_new_tokens=0)

    assert result.prompt_tokens == 2
    assert result.max_new_tokens == 0
    assert result.generated_token_ids == []
    assert result.text == "decoded:"
    assert result.all_finite is True
    assert result.prefill_seconds >= 0
    assert result.decode_seconds >= 0
    assert result.total_seconds >= 0


def test_step_generations_batch_produces_same_tokens_as_sequential(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Batch step of two requests produces the same token sequence as stepping each independently."""
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    manifest = build_manifest(tmp_path)

    with TpModelSession(manifest, parse_runtime_config(manifest.config), TpLaunchConfig(backend="gloo", device="cpu")) as session:
        # Sequential: step each request independently
        state_a_seq = session.start_generation("hello", max_new_tokens=3)
        state_b_seq = session.start_generation("hello", max_new_tokens=3)
        for _ in range(3):
            session.step_generation(state_a_seq)
        for _ in range(3):
            session.step_generation(state_b_seq)

        # Batch: step two requests together
        state_a_batch = session.start_generation("hello", max_new_tokens=3)
        state_b_batch = session.start_generation("hello", max_new_tokens=3)
        for _ in range(3):
            steps = session.step_generations_batch([state_a_batch, state_b_batch])
            assert len(steps) == 2
        result = session.finish_generation(state_a_batch)

    # Both paths should produce the same tokens (deterministic greedy decode)
    assert state_a_batch.generated_token_ids == state_a_seq.generated_token_ids
    assert state_b_batch.generated_token_ids == state_b_seq.generated_token_ids
    assert result.kv_cache.full_attention_layers == 1
    assert result.kv_cache.valid_tokens_total > 0


def test_step_generations_batch_single_request_delegates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With one state, step_generations_batch delegates to step_generation."""
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    manifest = build_manifest(tmp_path)

    with TpModelSession(manifest, parse_runtime_config(manifest.config), TpLaunchConfig(backend="gloo", device="cpu")) as session:
        state_single = session.start_generation("hello", max_new_tokens=2)
        state_batch = session.start_generation("hello", max_new_tokens=2)

        # Single step_generation
        step1 = session.step_generation(state_single)
        step2 = session.step_generation(state_single)

        # Batch with single element
        batch_steps = []
        batch_steps.append(session.step_generations_batch([state_batch])[0])
        batch_steps.append(session.step_generations_batch([state_batch])[0])

    assert state_single.generated_token_ids == state_batch.generated_token_ids
    assert batch_steps[0].token_id == step1.token_id
    assert batch_steps[1].token_id == step2.token_id
    assert batch_steps[1].is_complete is True


def test_step_generations_batch_handles_mixed_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When one request completes before another, batch step handles it correctly."""
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    manifest = build_manifest(tmp_path)

    with TpModelSession(manifest, parse_runtime_config(manifest.config), TpLaunchConfig(backend="gloo", device="cpu")) as session:
        # state_a completes after 1 token, state_b needs 3
        state_a = session.start_generation("hello", max_new_tokens=1)
        state_b = session.start_generation("hello", max_new_tokens=3)

        steps = session.step_generations_batch([state_a, state_b])

    assert steps[0].is_complete is True
    assert steps[1].is_complete is False
    assert state_a.completed is True
    assert state_b.completed is False
    assert len(state_a.generated_token_ids) == 1
    assert len(state_b.generated_token_ids) == 1


def test_step_generations_batch_uses_batched_greedy_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    manifest = build_manifest(tmp_path)
    original = engine.tp_greedy_next_tokens
    call_batches: list[int] = []

    def counted_greedy(logits, lm_head, runtime):
        call_batches.append(int(logits.shape[0]))
        return original(logits, lm_head, runtime)

    monkeypatch.setattr(engine, "tp_greedy_next_tokens", counted_greedy)

    with TpModelSession(manifest, parse_runtime_config(manifest.config), TpLaunchConfig(backend="gloo", device="cpu")) as session:
        state_a = session.start_generation("hello", max_new_tokens=1)
        state_b = session.start_generation("hello", max_new_tokens=1)
        steps = session.step_generations_batch([state_a, state_b])

    assert [step.token_id for step in steps] == state_a.generated_token_ids + state_b.generated_token_ids
    assert call_batches == [2]


def test_step_generations_batch_skips_redundant_token_broadcast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    manifest = build_manifest(tmp_path)
    sync_calls = 0

    def counted_sync(next_token, runtime):
        nonlocal sync_calls
        sync_calls += 1
        return next_token

    monkeypatch.setattr(engine, "_sync_next_token", counted_sync)

    with TpModelSession(manifest, parse_runtime_config(manifest.config), TpLaunchConfig(backend="gloo", device="cpu")) as session:
        state_a = session.start_generation("hello", max_new_tokens=1)
        state_b = session.start_generation("hello", max_new_tokens=1)
        session.step_generations_batch([state_a, state_b])

    assert sync_calls == 0


def test_step_generations_batch_appends_blocks_without_extra_dense_view(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    manifest = build_manifest(tmp_path)

    with TpModelSession(manifest, parse_runtime_config(manifest.config), TpLaunchConfig(backend="gloo", device="cpu")) as session:
        state_a = session.start_generation("hello", max_new_tokens=2)
        state_b = session.start_generation("hello", max_new_tokens=2)
        cache_a = state_a.decode_state.layers[0]
        cache_b = state_b.decode_state.layers[0]
        assert isinstance(cache_a, FullAttentionCache)
        assert isinstance(cache_b, FullAttentionCache)
        before_a = cache_a.stats()
        before_b = cache_b.stats()

        session.step_generations_batch([state_a, state_b])

        after_a = cache_a.stats()
        after_b = cache_b.stats()

    assert after_a.append_calls == before_a.append_calls + 1
    assert after_b.append_calls == before_b.append_calls + 1
    assert after_a.contiguous_view_calls + after_a.gather_view_calls == before_a.contiguous_view_calls + before_a.gather_view_calls + 1
    assert after_b.contiguous_view_calls + after_b.gather_view_calls == before_b.contiguous_view_calls + before_b.gather_view_calls + 1



def test_tp_model_session_fast_decode_defers_step_syncs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    manifest = build_manifest(tmp_path)

    with TpModelSession(manifest, parse_runtime_config(manifest.config), TpLaunchConfig(backend="gloo", device="cpu")) as session:
        state = session.start_generation("hello", max_new_tokens=2, fast_decode=True)
        sync_device_calls = 0
        sync_token_calls = 0
        isfinite_calls = 0
        original_isfinite = torch.isfinite

        def counted_sync_device(device):
            nonlocal sync_device_calls
            sync_device_calls += 1

        def counted_sync_token(next_token, runtime):
            nonlocal sync_token_calls
            sync_token_calls += 1
            return next_token

        def counted_isfinite(*args, **kwargs):
            nonlocal isfinite_calls
            isfinite_calls += 1
            return original_isfinite(*args, **kwargs)

        monkeypatch.setattr(engine, "_sync_device", counted_sync_device)
        monkeypatch.setattr(engine, "_sync_next_token", counted_sync_token)
        monkeypatch.setattr(torch, "isfinite", counted_isfinite)

        first = session.step_generation(state)
        second = session.step_generation(state)

        assert first.token_id == -1
        assert second.token_id == -1
        assert state.generated_token_ids == [-1, -1]
        assert sync_device_calls == 0
        assert sync_token_calls == 0
        assert isfinite_calls == 0

        result = session.finish_generation(state)

    assert sync_device_calls == 1
    assert result.fast_decode is True
    assert result.all_finite is True
    assert len(result.generated_token_ids) == 2
    assert all(token_id >= 0 for token_id in result.generated_token_ids)
    assert result.text == "decoded:" + ",".join(str(token) for token in result.generated_token_ids)



def _write_tiny_model(model_dir: Path) -> None:
    config = _runtime_config()
    text = config["text_config"]
    text["hidden_size"] = 256
    text["vocab_size"] = 320
    text["num_hidden_layers"] = 1
    text["layer_types"] = ["full_attention"]
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "model.language_model.embed_tokens.weight": ("BF16", (320, 256)),
        "model.language_model.norm.weight": ("BF16", (256,)),
        "lm_head.weight": ("BF16", (320, 256)),
    }
    add_full_attention_layer(tensors, 0)
    add_moe(tensors, 0)
    write_safetensors(model_dir / "model.safetensors", tensors)


def _patch_tokenizer(monkeypatch: pytest.MonkeyPatch, input_ids: torch.Tensor) -> None:
    class FakeTokenizer:
        def __call__(self, prompt: str, *, return_tensors: str, add_special_tokens: bool):
            assert return_tensors == "pt"
            assert add_special_tokens is True
            return {"input_ids": input_ids}

        def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens is False
            return "decoded:" + ",".join(str(token_id) for token_id in token_ids)

    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_, **__: FakeTokenizer())
