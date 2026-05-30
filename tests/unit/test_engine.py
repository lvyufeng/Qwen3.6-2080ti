from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from checkpoint import build_manifest
from engine import EngineError, TpModelRunner, TpModelSession
import engine
from runtime_config import parse_runtime_config
from tp_runtime import TpLaunchConfig
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
    assert result.all_finite is True
    assert len(result.generated_token_ids) == 2
    assert result.text == "decoded:" + ",".join(str(token) for token in result.generated_token_ids)
    assert result.cuda_memory.available is False


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
