from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import cli
from cli import CliError, _reference_layer, main
from test_weight_mapping import add_full_attention_layer, add_linear_attention_layer, add_moe, write_safetensors


def test_cli_summarizes_model_config(tmp_path: Path, capsys) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_moe",
                "hidden_size": 4096,
                "num_hidden_layers": 2,
                "num_experts": 128,
                "num_experts_per_tok": 8,
            }
        ),
        encoding="utf-8",
    )

    (tmp_path / "model.safetensors").write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")

    rc = main(["--model", str(tmp_path), "--prompt", "hello", "--max-new-tokens", "1"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "Loaded model config" in out
    assert "model_type: qwen3_moe" in out
    assert "num_experts: 128" in out
    assert "inference: not implemented yet" in out


def test_cli_inspects_runtime_config_and_checkpoint(tmp_path: Path, capsys) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_runtime_config()), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")

    rc = main(
        [
            "--model",
            str(tmp_path),
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
            "--inspect-config",
            "--inspect-checkpoint",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "Loaded runtime config" in out
    assert "runtime_linear_qkv_dim: 256" in out
    assert "Loaded checkpoint manifest" in out
    assert "tensor_count: 0" in out


def test_cli_inspects_tp4_mapping(tmp_path: Path, capsys) -> None:
    config = _runtime_config()
    text = config["text_config"]
    text["num_hidden_layers"] = 1
    text["layer_types"] = ["linear_attention"]
    text["num_experts"] = 8
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "model.language_model.embed_tokens.weight": ("BF16", (320, 256)),
        "model.language_model.norm.weight": ("BF16", (256,)),
        "lm_head.weight": ("BF16", (320, 256)),
    }
    add_linear_attention_layer(tensors, 0)
    add_moe(tensors, 0, num_experts=8)
    write_safetensors(tmp_path / "model.safetensors", tensors)

    rc = main(["--model", str(tmp_path), "--prompt", "hello", "--max-new-tokens", "1", "--inspect-tp", "4"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "Tensor-parallel mapping layout" in out
    assert "tp_rank_0: experts_per_layer=2 expert_range=[0,2)" in out
    assert "tp_rank_3: experts_per_layer=2 expert_range=[6,8)" in out
    assert "tp_rank_0_shard_embed_tokens=parallel_embedding:shape=(80, 256):dim=0:start=0:size=80" in out
    assert "tp_rank_3_shard_lm_head=parallel_head:shape=(80, 256):dim=0:start=240:size=80" in out
    assert "tp_rank_0_shard_in_proj_qkv=packed_qkv_column_parallel:shape=(64, 256):dim=0:segments=0+16,64+16,128+32" in out
    assert "tp_rank_3_shard_conv1d=packed_conv1d_channel_parallel:shape=(64, 1, 4):dim=0:segments=48+16,112+16,224+32" in out
    assert "tp_rank_0_shard_out_proj=row_parallel:shape=(256, 32):dim=1:start=0:size=32" in out
    assert "tp_rank_0_shard_shared_expert.gate_proj=replicated:shape=(128, 256)" in out
    assert "tp_dense_mapped_bytes:" in out
    assert "tp_partition_complete: True" in out


def test_cli_tp_load_smoke_single_rank(tmp_path: Path, capsys) -> None:
    config = _runtime_config()
    text = config["text_config"]
    text["num_hidden_layers"] = 1
    text["layer_types"] = ["linear_attention"]
    text["num_experts"] = 8
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "model.language_model.embed_tokens.weight": ("BF16", (320, 256)),
        "model.language_model.norm.weight": ("BF16", (256,)),
        "lm_head.weight": ("BF16", (320, 256)),
    }
    add_linear_attention_layer(tensors, 0)
    add_moe(tensors, 0, num_experts=8)
    write_safetensors(tmp_path / "model.safetensors", tensors)

    rc = main(
        [
            "--model",
            str(tmp_path),
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
            "--tp-load-smoke",
            "--tp-world-size",
            "1",
            "--tp-backend",
            "gloo",
            "--tp-device",
            "cpu",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "TP mapped weight load smoke" in out
    assert "tp_load_mapped_tensors:" in out
    assert "tp_load_loaded_tensors:" in out
    assert "tp_load_loaded_bytes:" in out
    assert "tp_load_shard_embed_tokens=parallel_embedding:shape=(320, 256):dim=0:start=0:size=320" in out
    assert "tp_load_shard_in_proj_qkv=packed_qkv_column_parallel:shape=(256, 256):dim=0:segments=0+64,64+64,128+128" in out
    assert "tp_load_shard_shared_expert.gate_proj=replicated:shape=(128, 256)" in out


def test_cli_tp_runtime_smoke_single_rank(tmp_path: Path, capsys) -> None:
    config = _runtime_config()
    text = config["text_config"]
    text["num_hidden_layers"] = 1
    text["layer_types"] = ["linear_attention"]
    text["num_experts"] = 8
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "model.language_model.embed_tokens.weight": ("BF16", (320, 256)),
        "model.language_model.norm.weight": ("BF16", (256,)),
        "lm_head.weight": ("BF16", (320, 256)),
    }
    add_linear_attention_layer(tensors, 0)
    add_moe(tensors, 0, num_experts=8)
    write_safetensors(tmp_path / "model.safetensors", tensors)

    rc = main(
        [
            "--model",
            str(tmp_path),
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
            "--tp-runtime-smoke",
            "--tp-world-size",
            "1",
            "--tp-backend",
            "gloo",
            "--tp-device",
            "cpu",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "TP runtime smoke" in out
    assert "tp_runtime_rank: 0" in out
    assert "tp_runtime_expert_range: [0,8)" in out
    assert "tp_runtime_experts_per_layer: 8" in out
    assert "tp_runtime_shard_embed_tokens=parallel_embedding:shape=(320, 256):dim=0:start=0:size=320" in out
    assert "tp_runtime_shard_in_proj_qkv=packed_qkv_column_parallel:shape=(256, 256):dim=0:segments=0+64,64+64,128+128" in out
    assert "tp_runtime_shard_shared_expert.gate_proj=replicated:shape=(128, 256)" in out


def test_cli_tp_reference_forward_single_rank(tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _runtime_config()
    text = config["text_config"]
    text["hidden_size"] = 256
    text["vocab_size"] = 320
    text["num_hidden_layers"] = 1
    text["layer_types"] = ["full_attention"]
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "model.language_model.embed_tokens.weight": ("BF16", (320, 256)),
        "model.language_model.norm.weight": ("BF16", (256,)),
        "lm_head.weight": ("BF16", (320, 256)),
    }
    add_full_attention_layer(tensors, 0)
    add_moe(tensors, 0)
    write_safetensors(tmp_path / "model.safetensors", tensors)

    class FakeTokenizer:
        def __call__(self, prompt: str, *, return_tensors: str, add_special_tokens: bool):
            assert return_tensors == "pt"
            assert add_special_tokens is True
            return {"input_ids": torch.tensor([[1]])}

    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_, **__: FakeTokenizer())

    rc = main(
        [
            "--model",
            str(tmp_path),
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
            "--tp-reference-forward",
            "--tp-world-size",
            "1",
            "--tp-backend",
            "gloo",
            "--tp-device",
            "cpu",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "TP reference forward smoke" in out
    assert "tp_reference_logits_shape: (1, 1, 320)" in out
    assert "tp_reference_logits_finite: True" in out
    assert "tp_reference_next_token:" in out


def test_cli_tp_generate_single_rank(tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _runtime_config()
    text = config["text_config"]
    text["hidden_size"] = 256
    text["vocab_size"] = 320
    text["num_hidden_layers"] = 1
    text["layer_types"] = ["full_attention"]
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "model.language_model.embed_tokens.weight": ("BF16", (320, 256)),
        "model.language_model.norm.weight": ("BF16", (256,)),
        "lm_head.weight": ("BF16", (320, 256)),
    }
    add_full_attention_layer(tensors, 0)
    add_moe(tensors, 0)
    write_safetensors(tmp_path / "model.safetensors", tensors)

    class FakeTokenizer:
        def __call__(self, prompt: str, *, return_tensors: str, add_special_tokens: bool):
            assert return_tensors == "pt"
            assert add_special_tokens is True
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens is False
            return "decoded:" + ",".join(str(token_id) for token_id in token_ids)

    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_, **__: FakeTokenizer())

    rc = main(
        [
            "--model",
            str(tmp_path),
            "--prompt",
            "hello",
            "--max-new-tokens",
            "2",
            "--tp-generate",
            "--tp-world-size",
            "1",
            "--tp-backend",
            "gloo",
            "--tp-device",
            "cpu",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "TP resident greedy generation" in out
    assert "tp_generate_world_size: 1" in out
    assert "tp_generate_rank: 0" in out
    assert "tp_generate_device: cpu" in out
    assert "tp_generate_loaded_tensors:" in out
    assert "tp_generate_loaded_bytes:" in out
    assert "tp_generate_load_seconds:" in out
    assert "tp_generate_prefill_seconds:" in out
    assert "tp_generate_decode_seconds:" in out
    assert "tp_generate_total_seconds:" in out
    assert "tp_generate_decode_tokens_per_second:" in out
    assert "tp_generate_total_tokens_per_second:" in out
    assert "tp_generate_dispatch_calls:" in out
    assert "tp_generate_dispatch_moe_packed_single_scatter_calls:" in out
    assert "tp_generate_dispatch_moe_native_assignment_hits:" in out
    assert "tp_generate_paged_attention_calls:" in out
    assert "tp_generate_paged_attention_dense_fallbacks:" in out
    assert "tp_generate_kv_full_attention_layers: 1" in out
    assert "tp_generate_kv_page_metadata_calls_total:" in out
    assert "tp_generate_kv_page_table_entries_total:" in out
    assert "tp_generate_profile_enabled: False" in out
    assert "tp_generate_all_finite: True" in out
    assert "tp_generate_generated_token_ids:" in out
    assert "tp_generate_text: decoded:" in out
    assert "inference: not implemented yet" not in out


def test_cli_tp_generate_with_profile_prints_profile_lines(tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _runtime_config()
    text = config["text_config"]
    text["hidden_size"] = 256
    text["vocab_size"] = 320
    text["num_hidden_layers"] = 1
    text["layer_types"] = ["full_attention"]
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "model.language_model.embed_tokens.weight": ("BF16", (320, 256)),
        "model.language_model.norm.weight": ("BF16", (256,)),
        "lm_head.weight": ("BF16", (320, 256)),
    }
    add_full_attention_layer(tensors, 0)
    add_moe(tensors, 0)
    write_safetensors(tmp_path / "model.safetensors", tensors)

    class FakeTokenizer:
        def __call__(self, prompt: str, *, return_tensors: str, add_special_tokens: bool):
            assert return_tensors == "pt"
            assert add_special_tokens is True
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens is False
            return "decoded:" + ",".join(str(token_id) for token_id in token_ids)

    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_, **__: FakeTokenizer())

    rc = main(
        [
            "--model",
            str(tmp_path),
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
            "--tp-generate",
            "--tp-profile",
            "cpu",
            "--tp-world-size",
            "1",
            "--tp-backend",
            "gloo",
            "--tp-device",
            "cpu",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "tp_generate_profile_enabled: True" in out
    assert "tp_generate_profile_scope_0_name:" in out



def test_cli_tp_benchmark_single_rank(tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _runtime_config()
    text = config["text_config"]
    text["hidden_size"] = 256
    text["vocab_size"] = 320
    text["num_hidden_layers"] = 1
    text["layer_types"] = ["full_attention"]
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "model.language_model.embed_tokens.weight": ("BF16", (320, 256)),
        "model.language_model.norm.weight": ("BF16", (256,)),
        "lm_head.weight": ("BF16", (320, 256)),
    }
    add_full_attention_layer(tensors, 0)
    add_moe(tensors, 0)
    write_safetensors(tmp_path / "model.safetensors", tensors)

    class FakeTokenizer:
        def __call__(self, prompt: str, *, return_tensors: str, add_special_tokens: bool):
            assert return_tensors == "pt"
            assert add_special_tokens is True
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens is False
            return "decoded:" + ",".join(str(token_id) for token_id in token_ids)

    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_, **__: FakeTokenizer())

    rc = main(
        [
            "--model",
            str(tmp_path),
            "--prompt",
            "hello",
            "--max-new-tokens",
            "2",
            "--tp-benchmark",
            "--tp-world-size",
            "1",
            "--tp-backend",
            "gloo",
            "--tp-device",
            "cpu",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "TP resident generation benchmark" in out
    assert "tp_benchmark_iterations: 1" in out
    assert "tp_benchmark_warmup_iterations: 0" in out
    assert "tp_benchmark_prefill_seconds_avg:" in out
    assert "tp_benchmark_decode_seconds_avg:" in out
    assert "tp_generate_load_seconds:" in out
    assert "tp_generate_prefill_seconds:" in out
    assert "tp_generate_decode_seconds:" in out
    assert "tp_generate_total_seconds:" in out
    assert "tp_generate_decode_tokens_per_second:" in out
    assert "tp_generate_total_tokens_per_second:" in out
    assert "tp_generate_dispatch_calls:" in out
    assert "tp_generate_all_finite: True" in out
    assert "tp_generate_generated_token_ids:" in out
    assert "tp_generate_text: decoded:" in out
    assert "inference: not implemented yet" not in out


def test_cli_tp_worker_runs_protocol_loop_without_human_stdout(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_runtime_config()), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")
    calls = []

    def fake_protocol_loop(state, runtime):
        calls.append((state, runtime))
        print('{"ok":true}')

    monkeypatch.setattr(cli, "run_worker_protocol_loop", fake_protocol_loop)

    rc = main(
        [
            "--model",
            str(tmp_path),
            "--prompt",
            "unused",
            "--max-new-tokens",
            "1",
            "--tp-worker",
            "--tp-world-size",
            "1",
            "--tp-backend",
            "gloo",
            "--tp-device",
            "cpu",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert out == '{"ok":true}\n'
    assert "Loaded model config" not in out
    assert "TP resident generation benchmark" not in out
    assert "inference: not implemented yet" not in out
    assert len(calls) == 1
    state, runtime = calls[0]
    assert isinstance(state, cli.WorkerState)
    assert str(runtime.device) == "cpu"



def test_cli_tp_service_runs_http_service_without_human_stdout(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_runtime_config()), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")
    calls = []

    def fake_service(state, runtime, config):
        calls.append((state, runtime, config))
        print('{"service":true}')

    monkeypatch.setattr(cli, "serve_worker_http", fake_service)

    rc = main(
        [
            "--model",
            str(tmp_path),
            "--prompt",
            "unused",
            "--max-new-tokens",
            "1",
            "--tp-service",
            "--tp-host",
            "127.0.0.2",
            "--tp-port",
            "8123",
            "--tp-world-size",
            "1",
            "--tp-backend",
            "gloo",
            "--tp-device",
            "cpu",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert out == '{"service":true}\n'
    assert "Loaded model config" not in out
    assert "inference: not implemented yet" not in out
    assert len(calls) == 1
    state, runtime, config = calls[0]
    assert isinstance(state, cli.WorkerState)
    assert str(runtime.device) == "cpu"
    assert config.host == "127.0.0.2"
    assert config.port == 8123
    assert config.model_dir == str(tmp_path.resolve())
    assert config.max_active_requests == cli.DEFAULT_MAX_ACTIVE_REQUESTS
    assert config.max_pending_requests == cli.DEFAULT_MAX_PENDING_REQUESTS
    assert config.batch_step_mode == cli.STEP_MODE_COOPERATIVE
    assert state.max_active_requests == cli.DEFAULT_MAX_ACTIVE_REQUESTS
    assert state.max_pending_requests == cli.DEFAULT_MAX_PENDING_REQUESTS
    assert state.batch_step_mode == cli.STEP_MODE_COOPERATIVE

    config = _runtime_config()
    text = config["text_config"]
    text["hidden_size"] = 256
    text["vocab_size"] = 320
    text["num_hidden_layers"] = 1
    text["layer_types"] = ["full_attention"]
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "model.language_model.embed_tokens.weight": ("BF16", (320, 256)),
        "model.language_model.norm.weight": ("BF16", (256,)),
        "lm_head.weight": ("BF16", (320, 256)),
    }
    add_full_attention_layer(tensors, 0)
    add_moe(tensors, 0)
    write_safetensors(tmp_path / "model.safetensors", tensors)

    class FakeTokenizer:
        def __call__(self, prompt: str, *, return_tensors: str, add_special_tokens: bool):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_, **__: FakeTokenizer())

    rc = main(
        [
            "--model",
            str(tmp_path),
            "--prompt",
            "hi",
            "--max-new-tokens",
            "1",
            "--reference-decode-smoke",
            "--reference-max-tokens",
            "3",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "Reference decode smoke" in out
    assert "reference_decode_steps: 3" in out
    assert "reference_decode_all_finite: True" in out
    assert "reference_decode_next_tokens:" in out


def test_cli_tp_service_passes_custom_serving_controls(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_runtime_config()), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")
    calls = []

    def fake_service(state, runtime, config):
        calls.append((state, runtime, config))
        print('{"service":true}')

    monkeypatch.setattr(cli, "serve_worker_http", fake_service)

    rc = main(
        [
            "--model",
            str(tmp_path),
            "--prompt",
            "unused",
            "--max-new-tokens",
            "1",
            "--tp-service",
            "--tp-service-max-active",
            "2",
            "--tp-service-max-pending",
            "8",
            "--tp-service-batch-step-mode",
            cli.STEP_MODE_LEGACY,
            "--tp-world-size",
            "1",
            "--tp-backend",
            "gloo",
            "--tp-device",
            "cpu",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert out == '{"service":true}\n'
    assert len(calls) == 1
    state, _runtime, config = calls[0]
    assert config.max_active_requests == 2
    assert config.max_pending_requests == 8
    assert config.batch_step_mode == cli.STEP_MODE_LEGACY
    assert state.max_active_requests == 2
    assert state.max_pending_requests == 8
    assert state.batch_step_mode == cli.STEP_MODE_LEGACY


def test_cli_rejects_invalid_service_limits(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_runtime_config()), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")

    with pytest.raises(SystemExit):
        main(["--model", str(tmp_path), "--prompt", "unused", "--max-new-tokens", "1", "--tp-service-max-active", "0"])
    with pytest.raises(SystemExit):
        main(["--model", str(tmp_path), "--prompt", "unused", "--max-new-tokens", "1", "--tp-service-max-pending", "0"])



def test_cli_rejects_invalid_benchmark_controls(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_runtime_config()), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")

    with pytest.raises(SystemExit):
        main(["--model", str(tmp_path), "--prompt", "unused", "--max-new-tokens", "1", "--tp-benchmark-iterations", "0"])
    with pytest.raises(SystemExit):
        main(["--model", str(tmp_path), "--prompt", "unused", "--max-new-tokens", "1", "--tp-benchmark-warmup", "-1"])
    with pytest.raises(SystemExit):
        main(["--model", str(tmp_path), "--prompt", "unused", "--max-new-tokens", "1", "--tp-benchmark-prompt-tokens", "0"])


def test_reference_layer_defaults_to_first_full_attention_layer() -> None:
    layer = _reference_layer(_mapping(("linear_attention", "full_attention")), None)

    assert layer.index == 1


def test_reference_layer_rejects_linear_attention_layer() -> None:
    with pytest.raises(CliError, match="expected full_attention"):
        _reference_layer(_mapping(("linear_attention", "full_attention")), 0)


def _mapping(layer_types: tuple[str, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        layers=tuple(SimpleNamespace(index=index, layer_type=layer_type) for index, layer_type in enumerate(layer_types))
    )


def _runtime_config() -> dict[str, object]:
    return {
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 256,
            "vocab_size": 320,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "attn_output_gate": True,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 128,
            "shared_expert_intermediate_size": 128,
            "max_position_embeddings": 1024,
            "rms_norm_eps": 1e-6,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {"rope_theta": 10000000},
        }
    }
