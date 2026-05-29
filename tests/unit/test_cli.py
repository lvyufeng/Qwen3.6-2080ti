from __future__ import annotations

import json
from pathlib import Path

from cli import main


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
