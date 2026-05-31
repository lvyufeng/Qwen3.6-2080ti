from __future__ import annotations

import pytest

from runtime_config import ConfigError, parse_runtime_config


def test_parse_runtime_config_derives_runtime_dimensions() -> None:
    config = parse_runtime_config(_config())

    assert config.model_type == "qwen3_5_moe_text"
    assert config.hidden_size == 256
    assert config.vocab_size == 320
    assert config.num_hidden_layers == 2
    assert config.layer_types == ("linear_attention", "full_attention")
    assert config.linear_attention_layers == 1
    assert config.full_attention_layers == 1
    assert config.linear_attention.qkv_dim == 256
    assert config.linear_attention.value_state_dim == 128
    assert config.full_attention.attn_dim == 128
    assert config.full_attention.kv_dim == 64
    assert config.full_attention.q_dim == 256
    assert config.moe.num_experts == 2
    assert config.moe.experts_per_token == 1
    assert config.moe.packed_expert_dispatch is True
    assert config.fp8_scale_shape((257, 255)) == (3, 2)


def test_parse_runtime_config_can_disable_packed_expert_dispatch() -> None:
    raw = _config()
    raw["text_config"]["packed_expert_dispatch"] = False

    config = parse_runtime_config(raw)

    assert config.moe.packed_expert_dispatch is False


def test_parse_runtime_config_rejects_bad_packed_expert_dispatch_type() -> None:
    raw = _config()
    raw["text_config"]["packed_expert_dispatch"] = "yes"

    with pytest.raises(ConfigError, match="packed_expert_dispatch"):
        parse_runtime_config(raw)


def test_parse_runtime_config_rejects_bad_layer_types() -> None:
    raw = _config()
    raw["text_config"]["layer_types"] = ["linear_attention"]

    with pytest.raises(ConfigError, match="layer_types"):
        parse_runtime_config(raw)


def test_parse_runtime_config_requires_rope_parameters() -> None:
    raw = _config()
    del raw["text_config"]["rope_parameters"]

    with pytest.raises(ConfigError, match="rope_parameters"):
        parse_runtime_config(raw)


def _config() -> dict[str, object]:
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
