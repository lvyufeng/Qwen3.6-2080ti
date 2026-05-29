from __future__ import annotations

import torch

from checkpoint import TensorInfo
from reference_ops import (
    ReferenceWeights,
    dequantize_fp8_weight,
    embedding,
    full_attention,
    linear,
    rms_norm,
    topk_route,
)
from runtime_config import parse_runtime_config
from weight_mapping import ExpertMapping, FullAttentionMapping, LinearTensor, MoEMapping


def test_embedding_matches_torch_indexing() -> None:
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    input_ids = torch.tensor([[2, 0], [1, 2]])

    out = embedding(input_ids, weight)

    assert out.tolist() == [[[5.0, 6.0], [1.0, 2.0]], [[3.0, 4.0], [5.0, 6.0]]]


def test_rms_norm_uses_qwen_one_plus_weight() -> None:
    x = torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16)
    weight = torch.tensor([0.0, 1.0], dtype=torch.bfloat16)

    out = rms_norm(x, weight, eps=0.0)

    expected = torch.tensor([[3.0 / 3.5355339, 2.0 * 4.0 / 3.5355339]], dtype=torch.bfloat16)
    torch.testing.assert_close(out, expected)


def test_dequantize_fp8_weight_expands_128x128_scales() -> None:
    weight = torch.ones((129, 130), dtype=torch.float32)
    scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    out = dequantize_fp8_weight(weight, scale)

    assert out[0, 0].item() == 1.0
    assert out[0, 129].item() == 2.0
    assert out[128, 0].item() == 3.0
    assert out[128, 129].item() == 4.0


def test_linear_applies_dequantized_weight() -> None:
    x = torch.tensor([[1.0, 2.0]])
    weight = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
    scale = torch.tensor([[2.0]])

    out = linear(x, weight, scale)

    torch.testing.assert_close(out, torch.tensor([[6.0, 4.0]]))


def test_topk_route_matches_qwen_softmax_then_renormalize() -> None:
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    gate = torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 3.0]])

    routing = topk_route(hidden, gate, top_k=2)

    probabilities = torch.softmax(hidden @ gate.t(), dim=-1)
    expected_scores, expected_indices = torch.topk(probabilities, 2, dim=-1)
    expected_scores = expected_scores / expected_scores.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(routing.logits, probabilities)
    torch.testing.assert_close(routing.scores, expected_scores)
    torch.testing.assert_close(routing.indices, expected_indices)


def test_full_attention_preserves_shape_and_causal_mask() -> None:
    config = parse_runtime_config(_attention_config())
    mapping = FullAttentionMapping(
        q_proj=_linear_shape("q", (8, 4)),
        k_proj=_linear_shape("k", (4, 4)),
        v_proj=_linear_shape("v", (4, 4)),
        o_proj=_linear_shape("o", (4, 4)),
        q_norm=_info("q_norm", (4,)),
        k_norm=_info("k_norm", (4,)),
    )
    tensors = {
        "q": torch.cat((torch.eye(4), torch.full((4, 4), 20.0)), dim=0),
        "k": torch.eye(4),
        "v": torch.eye(4),
        "o": torch.eye(4),
        "q_norm": torch.zeros(4),
        "k_norm": torch.zeros(4),
    }
    weights = ReferenceWeights(_FakeLoader(tensors))
    hidden = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]])
    changed_future = hidden.clone()
    changed_future[:, 1] = torch.tensor([9.0, 9.0, 9.0, 9.0])

    out = full_attention(hidden, mapping, config, weights)
    changed_out = full_attention(changed_future, mapping, config, weights)

    assert out.shape == hidden.shape
    torch.testing.assert_close(out[:, 0], changed_out[:, 0])
    assert not torch.allclose(out[:, 1], changed_out[:, 1])


def test_reference_weights_moe_matches_manual_top1_expert_path() -> None:
    loader = _FakeLoader(
        {
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
    mapping = MoEMapping(
        gate=_info("gate", (2, 2)),
        experts=(
            ExpertMapping(_linear("e0_gate"), _linear("e0_up"), _linear("e0_down")),
            ExpertMapping(_linear("e1_gate"), _linear("e1_up"), _linear("e1_down")),
        ),
        shared_expert=ExpertMapping(_linear("shared_gate_proj"), _linear("shared_up_proj"), _linear("shared_down_proj")),
        shared_expert_gate=_info("shared_gate", (1, 2)),
    )
    config = parse_runtime_config(_config())
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    out = ReferenceWeights(loader).moe(hidden, mapping, config)

    expected = torch.stack(
        [
            torch.nn.functional.silu(torch.tensor([1.0, 0.0])) * torch.tensor([1.0, 0.0]),
            torch.nn.functional.silu(torch.tensor([0.0, 2.0])) * torch.tensor([0.0, 1.0]),
        ]
    ).unsqueeze(0)
    torch.testing.assert_close(out, expected)


class _FakeLoader:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def tensor(self, name: str, *, device: str | None = None) -> torch.Tensor:
        tensor = self.tensors[name]
        return tensor if device is None else tensor.to(device)


def _linear(name: str) -> LinearTensor:
    return LinearTensor(weight=_info(name, (2, 2)), scale=None)


def _linear_shape(name: str, shape: tuple[int, int]) -> LinearTensor:
    return LinearTensor(weight=_info(name, shape), scale=None)


def _info(name: str, shape: tuple[int, ...]) -> TensorInfo:
    return TensorInfo(name=name, dtype="BF16", shape=shape, shard="model.safetensors", begin=0, end=0, data_start=0)


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
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 2,
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


def _attention_config() -> dict[str, object]:
    raw = _config()
    text = raw["text_config"]
    text["hidden_size"] = 4
    text["vocab_size"] = 8
    text["head_dim"] = 4
    text["partial_rotary_factor"] = 0.5
    text["moe_intermediate_size"] = 4
    text["shared_expert_intermediate_size"] = 4
    return raw
