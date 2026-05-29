from __future__ import annotations

from pathlib import Path

import pytest
import torch

from checkpoint import TensorInfo
from reference_ops import (
    ReferenceWeights,
    decoder_layer,
    dequantize_fp8_weight,
    embedding,
    full_attention,
    language_model,
    linear,
    linear_attention,
    rms_norm,
    topk_route,
)
from runtime_config import parse_runtime_config
from tensor_parallel import TensorParallel
from weight_mapping import (
    ExpertMapping,
    FullAttentionMapping,
    LanguageModelMapping,
    LayerMapping,
    LinearAttentionMapping,
    LinearTensor,
    MoEMapping,
)


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
    q_proj = torch.cat((torch.eye(4), torch.zeros((4, 4))), dim=0)
    loader = _FakeLoader(
        {
            "q": q_proj,
            "k": torch.eye(4),
            "v": torch.eye(4),
            "o": torch.eye(4),
            "q_norm": torch.zeros(4),
            "k_norm": torch.zeros(4),
        }
    )
    mapping = FullAttentionMapping(
        q_proj=_linear_shape("q", (8, 4)),
        k_proj=_linear_shape("k", (4, 4)),
        v_proj=_linear_shape("v", (4, 4)),
        o_proj=_linear_shape("o", (4, 4)),
        q_norm=_info("q_norm", (4,)),
        k_norm=_info("k_norm", (4,)),
    )
    config = parse_runtime_config(_attention_config())
    hidden = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])

    out = full_attention(hidden, mapping, config, ReferenceWeights(loader))
    changed_future = hidden.clone()
    changed_future[:, 2] = torch.tensor([10.0, 20.0, 30.0, 40.0])
    changed_out = full_attention(changed_future, mapping, config, ReferenceWeights(loader))

    assert out.shape == hidden.shape
    torch.testing.assert_close(out[:, :2], changed_out[:, :2])
    assert not torch.allclose(out[:, 2], changed_out[:, 2])


def test_linear_attention_preserves_shape_and_causal_conv() -> None:
    conv = torch.zeros((3, 1, 4))
    conv[:, :, -1] = 1.0
    loader = _FakeLoader(
        {
            "qkv": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            "z": torch.tensor([[1.0, 0.0]]),
            "out": torch.tensor([[1.0], [2.0]]),
            "a": torch.zeros((1, 2)),
            "b": torch.zeros((1, 2)),
            "conv": conv,
            "a_log": torch.zeros(1),
            "dt_bias": torch.zeros(1),
            "norm": torch.ones(1),
        }
    )
    mapping = LinearAttentionMapping(
        in_proj_qkv=_linear_shape("qkv", (3, 2)),
        in_proj_z=_linear_shape("z", (1, 2)),
        out_proj=_linear_shape("out", (2, 1)),
        in_proj_a=_linear_shape("a", (1, 2)),
        in_proj_b=_linear_shape("b", (1, 2)),
        conv1d_weight=_info("conv", (3, 1, 4)),
        a_log=_info("a_log", (1,)),
        dt_bias=_info("dt_bias", (1,)),
        norm=_info("norm", (1,)),
    )
    config = parse_runtime_config(_config())
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

    out = linear_attention(hidden, mapping, config, ReferenceWeights(loader))
    changed_future = hidden.clone()
    changed_future[:, 2] = torch.tensor([50.0, 60.0])
    changed_out = linear_attention(changed_future, mapping, config, ReferenceWeights(loader))

    assert out.shape == hidden.shape
    torch.testing.assert_close(out[:, :2], changed_out[:, :2])
    assert not torch.allclose(out[:, 2], changed_out[:, 2])


def test_language_model_runs_layers_final_norm_and_lm_head() -> None:
    loader = _FakeLoader(
        {
            "embed": torch.eye(4, 2),
            "final_norm": torch.zeros(2),
            "lm_head": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]),
        }
    )
    mapping = LanguageModelMapping(
        model_dir=Path("."),
        embed_tokens=_info("embed", (4, 2)),
        final_norm=_info("final_norm", (2,)),
        lm_head=_info("lm_head", (4, 2)),
        layers=(),
        mapped_tensor_names=frozenset(),
        ignored_tensor_names=frozenset(),
        unmapped_language_tensor_names=(),
    )
    config = parse_runtime_config(_config())
    input_ids = torch.tensor([[0, 1]])

    out = language_model(input_ids, mapping, config, ReferenceWeights(loader))

    expected_hidden = rms_norm(embedding(input_ids, loader.tensor("embed")), loader.tensor("final_norm"), config.rms_norm_eps)
    torch.testing.assert_close(out, expected_hidden.float() @ loader.tensor("lm_head").t())


def test_language_model_logits_match_transformers_tiny_full_attention() -> None:
    transformers = pytest.importorskip("transformers")
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    torch.manual_seed(0)
    hidden_size = 4
    vocab_size = 8
    intermediate_size = 4
    num_experts = 2
    hf_config = Qwen3_5MoeTextConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=hidden_size,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        moe_intermediate_size=intermediate_size,
        shared_expert_intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=1,
        max_position_embeddings=8,
        tie_word_embeddings=False,
        use_cache=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 1],
        },
    )
    model = Qwen3_5MoeForCausalLM(hf_config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)
        model.model.layers[0].mlp.gate.weight.zero_()
        model.model.layers[0].mlp.gate.weight[0, 0] = 5.0
        model.model.layers[0].mlp.gate.weight[1, 1] = 5.0

    tensors = _tensors_from_transformers_tiny_full_attention(model)
    mapping = _tiny_full_attention_mapping(hidden_size, vocab_size, intermediate_size, num_experts)
    config = parse_runtime_config(_tiny_full_attention_config(hidden_size, vocab_size, intermediate_size, num_experts))
    input_ids = torch.tensor([[0, 1, 2]])

    with torch.no_grad():
        hf_logits = model(input_ids, use_cache=False).logits
        ref_logits = language_model(input_ids, mapping, config, ReferenceWeights(_FakeLoader(tensors)))

    torch.testing.assert_close(ref_logits, hf_logits, atol=0.0, rtol=0.0)


def test_decoder_layer_matches_transformers_tiny_linear_attention() -> None:
    transformers = pytest.importorskip("transformers")
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextRotaryEmbedding

    torch.manual_seed(1)
    hidden_size = 4
    vocab_size = 8
    intermediate_size = 4
    num_experts = 2
    hf_config = Qwen3_5MoeTextConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=hidden_size,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=4,
        moe_intermediate_size=intermediate_size,
        shared_expert_intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=1,
        max_position_embeddings=8,
        tie_word_embeddings=False,
        use_cache=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 1],
        },
    )
    hf_layer = Qwen3_5MoeDecoderLayer(hf_config, 0).eval()
    with torch.no_grad():
        for parameter in hf_layer.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)
        hf_layer.mlp.gate.weight.zero_()
        hf_layer.mlp.gate.weight[0, 0] = 5.0
        hf_layer.mlp.gate.weight[1, 1] = 5.0

    hidden = torch.randn(1, 3, hidden_size)
    position_ids = torch.arange(hidden.shape[1]).unsqueeze(0)
    rotary = Qwen3_5MoeTextRotaryEmbedding(hf_config)
    position_embeddings = rotary(hidden, position_ids)
    tensors = _tensors_from_transformers_tiny_linear_attention(hf_layer)
    mapping = _tiny_linear_attention_layer_mapping(hidden_size, intermediate_size, num_experts)
    config = parse_runtime_config(_tiny_linear_attention_config(hidden_size, vocab_size, intermediate_size, num_experts))

    with torch.no_grad():
        hf_out = hf_layer(
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=None,
            position_ids=position_ids,
        )
        ref_out = decoder_layer(hidden, mapping, config, ReferenceWeights(_FakeLoader(tensors)))

    torch.testing.assert_close(ref_out, hf_out, atol=0.0, rtol=0.0)


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
            ExpertMapping(0, _linear("e0_gate"), _linear("e0_up"), _linear("e0_down")),
            ExpertMapping(1, _linear("e1_gate"), _linear("e1_up"), _linear("e1_down")),
        ),
        shared_expert=ExpertMapping(-1, _linear("shared_gate_proj"), _linear("shared_up_proj"), _linear("shared_down_proj")),
        shared_expert_gate=_info("shared_gate", (1, 2)),
        expert_start=0,
        expert_end=2,
        num_experts=2,
        tp=TensorParallel(world_size=1, rank=0),
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

def test_reference_weights_moe_tp_outputs_sum_to_dense_moe() -> None:
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
    dense = _moe_mapping((0, 1), TensorParallel(world_size=1, rank=0))
    rank0 = _moe_mapping((0,), TensorParallel(world_size=2, rank=0))
    rank1 = _moe_mapping((1,), TensorParallel(world_size=2, rank=1))
    config = parse_runtime_config(_config())
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    dense_out = ReferenceWeights(loader).moe(hidden, dense, config)
    tp_out = ReferenceWeights(loader).moe(hidden, rank0, config) + ReferenceWeights(loader).moe(hidden, rank1, config)

    torch.testing.assert_close(tp_out, dense_out)


def _moe_mapping(local_experts: tuple[int, ...], tp: TensorParallel) -> MoEMapping:
    names = {0: "e0", 1: "e1"}
    experts = tuple(
        ExpertMapping(i, _linear(f"{names[i]}_gate"), _linear(f"{names[i]}_up"), _linear(f"{names[i]}_down"))
        for i in local_experts
    )
    return MoEMapping(
        gate=_info("gate", (2, 2)),
        experts=experts,
        shared_expert=ExpertMapping(-1, _linear("shared_gate_proj"), _linear("shared_up_proj"), _linear("shared_down_proj")),
        shared_expert_gate=_info("shared_gate", (1, 2)),
        expert_start=local_experts[0],
        expert_end=local_experts[-1] + 1,
        num_experts=2,
        tp=tp,
    )


def _tensors_from_transformers_tiny_full_attention(model) -> dict[str, torch.Tensor]:
    layer = model.model.layers[0]
    tensors = {
        "embed": model.model.embed_tokens.weight.detach().clone(),
        "final_norm": model.model.norm.weight.detach().clone(),
        "lm_head": model.lm_head.weight.detach().clone(),
        "q": layer.self_attn.q_proj.weight.detach().clone(),
        "k": layer.self_attn.k_proj.weight.detach().clone(),
        "v": layer.self_attn.v_proj.weight.detach().clone(),
        "o": layer.self_attn.o_proj.weight.detach().clone(),
        "q_norm": layer.self_attn.q_norm.weight.detach().clone(),
        "k_norm": layer.self_attn.k_norm.weight.detach().clone(),
        "input_norm": layer.input_layernorm.weight.detach().clone(),
        "post_norm": layer.post_attention_layernorm.weight.detach().clone(),
        "gate": layer.mlp.gate.weight.detach().clone(),
        "shared_gate": layer.mlp.shared_expert_gate.weight.detach().clone(),
        "shared_gate_proj": layer.mlp.shared_expert.gate_proj.weight.detach().clone(),
        "shared_up_proj": layer.mlp.shared_expert.up_proj.weight.detach().clone(),
        "shared_down_proj": layer.mlp.shared_expert.down_proj.weight.detach().clone(),
    }
    for expert in range(layer.mlp.experts.gate_up_proj.shape[0]):
        packed = layer.mlp.experts.gate_up_proj.detach()[expert]
        intermediate = packed.shape[0] // 2
        tensors[f"e{expert}_gate"] = packed[:intermediate].clone()
        tensors[f"e{expert}_up"] = packed[intermediate:].clone()
        tensors[f"e{expert}_down"] = layer.mlp.experts.down_proj.detach()[expert].clone()
    return tensors


def _tiny_full_attention_mapping(
    hidden_size: int,
    vocab_size: int,
    intermediate_size: int,
    num_experts: int,
) -> LanguageModelMapping:
    return LanguageModelMapping(
        model_dir=Path("."),
        embed_tokens=_info("embed", (vocab_size, hidden_size)),
        final_norm=_info("final_norm", (hidden_size,)),
        lm_head=_info("lm_head", (vocab_size, hidden_size)),
        layers=(
            LayerMapping(
                index=0,
                layer_type="full_attention",
                input_layernorm=_info("input_norm", (hidden_size,)),
                attention=FullAttentionMapping(
                    q_proj=_linear_shape("q", (hidden_size * 2, hidden_size)),
                    k_proj=_linear_shape("k", (hidden_size, hidden_size)),
                    v_proj=_linear_shape("v", (hidden_size, hidden_size)),
                    o_proj=_linear_shape("o", (hidden_size, hidden_size)),
                    q_norm=_info("q_norm", (hidden_size,)),
                    k_norm=_info("k_norm", (hidden_size,)),
                ),
                post_attention_layernorm=_info("post_norm", (hidden_size,)),
                mlp=MoEMapping(
                    gate=_info("gate", (num_experts, hidden_size)),
                    experts=tuple(
                        ExpertMapping(
                            expert,
                            _linear_shape(f"e{expert}_gate", (intermediate_size, hidden_size)),
                            _linear_shape(f"e{expert}_up", (intermediate_size, hidden_size)),
                            _linear_shape(f"e{expert}_down", (hidden_size, intermediate_size)),
                        )
                        for expert in range(num_experts)
                    ),
                    shared_expert=ExpertMapping(
                        -1,
                        _linear_shape("shared_gate_proj", (intermediate_size, hidden_size)),
                        _linear_shape("shared_up_proj", (intermediate_size, hidden_size)),
                        _linear_shape("shared_down_proj", (hidden_size, intermediate_size)),
                    ),
                    shared_expert_gate=_info("shared_gate", (1, hidden_size)),
                    expert_start=0,
                    expert_end=num_experts,
                    num_experts=num_experts,
                    tp=TensorParallel(world_size=1, rank=0),
                ),
            ),
        ),
        mapped_tensor_names=frozenset(),
        ignored_tensor_names=frozenset(),
        unmapped_language_tensor_names=(),
    )


def _tensors_from_transformers_tiny_linear_attention(layer) -> dict[str, torch.Tensor]:
    tensors = {
        "input_norm": layer.input_layernorm.weight.detach().clone(),
        "post_norm": layer.post_attention_layernorm.weight.detach().clone(),
        "qkv": layer.linear_attn.in_proj_qkv.weight.detach().clone(),
        "z": layer.linear_attn.in_proj_z.weight.detach().clone(),
        "out": layer.linear_attn.out_proj.weight.detach().clone(),
        "a": layer.linear_attn.in_proj_a.weight.detach().clone(),
        "b": layer.linear_attn.in_proj_b.weight.detach().clone(),
        "conv": layer.linear_attn.conv1d.weight.detach().clone(),
        "a_log": layer.linear_attn.A_log.detach().clone(),
        "dt_bias": layer.linear_attn.dt_bias.detach().clone(),
        "norm": layer.linear_attn.norm.weight.detach().clone(),
        "gate": layer.mlp.gate.weight.detach().clone(),
        "shared_gate": layer.mlp.shared_expert_gate.weight.detach().clone(),
        "shared_gate_proj": layer.mlp.shared_expert.gate_proj.weight.detach().clone(),
        "shared_up_proj": layer.mlp.shared_expert.up_proj.weight.detach().clone(),
        "shared_down_proj": layer.mlp.shared_expert.down_proj.weight.detach().clone(),
    }
    for expert in range(layer.mlp.experts.gate_up_proj.shape[0]):
        packed = layer.mlp.experts.gate_up_proj.detach()[expert]
        intermediate = packed.shape[0] // 2
        tensors[f"e{expert}_gate"] = packed[:intermediate].clone()
        tensors[f"e{expert}_up"] = packed[intermediate:].clone()
        tensors[f"e{expert}_down"] = layer.mlp.experts.down_proj.detach()[expert].clone()
    return tensors


def _tiny_linear_attention_layer_mapping(hidden_size: int, intermediate_size: int, num_experts: int) -> LayerMapping:
    return LayerMapping(
        index=0,
        layer_type="linear_attention",
        input_layernorm=_info("input_norm", (hidden_size,)),
        attention=LinearAttentionMapping(
            in_proj_qkv=_linear_shape("qkv", (6, hidden_size)),
            in_proj_z=_linear_shape("z", (2, hidden_size)),
            out_proj=_linear_shape("out", (hidden_size, 2)),
            in_proj_a=_linear_shape("a", (1, hidden_size)),
            in_proj_b=_linear_shape("b", (1, hidden_size)),
            conv1d_weight=_info("conv", (6, 1, 4)),
            a_log=_info("a_log", (1,)),
            dt_bias=_info("dt_bias", (1,)),
            norm=_info("norm", (2,)),
        ),
        post_attention_layernorm=_info("post_norm", (hidden_size,)),
        mlp=_tiny_moe_mapping(hidden_size, intermediate_size, num_experts),
    )


def _tiny_moe_mapping(hidden_size: int, intermediate_size: int, num_experts: int) -> MoEMapping:
    return MoEMapping(
        gate=_info("gate", (num_experts, hidden_size)),
        experts=tuple(
            ExpertMapping(
                expert,
                _linear_shape(f"e{expert}_gate", (intermediate_size, hidden_size)),
                _linear_shape(f"e{expert}_up", (intermediate_size, hidden_size)),
                _linear_shape(f"e{expert}_down", (hidden_size, intermediate_size)),
            )
            for expert in range(num_experts)
        ),
        shared_expert=ExpertMapping(
            -1,
            _linear_shape("shared_gate_proj", (intermediate_size, hidden_size)),
            _linear_shape("shared_up_proj", (intermediate_size, hidden_size)),
            _linear_shape("shared_down_proj", (hidden_size, intermediate_size)),
        ),
        shared_expert_gate=_info("shared_gate", (1, hidden_size)),
        expert_start=0,
        expert_end=num_experts,
        num_experts=num_experts,
        tp=TensorParallel(world_size=1, rank=0),
    )


def _tiny_linear_attention_config(
    hidden_size: int,
    vocab_size: int,
    intermediate_size: int,
    num_experts: int,
) -> dict[str, object]:
    return {
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "num_hidden_layers": 1,
            "layer_types": ["linear_attention"],
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 2,
            "linear_value_head_dim": 2,
            "linear_conv_kernel_dim": 4,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": hidden_size,
            "attn_output_gate": True,
            "num_experts": num_experts,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": intermediate_size,
            "shared_expert_intermediate_size": intermediate_size,
            "max_position_embeddings": 8,
            "rms_norm_eps": 1e-6,
            "partial_rotary_factor": 0.5,
            "rope_parameters": {"rope_theta": 10000},
        }
    }


def _tiny_full_attention_config(
    hidden_size: int,
    vocab_size: int,
    intermediate_size: int,
    num_experts: int,
) -> dict[str, object]:
    return {
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "num_hidden_layers": 1,
            "layer_types": ["full_attention"],
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 2,
            "linear_value_head_dim": 2,
            "linear_conv_kernel_dim": 4,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": hidden_size,
            "attn_output_gate": True,
            "num_experts": num_experts,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": intermediate_size,
            "shared_expert_intermediate_size": intermediate_size,
            "max_position_embeddings": 8,
            "rms_norm_eps": 1e-6,
            "partial_rotary_factor": 0.5,
            "rope_parameters": {"rope_theta": 10000},
        }
    }


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
