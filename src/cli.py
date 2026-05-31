from __future__ import annotations

import argparse
from pathlib import Path

from checkpoint import CheckpointError, Manifest, build_manifest
from decode_state import DecodeState
from engine import EngineError, GenerateResult, TpModelRunner
from fp8_smoke import Fp8SmokeReport, inspect_fp8_checkpoint
from loader import LoaderError, TensorLoader
from reference_ops import ReferenceWeights, decode_step, decoder_layer, embedding, language_model
from runtime_config import ConfigError, RuntimeConfig, parse_runtime_config
from service import ServiceConfig, serve_worker_http
from tensor_parallel import TensorParallel
from tp_runtime import TpLaunchConfig, TpRuntime, TpRuntimeError, mapped_tensor_bytes, tp_decode_step, tp_language_model
from tp_weights import MappedWeights
from weight_mapping import LanguageModelMapping, MappingError, build_language_model_mapping
from worker import (
    DEFAULT_MAX_ACTIVE_REQUESTS,
    DEFAULT_MAX_PENDING_REQUESTS,
    STEP_MODE_COOPERATIVE,
    STEP_MODE_LEGACY,
    WorkerState,
    run_worker_protocol_loop,
)


class CliError(RuntimeError):
    pass



def _summarize_config(config: dict[str, object]) -> list[str]:
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        config = text_config
    keys = [
        "model_type",
        "architectures",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "num_experts",
        "num_experts_per_tok",
        "vocab_size",
        "torch_dtype",
    ]
    lines: list[str] = []
    for key in keys:
        if key in config:
            lines.append(f"{key}: {config[key]}")
    return lines


def _summarize_runtime_config(config: RuntimeConfig) -> list[str]:
    return [
        f"runtime_model_type: {config.model_type}",
        f"runtime_layers: {config.num_hidden_layers}",
        f"runtime_linear_attention_layers: {config.linear_attention_layers}",
        f"runtime_full_attention_layers: {config.full_attention_layers}",
        f"runtime_hidden_size: {config.hidden_size}",
        f"runtime_linear_qkv_dim: {config.linear_attention.qkv_dim}",
        f"runtime_linear_value_state_dim: {config.linear_attention.value_state_dim}",
        f"runtime_attention_q_dim: {config.full_attention.q_dim}",
        f"runtime_attention_kv_dim: {config.full_attention.kv_dim}",
        f"runtime_experts_per_layer: {config.moe.num_experts}",
        f"runtime_experts_per_token: {config.moe.experts_per_token}",
        f"runtime_packed_expert_dispatch: {config.moe.packed_expert_dispatch}",
        f"runtime_native_fused_expert_dispatch: {config.moe.native_fused_expert_dispatch}",
        f"runtime_fp8_block_size: {config.fp8_block_size}",
    ]


def _summarize_manifest(manifest: Manifest) -> list[str]:
    shard_count = len({tensor.shard for tensor in manifest.tensors.values()})
    return [
        f"safetensors_shards: {shard_count}",
        f"tensor_count: {len(manifest.tensors)}",
        f"fp8_tensor_count: {manifest.fp8_tensor_count}",
        f"scale_links: {len(manifest.scale_of)}",
        f"manifest_bytes: {manifest.total_bytes}",
        f"manifest_params_without_scales: {manifest.param_count}",
    ]


def _summarize_fp8(report: Fp8SmokeReport) -> list[str]:
    missing = ",".join(report.missing_scales[:8])
    if len(report.missing_scales) > 8:
        missing += f",...(+{len(report.missing_scales) - 8})"
    return [
        f"fp8_smoke_ok: {report.ok}",
        f"fp8_weight_tensors: {report.fp8_tensors}",
        f"fp8_scale_links: {report.scale_links}",
        f"fp8_missing_scales: {len(report.missing_scales)}",
        f"fp8_weight_bytes: {report.fp8_bytes}",
        f"fp8_scale_bytes: {report.scale_bytes}",
        f"fp8_missing_scale_examples: {missing}" if missing else "fp8_missing_scale_examples: none",
    ]


def _summarize_mapping(mapping: LanguageModelMapping) -> list[str]:
    return [
        f"layers: {len(mapping.layers)}",
        f"linear_attention_layers: {mapping.linear_attention_layers}",
        f"full_attention_layers: {mapping.full_attention_layers}",
        f"experts_per_layer: {mapping.experts_per_layer}",
        f"routed_experts_total: {mapping.routed_experts}",
        f"mapped_tensors: {len(mapping.mapped_tensor_names)}",
        f"ignored_tensors: {len(mapping.ignored_tensor_names)}",
        f"unmapped_language_tensors: {len(mapping.unmapped_language_tensor_names)}",
    ]

def _shard_tag(tensor) -> str:
    shard = tensor.shard
    parts = [shard.rule, f"shape={tuple(tensor.shape)}"]
    if shard.dim is not None:
        parts.append(f"dim={shard.dim}")
    if shard.start is not None:
        parts.append(f"start={shard.start}")
    if shard.size is not None:
        parts.append(f"size={shard.size}")
    if shard.segments:
        segments = ",".join(f"{segment.start}+{segment.size}" for segment in shard.segments)
        parts.append(f"segments={segments}")
    return ":".join(parts)


def _summarize_tp_shards(mapping: LanguageModelMapping) -> list[str]:
    first_layer = mapping.layers[0]
    attention = first_layer.attention
    lines = [
        f"  embed_tokens={_shard_tag(mapping.embed_tokens)}",
        f"  lm_head={_shard_tag(mapping.lm_head)}",
    ]
    if first_layer.layer_type == "full_attention":
        lines.extend(
            [
                f"  q_proj={_shard_tag(attention.q_proj.weight)}",
                f"  k_proj={_shard_tag(attention.k_proj.weight)}",
                f"  v_proj={_shard_tag(attention.v_proj.weight)}",
                f"  o_proj={_shard_tag(attention.o_proj.weight)}",
            ]
        )
    else:
        lines.extend(
            [
                f"  in_proj_qkv={_shard_tag(attention.in_proj_qkv.weight)}",
                f"  conv1d={_shard_tag(attention.conv1d_weight)}",
                f"  out_proj={_shard_tag(attention.out_proj.weight)}",
            ]
        )
    shared = first_layer.mlp.shared_expert
    lines.append(f"  shared_expert.gate_proj={_shard_tag(shared.gate_proj.weight)}")
    return lines


def _cuda_memory_lines(prefix: str, device) -> list[str]:
    if getattr(device, "type", None) != "cuda":
        return [f"{prefix}_cuda_memory: unavailable"]
    import torch

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return [
        f"{prefix}_cuda_memory_free: {free_bytes}",
        f"{prefix}_cuda_memory_total: {total_bytes}",
        f"{prefix}_cuda_max_allocated: {torch.cuda.max_memory_allocated(device)}",
        f"{prefix}_cuda_max_reserved: {torch.cuda.max_memory_reserved(device)}",
    ]


def _tensor_report_lines(prefix: str, weights: MappedWeights, names: list[str]) -> list[str]:
    lines: list[str] = []
    for label, name in enumerate(names):
        try:
            tensor = weights.tensor(name)
        except KeyError:
            continue
        lines.append(
            f"{prefix}_tensor_{label}: name={name} shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"
        )
    return lines


def _representative_tensor_names(mapping: LanguageModelMapping) -> list[str]:
    first_layer = mapping.layers[0]
    names = [mapping.embed_tokens.name, mapping.lm_head.name]
    attention = first_layer.attention
    if first_layer.layer_type == "full_attention":
        names.extend([attention.q_proj.weight.name, attention.o_proj.weight.name])
    else:
        names.extend([attention.in_proj_qkv.weight.name, attention.out_proj.weight.name])
    if first_layer.mlp.experts:
        names.append(first_layer.mlp.experts[0].gate_proj.weight.name)
    names.append(first_layer.mlp.shared_expert.gate_proj.weight.name)
    return names


def _summarize_tp_mapping(manifest: Manifest, world_size: int) -> list[str]:
    lines = [f"tp_world_size: {world_size}"]
    total_local = 0
    for rank in range(world_size):
        mapping = build_language_model_mapping(
            manifest, strict=True, tensor_parallel=TensorParallel(world_size=world_size, rank=rank)
        )
        first_moe = mapping.layers[0].mlp
        total_local += mapping.routed_experts
        lines.append(
            f"tp_rank_{rank}: experts_per_layer={mapping.experts_per_layer} "
            f"expert_range=[{first_moe.expert_start},{first_moe.expert_end}) "
            f"local_routed_experts={mapping.routed_experts} mapped_tensors={len(mapping.mapped_tensor_names)} "
            f"mapped_bytes={mapped_tensor_bytes(mapping)}"
        )
        lines.extend(f"tp_rank_{rank}_shard_{line.strip()}" for line in _summarize_tp_shards(mapping))
    dense = build_language_model_mapping(manifest, strict=True)
    lines.append(f"tp_total_local_experts: {total_local}")
    lines.append(f"tp_dense_routed_experts: {dense.routed_experts}")
    lines.append(f"tp_partition_complete: {total_local == dense.routed_experts}")
    lines.append(f"tp_dense_mapped_bytes: {mapped_tensor_bytes(dense)}")
    return lines


def _summarize_tp_runtime_smoke(
    manifest: Manifest,
    runtime_config: RuntimeConfig,
    world_size: int | None,
    rank: int | None,
    local_rank: int | None,
    backend: str,
    init_method: str | None,
    device: str | None,
) -> list[str]:
    launch = _tp_launch_from_args(world_size, rank, local_rank, backend, init_method, device)
    tp = TensorParallel(world_size=launch.world_size, rank=launch.rank)
    mapping = build_language_model_mapping(manifest, strict=True, tensor_parallel=tp)
    with TpRuntime(launch) as runtime:
        runtime.barrier()
        first_moe = mapping.layers[0].mlp
        lines = [
            f"tp_runtime_backend: {backend}",
            f"tp_runtime_world_size: {launch.world_size}",
            f"tp_runtime_rank: {launch.rank}",
            f"tp_runtime_local_rank: {launch.local_rank}",
            f"tp_runtime_device: {runtime.device}",
            f"tp_runtime_layers: {runtime_config.num_hidden_layers}",
            f"tp_runtime_expert_range: [{first_moe.expert_start},{first_moe.expert_end})",
            f"tp_runtime_experts_per_layer: {mapping.experts_per_layer}",
            f"tp_runtime_local_routed_experts: {mapping.routed_experts}",
            f"tp_runtime_mapped_tensors: {len(mapping.mapped_tensor_names)}",
            f"tp_runtime_mapped_bytes: {mapped_tensor_bytes(mapping)}",
        ]
        lines.extend(f"tp_runtime_shard_{line.strip()}" for line in _summarize_tp_shards(mapping))
        runtime.barrier()
        return lines


def _tp_launch_from_args(
    world_size: int | None,
    rank: int | None,
    local_rank: int | None,
    backend: str,
    init_method: str | None,
    device: str | None,
) -> TpLaunchConfig:
    return TpLaunchConfig.from_env(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        backend=backend,
        init_method=init_method,
        device=device,
    )


def _summarize_tp_generate(
    manifest: Manifest,
    runtime_config: RuntimeConfig,
    prompt: str,
    max_new_tokens: int,
    world_size: int | None,
    rank: int | None,
    local_rank: int | None,
    backend: str,
    init_method: str | None,
    device: str | None,
) -> list[str]:
    launch = _tp_launch_from_args(world_size, rank, local_rank, backend, init_method, device)
    runner = TpModelRunner(manifest, runtime_config, launch)
    return _format_tp_generate_result(runner.generate(prompt, max_new_tokens))


def _format_tp_generate_result(result: GenerateResult) -> list[str]:
    dispatch = result.dispatch_stats
    lines = [
        f"tp_generate_backend: {result.backend}",
        f"tp_generate_world_size: {result.world_size}",
        f"tp_generate_rank: {result.rank}",
        f"tp_generate_local_rank: {result.local_rank}",
        f"tp_generate_device: {result.device}",
        f"tp_generate_prompt_tokens: {result.prompt_tokens}",
        f"tp_generate_max_new_tokens: {result.max_new_tokens}",
        f"tp_generate_layers: {result.layers}",
        f"tp_generate_mapped_tensors: {result.mapped_tensors}",
        f"tp_generate_mapped_bytes: {result.mapped_bytes}",
        f"tp_generate_loaded_tensors: {result.load_stats.tensor_count}",
        f"tp_generate_loaded_bytes: {result.load_stats.bytes}",
        f"tp_generate_load_seconds: {result.load_seconds:.6f}",
        f"tp_generate_prefill_seconds: {result.prefill_seconds:.6f}",
        f"tp_generate_decode_seconds: {result.decode_seconds:.6f}",
        f"tp_generate_total_seconds: {result.total_seconds:.6f}",
        f"tp_generate_decode_tokens_per_second: {result.decode_tokens_per_second:.6f}",
        f"tp_generate_total_tokens_per_second: {result.total_tokens_per_second:.6f}",
        f"tp_generate_dispatch_calls: {dispatch.calls}",
        f"tp_generate_dispatch_fp8_weight_calls: {dispatch.fp8_weight_calls}",
        f"tp_generate_dispatch_eligible_cuda_calls: {dispatch.eligible_cuda_calls}",
        f"tp_generate_dispatch_cuda_kernel_hits: {dispatch.cuda_kernel_hits}",
        f"tp_generate_dispatch_fallback_calls: {dispatch.fallback_calls}",
        f"tp_generate_dispatch_fallback_disabled_cuda_kernel: {dispatch.fallback_disabled_cuda_kernel}",
        f"tp_generate_dispatch_fallback_missing_scale: {dispatch.fallback_missing_scale}",
        f"tp_generate_dispatch_fallback_hidden_not_cuda: {dispatch.fallback_hidden_not_cuda}",
        f"tp_generate_dispatch_fallback_weight_not_cuda: {dispatch.fallback_weight_not_cuda}",
        f"tp_generate_dispatch_fallback_scale_not_cuda: {dispatch.fallback_scale_not_cuda}",
        f"tp_generate_dispatch_fallback_weight_dtype: {dispatch.fallback_weight_dtype}",
        f"tp_generate_dispatch_fallback_scale_dtype: {dispatch.fallback_scale_dtype}",
        f"tp_generate_dispatch_fallback_hidden_alignment: {dispatch.fallback_hidden_alignment}",
        f"tp_generate_dispatch_fallback_weight_alignment: {dispatch.fallback_weight_alignment}",
        f"tp_generate_dispatch_moe_calls: {dispatch.moe_calls}",
        f"tp_generate_dispatch_moe_packed_calls: {dispatch.moe_packed_calls}",
        f"tp_generate_dispatch_moe_loop_calls: {dispatch.moe_loop_calls}",
        f"tp_generate_dispatch_moe_assignments: {dispatch.moe_assignments}",
        f"tp_generate_dispatch_moe_local_assignments: {dispatch.moe_local_assignments}",
        f"tp_generate_dispatch_moe_active_expert_groups: {dispatch.moe_active_expert_groups}",
        f"tp_generate_dispatch_moe_empty_local_dispatches: {dispatch.moe_empty_local_dispatches}",
        f"tp_generate_dispatch_moe_max_group_tokens: {dispatch.moe_max_group_tokens}",
        f"tp_generate_dispatch_moe_native_expert_calls: {dispatch.moe_native_expert_calls}",
        f"tp_generate_dispatch_moe_native_expert_eligible: {dispatch.moe_native_expert_eligible}",
        f"tp_generate_dispatch_moe_native_expert_hits: {dispatch.moe_native_expert_hits}",
        f"tp_generate_dispatch_moe_native_expert_fallbacks: {dispatch.moe_native_expert_fallbacks}",
        f"tp_generate_dispatch_moe_native_expert_fallback_disabled: {dispatch.moe_native_expert_fallback_disabled}",
        f"tp_generate_dispatch_moe_native_expert_fallback_missing_scale: {dispatch.moe_native_expert_fallback_missing_scale}",
        f"tp_generate_dispatch_moe_native_expert_fallback_dtype: {dispatch.moe_native_expert_fallback_dtype}",
        f"tp_generate_dispatch_moe_native_expert_fallback_device: {dispatch.moe_native_expert_fallback_device}",
        f"tp_generate_dispatch_moe_native_expert_fallback_shape: {dispatch.moe_native_expert_fallback_shape}",
        f"tp_generate_dispatch_moe_native_expert_fallback_group_tokens: {dispatch.moe_native_expert_fallback_group_tokens}",
        f"tp_generate_dispatch_moe_native_expert_fallback_exception: {dispatch.moe_native_expert_fallback_exception}",
        f"tp_generate_dispatch_moe_native_expert_max_group_tokens: {dispatch.moe_native_expert_max_group_tokens}",
        f"tp_generate_all_finite: {result.all_finite}",
    ]
    memory = result.cuda_memory
    if memory.available:
        lines.extend(
            [
                f"tp_generate_cuda_memory_free: {memory.free_bytes}",
                f"tp_generate_cuda_memory_total: {memory.total_bytes}",
                f"tp_generate_cuda_max_allocated: {memory.max_allocated}",
                f"tp_generate_cuda_max_reserved: {memory.max_reserved}",
            ]
        )
    else:
        lines.append("tp_generate_cuda_memory: unavailable")
    if result.rank == 0:
        lines.extend(
            [
                f"tp_generate_generated_token_ids: {','.join(str(token) for token in result.generated_token_ids)}",
                f"tp_generate_text: {result.text}",
            ]
        )
    return lines


def _format_tp_benchmark_result(result: GenerateResult) -> list[str]:
    return ["tp_benchmark_iterations: 1", *_format_tp_generate_result(result)]



def _summarize_tp_reference_forward(
    manifest: Manifest,
    runtime_config: RuntimeConfig,
    prompt: str,
    max_tokens: int,
    world_size: int | None,
    rank: int | None,
    local_rank: int | None,
    backend: str,
    init_method: str | None,
    device: str | None,
) -> list[str]:
    import torch
    from transformers import AutoTokenizer

    launch = _tp_launch_from_args(world_size, rank, local_rank, backend, init_method, device)
    tp = TensorParallel(world_size=launch.world_size, rank=launch.rank)
    mapping = build_language_model_mapping(manifest, strict=True, tensor_parallel=tp)
    with TpRuntime(launch) as runtime:
        tokenizer = AutoTokenizer.from_pretrained(manifest.model_dir, local_files_only=True, trust_remote_code=True)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"][:, :max_tokens].to(runtime.device)
        with TensorLoader(manifest) as loader:
            logits = tp_language_model(input_ids, mapping, runtime_config, ReferenceWeights(loader, device=str(runtime.device)), runtime)
        last_logits = logits[:, -1].float()
        top_value, top_index = torch.max(last_logits, dim=-1)
        runtime.barrier()
        lines = [
            f"tp_reference_backend: {backend}",
            f"tp_reference_world_size: {launch.world_size}",
            f"tp_reference_rank: {launch.rank}",
            f"tp_reference_device: {runtime.device}",
            f"tp_reference_prompt_tokens: {input_ids.numel()}",
            f"tp_reference_layers: {len(mapping.layers)}",
            f"tp_reference_logits_shape: {tuple(logits.shape)}",
            f"tp_reference_logits_dtype: {logits.dtype}",
            f"tp_reference_logits_finite: {torch.isfinite(logits.float()).all().item()}",
        ]
        lines.extend(_cuda_memory_lines("tp_reference", runtime.device))
        if launch.rank == 0:
            lines.extend(
                [
                    f"tp_reference_next_token: {int(top_index.item())}",
                    f"tp_reference_next_logit: {float(top_value.item())}",
                ]
            )
        runtime.barrier()
        return lines


def _summarize_tp_load_smoke(
    manifest: Manifest,
    world_size: int | None,
    rank: int | None,
    local_rank: int | None,
    backend: str,
    init_method: str | None,
    device: str | None,
) -> list[str]:
    launch = _tp_launch_from_args(world_size, rank, local_rank, backend, init_method, device)
    tp = TensorParallel(world_size=launch.world_size, rank=launch.rank)
    mapping = build_language_model_mapping(manifest, strict=True, tensor_parallel=tp)
    with TpRuntime(launch) as runtime:
        with TensorLoader(manifest) as loader:
            weights = MappedWeights(loader, mapping, device=str(runtime.device))
            stats = weights.preload()
        first_moe = mapping.layers[0].mlp
        runtime.barrier()
        lines = [
            f"tp_load_backend: {backend}",
            f"tp_load_world_size: {launch.world_size}",
            f"tp_load_rank: {launch.rank}",
            f"tp_load_device: {runtime.device}",
            f"tp_load_expert_range: [{first_moe.expert_start},{first_moe.expert_end})",
            f"tp_load_mapped_tensors: {len(mapping.mapped_tensor_names)}",
            f"tp_load_loaded_tensors: {stats.tensor_count}",
            f"tp_load_mapped_bytes: {mapped_tensor_bytes(mapping)}",
            f"tp_load_loaded_bytes: {stats.bytes}",
        ]
        lines.extend(_cuda_memory_lines("tp_load", runtime.device))
        lines.extend(f"tp_load_shard_{line.strip()}" for line in _summarize_tp_shards(mapping))
        lines.extend(_tensor_report_lines("tp_load", weights, _representative_tensor_names(mapping)))
        runtime.barrier()
        return lines


def _summarize_reference_decode(
    manifest: Manifest,
    config: RuntimeConfig,
    mapping: LanguageModelMapping,
    prompt: str,
    device: str,
    max_tokens: int,
) -> list[str]:
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(manifest.model_dir, local_files_only=True, trust_remote_code=True)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"][:, :max_tokens].to(device)
    with TensorLoader(manifest) as loader:
        weights = ReferenceWeights(loader, device=device)
        state = DecodeState.empty(mapping, config)
        step_tokens: list[int] = []
        step_finite: list[bool] = []
        for i in range(input_ids.shape[1]):
            logits = decode_step(input_ids[:, i : i + 1], mapping, config, weights, state)
            last = logits[:, -1].float()
            step_tokens.append(int(torch.argmax(last, dim=-1).item()))
            step_finite.append(bool(torch.isfinite(last).all().item()))
    return [
        f"reference_decode_device: {device}",
        f"reference_decode_prompt_tokens: {input_ids.numel()}",
        f"reference_decode_steps: {len(step_tokens)}",
        f"reference_decode_all_finite: {all(step_finite)}",
        f"reference_decode_next_tokens: {','.join(str(t) for t in step_tokens)}",
    ]


def _summarize_tensor_load(manifest: Manifest, name: str, device: str | None) -> list[str]:
    with TensorLoader(manifest) as loader:
        info = loader.tensor_info(name)
        tensor = loader.tensor(name, device=device or "cpu")
        return [
            f"tensor_name: {info.name}",
            f"tensor_dtype: {info.dtype}",
            f"tensor_shape: {info.shape}",
            f"tensor_shard: {info.shard}",
            f"tensor_payload_bytes: {info.nbytes}",
            f"torch_dtype: {tensor.dtype}",
            f"torch_device: {tensor.device}",
            f"torch_numel: {tensor.numel()}",
        ]


def _summarize_reference_prefill(
    manifest: Manifest,
    config: RuntimeConfig,
    mapping: LanguageModelMapping,
    prompt: str,
    device: str,
    layer_index: int | None,
    max_tokens: int,
) -> list[str]:
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(manifest.model_dir, local_files_only=True, trust_remote_code=True)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"][:, :max_tokens].to(device)
    layer = _reference_layer(mapping, layer_index)
    with TensorLoader(manifest) as loader:
        embed_weight = loader.tensor_shard(mapping.embed_tokens, device=device)
        hidden = embedding(input_ids, embed_weight)
        out = decoder_layer(hidden, layer, config, ReferenceWeights(loader, device=device))
    return [
        f"reference_device: {device}",
        f"reference_prompt_tokens: {input_ids.numel()}",
        f"reference_max_tokens: {max_tokens}",
        f"reference_token_id_examples: {','.join(str(x) for x in input_ids.reshape(-1)[:8].tolist())}",
        f"reference_layer: {layer.index}",
        f"reference_layer_type: {layer.layer_type}",
        f"reference_hidden_shape: {tuple(hidden.shape)}",
        f"reference_output_shape: {tuple(out.shape)}",
        f"reference_output_dtype: {out.dtype}",
        f"reference_output_finite: {torch.isfinite(out.float()).all().item()}",
    ]


def _summarize_reference_forward(
    manifest: Manifest,
    config: RuntimeConfig,
    mapping: LanguageModelMapping,
    prompt: str,
    device: str,
    max_tokens: int,
) -> list[str]:
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(manifest.model_dir, local_files_only=True, trust_remote_code=True)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"][:, :max_tokens].to(device)
    with TensorLoader(manifest) as loader:
        logits = language_model(input_ids, mapping, config, ReferenceWeights(loader, device=device))
    return [
        f"reference_device: {device}",
        f"reference_prompt_tokens: {input_ids.numel()}",
        f"reference_max_tokens: {max_tokens}",
        f"reference_token_id_examples: {','.join(str(x) for x in input_ids.reshape(-1)[:8].tolist())}",
        f"reference_layers: {len(mapping.layers)}",
        f"reference_logits_shape: {tuple(logits.shape)}",
        f"reference_logits_dtype: {logits.dtype}",
        f"reference_logits_finite: {torch.isfinite(logits.float()).all().item()}",
    ]


def _run_tp_worker(args: argparse.Namespace) -> int:
    launch = _tp_launch_from_args(
        args.tp_world_size,
        args.tp_rank,
        args.tp_local_rank,
        args.tp_backend,
        args.tp_init_method,
        args.tp_device,
    )
    try:
        with TpRuntime(launch) as runtime:
            run_worker_protocol_loop(WorkerState(launch), runtime)
    except TpRuntimeError as exc:
        raise CliError(str(exc)) from exc
    return 0


def _run_tp_service(args: argparse.Namespace, model_dir: Path) -> int:
    launch = _tp_launch_from_args(
        args.tp_world_size,
        args.tp_rank,
        args.tp_local_rank,
        args.tp_backend,
        args.tp_init_method,
        args.tp_device,
    )
    config = ServiceConfig(
        host=args.tp_host,
        port=args.tp_port,
        model_dir=str(model_dir),
        max_active_requests=args.tp_service_max_active,
        max_pending_requests=args.tp_service_max_pending,
        batch_step_mode=args.tp_service_batch_step_mode,
    )
    try:
        with TpRuntime(launch) as runtime:
            serve_worker_http(
                WorkerState(
                    launch,
                    max_active_requests=args.tp_service_max_active,
                    max_pending_requests=args.tp_service_max_pending,
                    batch_step_mode=args.tp_service_batch_step_mode,
                ),
                runtime,
                config,
            )
    except (TpRuntimeError, OSError) as exc:
        raise CliError(str(exc)) from exc
    return 0


def _reference_layer(mapping: LanguageModelMapping, layer_index: int | None):
    if layer_index is None:
        for layer in mapping.layers:
            if layer.layer_type == "full_attention":
                return layer
        raise CliError("checkpoint has no full_attention layer for reference prefill")
    if layer_index < 0 or layer_index >= len(mapping.layers):
        raise CliError(f"reference layer index out of range: {layer_index}")
    layer = mapping.layers[layer_index]
    if layer.layer_type != "full_attention":
        raise CliError(f"reference layer {layer_index} is {layer.layer_type}, expected full_attention")
    return layer


def run(args: argparse.Namespace) -> int:
    model_dir = args.model.expanduser().resolve()
    if not model_dir.is_dir():
        raise CliError(f"model path is not a directory: {model_dir}")

    try:
        manifest = build_manifest(model_dir)
    except CheckpointError as exc:
        raise CliError(str(exc)) from exc
    if args.tp_worker:
        return _run_tp_worker(args)
    if args.tp_service:
        return _run_tp_service(args, model_dir)
    config = manifest.config
    print("Loaded model config")
    print(f"model_dir: {model_dir}")
    for line in _summarize_config(config):
        print(line)
    if args.inspect_config:
        try:
            runtime_config = parse_runtime_config(config)
        except ConfigError as exc:
            raise CliError(str(exc)) from exc
        print("Loaded runtime config")
        for line in _summarize_runtime_config(runtime_config):
            print(line)
    if args.inspect_checkpoint:
        print("Loaded checkpoint manifest")
        for line in _summarize_manifest(manifest):
            print(line)
    if args.smoke_fp8:
        report = inspect_fp8_checkpoint(manifest)
        print("FP8 checkpoint smoke")
        for line in _summarize_fp8(report):
            print(line)
        if not report.ok:
            raise CliError("FP8 checkpoint smoke failed")
    if args.inspect_mapping:
        try:
            mapping = build_language_model_mapping(manifest, strict=True)
        except MappingError as exc:
            raise CliError(str(exc)) from exc
        print("Loaded language model mapping")
        for line in _summarize_mapping(mapping):
            print(line)
    if args.inspect_tp:
        try:
            print("Tensor-parallel mapping layout")
            for line in _summarize_tp_mapping(manifest, args.inspect_tp):
                print(line)
        except MappingError as exc:
            raise CliError(str(exc)) from exc
    if args.inspect_tensor:
        try:
            print("Loaded tensor payload")
            for line in _summarize_tensor_load(manifest, args.inspect_tensor, args.tensor_device):
                print(line)
        except LoaderError as exc:
            raise CliError(str(exc)) from exc
    if args.tp_load_smoke:
        try:
            print("TP mapped weight load smoke")
            for line in _summarize_tp_load_smoke(
                manifest,
                args.tp_world_size,
                args.tp_rank,
                args.tp_local_rank,
                args.tp_backend,
                args.tp_init_method,
                args.tp_device,
            ):
                print(line)
        except (MappingError, LoaderError, TpRuntimeError) as exc:
            raise CliError(str(exc)) from exc
    if args.tp_runtime_smoke:
        try:
            runtime_config = parse_runtime_config(config)
            print("TP runtime smoke")
            for line in _summarize_tp_runtime_smoke(
                manifest,
                runtime_config,
                args.tp_world_size,
                args.tp_rank,
                args.tp_local_rank,
                args.tp_backend,
                args.tp_init_method,
                args.tp_device,
            ):
                print(line)
        except (ConfigError, MappingError, TpRuntimeError) as exc:
            raise CliError(str(exc)) from exc
    if args.tp_reference_forward:
        try:
            runtime_config = parse_runtime_config(config)
            print("TP reference forward smoke")
            for line in _summarize_tp_reference_forward(
                manifest,
                runtime_config,
                args.prompt,
                args.reference_max_tokens,
                args.tp_world_size,
                args.tp_rank,
                args.tp_local_rank,
                args.tp_backend,
                args.tp_init_method,
                args.tp_device,
            ):
                print(line)
        except (ConfigError, MappingError, LoaderError, TpRuntimeError) as exc:
            raise CliError(str(exc)) from exc
        except Exception as exc:
            raise CliError(f"TP reference forward failed: {exc}") from exc
    if args.tp_generate:
        try:
            runtime_config = parse_runtime_config(config)
            print("TP resident greedy generation")
            for line in _summarize_tp_generate(
                manifest,
                runtime_config,
                args.prompt,
                args.max_new_tokens,
                args.tp_world_size,
                args.tp_rank,
                args.tp_local_rank,
                args.tp_backend,
                args.tp_init_method,
                args.tp_device,
            ):
                print(line)
        except (ConfigError, EngineError, MappingError, LoaderError, TpRuntimeError, CliError) as exc:
            raise CliError(str(exc)) from exc
        except Exception as exc:
            raise CliError(f"TP resident generation failed: {exc}") from exc
        return 0
    if args.tp_benchmark:
        try:
            runtime_config = parse_runtime_config(config)
            print("TP resident generation benchmark")
            launch = _tp_launch_from_args(
                args.tp_world_size,
                args.tp_rank,
                args.tp_local_rank,
                args.tp_backend,
                args.tp_init_method,
                args.tp_device,
            )
            runner = TpModelRunner(manifest, runtime_config, launch)
            for line in _format_tp_benchmark_result(runner.generate(args.prompt, args.max_new_tokens)):
                print(line)
        except (ConfigError, EngineError, MappingError, LoaderError, TpRuntimeError, CliError) as exc:
            raise CliError(str(exc)) from exc
        except Exception as exc:
            raise CliError(f"TP benchmark failed: {exc}") from exc
        return 0
    if args.reference_prefill:
        try:
            runtime_config = parse_runtime_config(config)
            mapping = build_language_model_mapping(manifest, strict=True)
            device = args.reference_device or args.tensor_device or "cpu"
            print("Reference prefill smoke")
            for line in _summarize_reference_prefill(
                manifest,
                runtime_config,
                mapping,
                args.prompt,
                device,
                args.reference_layer,
                args.reference_max_tokens,
            ):
                print(line)
        except (ConfigError, MappingError, LoaderError) as exc:
            raise CliError(str(exc)) from exc
        except Exception as exc:
            raise CliError(f"reference prefill failed: {exc}") from exc
    if args.reference_forward:
        try:
            runtime_config = parse_runtime_config(config)
            mapping = build_language_model_mapping(manifest, strict=True)
            device = args.reference_device or args.tensor_device or "cpu"
            print("Reference forward smoke")
            for line in _summarize_reference_forward(
                manifest,
                runtime_config,
                mapping,
                args.prompt,
                device,
                args.reference_max_tokens,
            ):
                print(line)
        except (ConfigError, MappingError, LoaderError) as exc:
            raise CliError(str(exc)) from exc
        except Exception as exc:
            raise CliError(f"reference forward failed: {exc}") from exc
    if args.reference_decode_smoke:
        try:
            runtime_config = parse_runtime_config(config)
            mapping = build_language_model_mapping(manifest, strict=True)
            device = args.reference_device or args.tensor_device or "cpu"
            print("Reference decode smoke")
            for line in _summarize_reference_decode(
                manifest,
                runtime_config,
                mapping,
                args.prompt,
                device,
                args.reference_max_tokens,
            ):
                print(line)
        except (ConfigError, MappingError, LoaderError) as exc:
            raise CliError(str(exc)) from exc
        except Exception as exc:
            raise CliError(f"reference decode failed: {exc}") from exc
    print(f"prompt_tokens: pending tokenizer ({len(args.prompt)} prompt chars)")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print("inference: not implemented yet")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen36-run",
        description="Run Qwen3.6 FP8 checkpoints on RTX 2080 Ti.",
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to a Hugging Face model snapshot directory.")
    parser.add_argument("--prompt", required=True, help="Prompt text to generate from.")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Maximum number of tokens to generate.")
    parser.add_argument(
        "--inspect-config",
        action="store_true",
        help="Validate and summarize runtime dimensions derived from config.json.",
    )
    parser.add_argument(
        "--inspect-checkpoint",
        action="store_true",
        help="Print safetensors manifest metadata without reading tensor payloads.",
    )
    parser.add_argument(
        "--smoke-fp8",
        action="store_true",
        help="Validate FP8 tensors have linked scale metadata before inference.",
    )
    parser.add_argument(
        "--inspect-mapping",
        action="store_true",
        help="Validate and summarize the text MoE tensor mapping.",
    )
    parser.add_argument(
        "--inspect-tp",
        type=int,
        metavar="WORLD_SIZE",
        help="Validate per-rank tensor-parallel sharding for the given world size (e.g. 4).",
    )
    parser.add_argument(
        "--inspect-tensor",
        help="Read one tensor payload by name and print its location and decoded shape.",
    )
    parser.add_argument(
        "--tensor-device",
        help="Optionally copy --inspect-tensor to this torch device, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--tp-load-smoke",
        action="store_true",
        help="Load only this TP rank's mapped tensors and report resident weight bytes.",
    )
    parser.add_argument(
        "--tp-runtime-smoke",
        action="store_true",
        help="Initialize the TP runtime and print this rank's tensor-parallel mapping summary.",
    )
    parser.add_argument(
        "--tp-reference-forward",
        action="store_true",
        help="Run the prompt through the TP reference language model and report logits metadata.",
    )
    parser.add_argument(
        "--tp-generate",
        action="store_true",
        help="Run resident tensor-parallel greedy generation with mapped weights.",
    )
    parser.add_argument(
        "--tp-benchmark",
        action="store_true",
        help="Run resident tensor-parallel generation and print baseline timing/throughput metrics.",
    )
    parser.add_argument(
        "--tp-worker",
        action="store_true",
        help="Run a resident tensor-parallel worker that reads JSONL commands on rank 0.",
    )
    parser.add_argument(
        "--tp-service",
        action="store_true",
        help="Run a resident tensor-parallel HTTP/SSE service on rank 0.",
    )
    parser.add_argument("--tp-host", default="127.0.0.1", help="Host interface for --tp-service.")
    parser.add_argument("--tp-port", type=int, default=8000, help="TCP port for --tp-service.")
    parser.add_argument(
        "--tp-service-max-active",
        type=int,
        default=DEFAULT_MAX_ACTIVE_REQUESTS,
        help="Maximum active generation requests admitted by --tp-service.",
    )
    parser.add_argument(
        "--tp-service-max-pending",
        type=int,
        default=DEFAULT_MAX_PENDING_REQUESTS,
        help="Maximum pending generation requests queued by --tp-service.",
    )
    parser.add_argument(
        "--tp-service-batch-step-mode",
        choices=(STEP_MODE_COOPERATIVE, STEP_MODE_LEGACY),
        default=STEP_MODE_COOPERATIVE,
        help="Default service batch-step mode metadata; SSE streams use cooperative ticks.",
    )
    parser.add_argument("--tp-world-size", type=int, help="TP runtime world size; defaults to WORLD_SIZE or 1.")
    parser.add_argument("--tp-rank", type=int, help="TP runtime global rank; defaults to RANK or 0.")
    parser.add_argument("--tp-local-rank", type=int, help="TP runtime local CUDA rank; defaults to LOCAL_RANK or rank.")
    parser.add_argument("--tp-backend", choices=("nccl", "gloo"), default="nccl", help="torch.distributed backend.")
    parser.add_argument("--tp-init-method", help="torch.distributed init method, e.g. tcp://127.0.0.1:29500.")
    parser.add_argument("--tp-device", help="Torch device override for this TP rank, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--reference-prefill",
        action="store_true",
        help="Tokenize the prompt and run embedding plus one full-attention reference decoder layer.",
    )
    parser.add_argument(
        "--reference-forward",
        action="store_true",
        help="Tokenize the prompt and run the full reference language model to logits.",
    )
    parser.add_argument(
        "--reference-decode-smoke",
        action="store_true",
        help="Run multi-step stateful decode on the dense reference path and report per-step tokens.",
    )
    parser.add_argument(
        "--reference-layer",
        type=int,
        help="Full-attention layer index for --reference-prefill; defaults to the first full-attention layer.",
    )
    parser.add_argument(
        "--reference-max-tokens",
        type=int,
        default=8,
        help="Maximum prompt tokens to feed through --reference-prefill.",
    )
    parser.add_argument(
        "--reference-device",
        help="Torch device for --reference-prefill, e.g. cpu or cuda:0.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.reference_max_tokens <= 0:
        parser.error("--reference-max-tokens must be positive")
    if args.tp_world_size is not None and args.tp_world_size <= 0:
        parser.error("--tp-world-size must be positive")
    if args.tp_rank is not None and args.tp_rank < 0:
        parser.error("--tp-rank must be non-negative")
    if args.tp_world_size is not None and args.tp_rank is not None and args.tp_rank >= args.tp_world_size:
        parser.error("--tp-rank must be in [0, --tp-world-size)")
    if args.tp_local_rank is not None and args.tp_local_rank < 0:
        parser.error("--tp-local-rank must be non-negative")
    if args.tp_port <= 0 or args.tp_port > 65535:
        parser.error("--tp-port must be in [1, 65535]")
    if args.tp_service_max_active <= 0:
        parser.error("--tp-service-max-active must be positive")
    if args.tp_service_max_pending <= 0:
        parser.error("--tp-service-max-pending must be positive")
    if args.inspect_tp is not None and args.inspect_tp <= 0:
        parser.error("--inspect-tp must be positive")
    try:
        return run(args)
    except CliError as exc:
        parser.exit(2, f"qwen36-run: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
