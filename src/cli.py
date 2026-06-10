from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path

from checkpoint import CheckpointError, Manifest, build_manifest
from decode_state import DecodeState
from engine import EngineError, GenerateResult, TpModelRunner, TpModelSession
from fp8_smoke import Fp8SmokeReport, inspect_fp8_checkpoint
from loader import LoaderError, TensorLoader
from reference_ops import ReferenceWeights, decode_step, decoder_layer, embedding, language_model
from runtime_config import ConfigError, RuntimeConfig, parse_runtime_config
from service import ServiceConfig, serve_worker_http
from tensor_parallel import TensorParallel
from tp_runtime import RuntimeProfileConfig, TpLaunchConfig, TpRuntime, TpRuntimeError, mapped_tensor_bytes, tp_decode_step, tp_language_model
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


@dataclass(frozen=True)
class TpConcurrentBenchmarkRun:
    results: list[GenerateResult]
    concurrency: int
    prefill_wall_seconds: float
    decode_wall_seconds: float
    total_wall_seconds: float
    batch_step_calls: int
    generated_tokens: int
    all_finite: bool



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
        f"runtime_paged_kv_metadata: {config.full_attention.paged_kv_metadata}",
        f"runtime_native_paged_attention: {config.full_attention.native_paged_attention}",
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
    profile_config: RuntimeProfileConfig | None = None,
    fast_decode: bool = False,
    cuda_graph_probe: bool = False,
) -> list[str]:
    launch = _tp_launch_from_args(world_size, rank, local_rank, backend, init_method, device)
    runner = TpModelRunner(manifest, runtime_config, launch, profile_config=profile_config)
    return _format_tp_generate_result(runner.generate(prompt, max_new_tokens, fast_decode=fast_decode, cuda_graph_probe=cuda_graph_probe))


def _format_tp_generate_result(result: GenerateResult) -> list[str]:
    dispatch = result.dispatch_stats
    lines = [
        f"tp_generate_backend: {result.backend}",
        f"tp_generate_world_size: {result.world_size}",
        f"tp_generate_rank: {result.rank}",
        f"tp_generate_local_rank: {result.local_rank}",
        f"tp_generate_device: {result.device}",
        f"tp_generate_fast_decode: {result.fast_decode}",
        f"tp_generate_cuda_graph_probe_enabled: {result.cuda_graph_probe.enabled}",
        f"tp_generate_cuda_graph_probe_eligible: {result.cuda_graph_probe.eligible}",
        f"tp_generate_cuda_graph_probe_reasons: {','.join(result.cuda_graph_probe.reasons)}",
        f"tp_generate_cuda_graph_probe_notes: {','.join(result.cuda_graph_probe.notes)}",
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
        f"tp_generate_dispatch_moe_single_token_dispatch_calls: {dispatch.moe_single_token_dispatch_calls}",
        f"tp_generate_dispatch_moe_single_token_dispatch_hits: {dispatch.moe_single_token_dispatch_hits}",
        f"tp_generate_dispatch_moe_single_token_local_assignments: {dispatch.moe_single_token_local_assignments}",
        f"tp_generate_dispatch_moe_assignments: {dispatch.moe_assignments}",
        f"tp_generate_dispatch_moe_local_assignments: {dispatch.moe_local_assignments}",
        f"tp_generate_dispatch_moe_active_expert_groups: {dispatch.moe_active_expert_groups}",
        f"tp_generate_dispatch_moe_empty_local_dispatches: {dispatch.moe_empty_local_dispatches}",
        f"tp_generate_dispatch_moe_max_group_tokens: {dispatch.moe_max_group_tokens}",
        f"tp_generate_dispatch_moe_packed_index_add_calls: {dispatch.moe_packed_index_add_calls}",
        f"tp_generate_dispatch_moe_packed_single_scatter_calls: {dispatch.moe_packed_single_scatter_calls}",
        f"tp_generate_dispatch_moe_group_size_1: {dispatch.moe_group_size_1}",
        f"tp_generate_dispatch_moe_group_size_2_to_4: {dispatch.moe_group_size_2_to_4}",
        f"tp_generate_dispatch_moe_group_size_5_to_8: {dispatch.moe_group_size_5_to_8}",
        f"tp_generate_dispatch_moe_group_size_over_8: {dispatch.moe_group_size_over_8}",
        f"tp_generate_dispatch_moe_native_assignment_calls: {dispatch.moe_native_assignment_calls}",
        f"tp_generate_dispatch_moe_native_assignment_eligible: {dispatch.moe_native_assignment_eligible}",
        f"tp_generate_dispatch_moe_native_assignment_hits: {dispatch.moe_native_assignment_hits}",
        f"tp_generate_dispatch_moe_native_assignment_fallbacks: {dispatch.moe_native_assignment_fallbacks}",
        f"tp_generate_dispatch_moe_native_assignment_fallback_small: {dispatch.moe_native_assignment_fallback_small}",
        f"tp_generate_dispatch_moe_native_assignment_fallback_device: {dispatch.moe_native_assignment_fallback_device}",
        f"tp_generate_dispatch_moe_native_assignment_fallback_dtype: {dispatch.moe_native_assignment_fallback_dtype}",
        f"tp_generate_dispatch_moe_native_assignment_fallback_shape: {dispatch.moe_native_assignment_fallback_shape}",
        f"tp_generate_dispatch_moe_native_assignment_fallback_exception: {dispatch.moe_native_assignment_fallback_exception}",
        f"tp_generate_dispatch_moe_native_assignment_offsets_calls: {dispatch.moe_native_assignment_offsets_calls}",
        f"tp_generate_dispatch_moe_native_assignment_offsets_eligible: {dispatch.moe_native_assignment_offsets_eligible}",
        f"tp_generate_dispatch_moe_native_assignment_offsets_hits: {dispatch.moe_native_assignment_offsets_hits}",
        f"tp_generate_dispatch_moe_native_assignment_offsets_fallbacks: {dispatch.moe_native_assignment_offsets_fallbacks}",
        f"tp_generate_dispatch_moe_native_assignment_offsets_fallback_disabled: {dispatch.moe_native_assignment_offsets_fallback_disabled}",
        f"tp_generate_dispatch_moe_native_assignment_offsets_fallback_small: {dispatch.moe_native_assignment_offsets_fallback_small}",
        f"tp_generate_dispatch_moe_native_assignment_offsets_fallback_capacity: {dispatch.moe_native_assignment_offsets_fallback_capacity}",
        f"tp_generate_dispatch_moe_native_assignment_offsets_fallback_device: {dispatch.moe_native_assignment_offsets_fallback_device}",
        f"tp_generate_dispatch_moe_native_assignment_offsets_fallback_dtype: {dispatch.moe_native_assignment_offsets_fallback_dtype}",
        f"tp_generate_dispatch_moe_native_assignment_offsets_fallback_shape: {dispatch.moe_native_assignment_offsets_fallback_shape}",
        f"tp_generate_dispatch_moe_native_assignment_offsets_fallback_exception: {dispatch.moe_native_assignment_offsets_fallback_exception}",
        f"tp_generate_dispatch_moe_native_assignment_offsets_capacity: {dispatch.moe_native_assignment_offsets_capacity}",
        f"tp_generate_dispatch_moe_native_assignment_parallel_calls: {dispatch.moe_native_assignment_parallel_calls}",
        f"tp_generate_dispatch_moe_native_assignment_parallel_eligible: {dispatch.moe_native_assignment_parallel_eligible}",
        f"tp_generate_dispatch_moe_native_assignment_parallel_hits: {dispatch.moe_native_assignment_parallel_hits}",
        f"tp_generate_dispatch_moe_native_assignment_parallel_fallbacks: {dispatch.moe_native_assignment_parallel_fallbacks}",
        f"tp_generate_dispatch_moe_native_assignment_parallel_fallback_disabled: {dispatch.moe_native_assignment_parallel_fallback_disabled}",
        f"tp_generate_dispatch_moe_native_assignment_parallel_fallback_small: {dispatch.moe_native_assignment_parallel_fallback_small}",
        f"tp_generate_dispatch_moe_native_assignment_parallel_fallback_capacity: {dispatch.moe_native_assignment_parallel_fallback_capacity}",
        f"tp_generate_dispatch_moe_native_assignment_parallel_fallback_device: {dispatch.moe_native_assignment_parallel_fallback_device}",
        f"tp_generate_dispatch_moe_native_assignment_parallel_fallback_dtype: {dispatch.moe_native_assignment_parallel_fallback_dtype}",
        f"tp_generate_dispatch_moe_native_assignment_parallel_fallback_shape: {dispatch.moe_native_assignment_parallel_fallback_shape}",
        f"tp_generate_dispatch_moe_native_assignment_parallel_fallback_missing_scale: {dispatch.moe_native_assignment_parallel_fallback_missing_scale}",
        f"tp_generate_dispatch_moe_native_assignment_parallel_fallback_exception: {dispatch.moe_native_assignment_parallel_fallback_exception}",
        f"tp_generate_dispatch_moe_native_assignment_parallel_capacity: {dispatch.moe_native_assignment_parallel_capacity}",
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
        f"tp_generate_dispatch_moe_native_scatter_calls: {dispatch.moe_native_scatter_calls}",
        f"tp_generate_dispatch_moe_native_scatter_hits: {dispatch.moe_native_scatter_hits}",
        f"tp_generate_dispatch_moe_native_scatter_fallbacks: {dispatch.moe_native_scatter_fallbacks}",
        f"tp_generate_dispatch_moe_native_scatter_fallback_small: {dispatch.moe_native_scatter_fallback_small}",
        f"tp_generate_dispatch_moe_native_scatter_fallback_device: {dispatch.moe_native_scatter_fallback_device}",
        f"tp_generate_dispatch_moe_native_scatter_fallback_dtype: {dispatch.moe_native_scatter_fallback_dtype}",
        f"tp_generate_dispatch_moe_native_scatter_fallback_shape: {dispatch.moe_native_scatter_fallback_shape}",
        f"tp_generate_dispatch_moe_native_scatter_fallback_exception: {dispatch.moe_native_scatter_fallback_exception}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_calls: {dispatch.moe_native_grouped_dispatch_calls}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_eligible: {dispatch.moe_native_grouped_dispatch_eligible}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_hits: {dispatch.moe_native_grouped_dispatch_hits}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_fallbacks: {dispatch.moe_native_grouped_dispatch_fallbacks}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_fallback_disabled: {dispatch.moe_native_grouped_dispatch_fallback_disabled}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_fallback_small: {dispatch.moe_native_grouped_dispatch_fallback_small}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_fallback_device: {dispatch.moe_native_grouped_dispatch_fallback_device}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_fallback_dtype: {dispatch.moe_native_grouped_dispatch_fallback_dtype}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_fallback_shape: {dispatch.moe_native_grouped_dispatch_fallback_shape}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_fallback_missing_scale: {dispatch.moe_native_grouped_dispatch_fallback_missing_scale}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_fallback_exception: {dispatch.moe_native_grouped_dispatch_fallback_exception}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_offsets_calls: {dispatch.moe_native_grouped_dispatch_offsets_calls}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_offsets_eligible: {dispatch.moe_native_grouped_dispatch_offsets_eligible}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_offsets_hits: {dispatch.moe_native_grouped_dispatch_offsets_hits}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_offsets_fallbacks: {dispatch.moe_native_grouped_dispatch_offsets_fallbacks}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_offsets_fallback_disabled: {dispatch.moe_native_grouped_dispatch_offsets_fallback_disabled}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_offsets_fallback_device: {dispatch.moe_native_grouped_dispatch_offsets_fallback_device}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_offsets_fallback_dtype: {dispatch.moe_native_grouped_dispatch_offsets_fallback_dtype}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_offsets_fallback_shape: {dispatch.moe_native_grouped_dispatch_offsets_fallback_shape}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_offsets_fallback_missing_scale: {dispatch.moe_native_grouped_dispatch_offsets_fallback_missing_scale}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_offsets_fallback_exception: {dispatch.moe_native_grouped_dispatch_offsets_fallback_exception}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_offsets_segmented_calls: {dispatch.moe_native_grouped_dispatch_offsets_segmented_calls}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_offsets_segmented_eligible: {dispatch.moe_native_grouped_dispatch_offsets_segmented_eligible}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_offsets_segmented_hits: {dispatch.moe_native_grouped_dispatch_offsets_segmented_hits}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_offsets_segmented_fallbacks: {dispatch.moe_native_grouped_dispatch_offsets_segmented_fallbacks}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_offsets_segmented_fallback_disabled: {dispatch.moe_native_grouped_dispatch_offsets_segmented_fallback_disabled}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_offsets_segmented_fallback_small: {dispatch.moe_native_grouped_dispatch_offsets_segmented_fallback_small}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_offsets_segmented_fallback_device: {dispatch.moe_native_grouped_dispatch_offsets_segmented_fallback_device}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_offsets_segmented_fallback_dtype: {dispatch.moe_native_grouped_dispatch_offsets_segmented_fallback_dtype}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_offsets_segmented_fallback_shape: {dispatch.moe_native_grouped_dispatch_offsets_segmented_fallback_shape}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_offsets_segmented_fallback_missing_scale: {dispatch.moe_native_grouped_dispatch_offsets_segmented_fallback_missing_scale}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_offsets_segmented_fallback_exception: {dispatch.moe_native_grouped_dispatch_offsets_segmented_fallback_exception}",
        f"tp_generate_dispatch_moe_native_grouped_dispatch_offsets_segmented_capacity: {dispatch.moe_native_grouped_dispatch_offsets_segmented_capacity}",
        f"tp_generate_dispatch_moe_native_tensor_core_calls: {dispatch.moe_native_tensor_core_calls}",
        f"tp_generate_dispatch_moe_native_tensor_core_eligible: {dispatch.moe_native_tensor_core_eligible}",
        f"tp_generate_dispatch_moe_native_tensor_core_hits: {dispatch.moe_native_tensor_core_hits}",
        f"tp_generate_dispatch_moe_native_tensor_core_fallbacks: {dispatch.moe_native_tensor_core_fallbacks}",
        f"tp_generate_dispatch_moe_native_tensor_core_fallback_disabled: {dispatch.moe_native_tensor_core_fallback_disabled}",
        f"tp_generate_dispatch_moe_native_tensor_core_fallback_small: {dispatch.moe_native_tensor_core_fallback_small}",
        f"tp_generate_dispatch_moe_native_tensor_core_fallback_device: {dispatch.moe_native_tensor_core_fallback_device}",
        f"tp_generate_dispatch_moe_native_tensor_core_fallback_dtype: {dispatch.moe_native_tensor_core_fallback_dtype}",
        f"tp_generate_dispatch_moe_native_tensor_core_fallback_shape: {dispatch.moe_native_tensor_core_fallback_shape}",
        f"tp_generate_dispatch_moe_native_tensor_core_fallback_exception: {dispatch.moe_native_tensor_core_fallback_exception}",
        f"tp_generate_paged_attention_calls: {result.paged_attention_stats.calls}",
        f"tp_generate_paged_attention_eligible: {result.paged_attention_stats.eligible}",
        f"tp_generate_paged_attention_dense_fallbacks: {result.paged_attention_stats.dense_fallbacks}",
        f"tp_generate_paged_attention_native_hits: {result.paged_attention_stats.native_hits}",
        f"tp_generate_paged_attention_fallback_disabled: {result.paged_attention_stats.fallback_disabled}",
        f"tp_generate_paged_attention_fallback_no_kernel: {result.paged_attention_stats.fallback_no_kernel}",
        f"tp_generate_paged_attention_fallback_prefill_or_multitoken: {result.paged_attention_stats.fallback_prefill_or_multitoken}",
        f"tp_generate_paged_attention_fallback_cpu: {result.paged_attention_stats.fallback_cpu}",
        f"tp_generate_paged_attention_fallback_per_request_pools: {result.paged_attention_stats.fallback_per_request_pools}",
        f"tp_generate_paged_attention_fallback_shape: {result.paged_attention_stats.fallback_shape}",
        f"tp_generate_all_finite: {result.all_finite}",
    ]
    lines.extend(_format_kv_cache_lines("tp_generate", result.kv_cache.to_dict()))
    lines.extend(_format_profile_lines("tp_generate", result.profile.to_dict()))
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


def _format_kv_cache_lines(prefix: str, kv_cache: dict[str, object]) -> list[str]:
    keys = [
        "full_attention_layers",
        "block_size",
        "valid_tokens_total",
        "capacity_tokens_total",
        "logical_blocks_total",
        "allocated_blocks_total",
        "free_blocks_total",
        "append_calls_total",
        "appended_tokens_total",
        "growth_events_total",
        "release_calls_total",
        "released_blocks_total",
        "contiguous_view_calls_total",
        "gather_view_calls_total",
        "gather_copied_tokens_total",
        "page_table_tensor_calls_total",
        "page_metadata_calls_total",
        "batched_page_table_calls_total",
        "page_table_entries_total",
        "non_contiguous_layers",
        "estimated_total_bytes",
        "max_valid_tokens",
        "max_capacity_tokens",
    ]
    return [f"{prefix}_kv_{key}: {kv_cache.get(key)}" for key in keys]


def _profile_scope_seconds(scopes: dict[object, object], predicate) -> float:
    total = 0.0
    for name, data in scopes.items():
        if not isinstance(data, dict):
            continue
        scope_name = str(name)
        if predicate(scope_name):
            total += float(data.get("total_seconds", 0.0))
    return total


def _profile_summary_denominator_seconds(scopes: dict[object, object], total_scope_seconds: float) -> float:
    decode_step_seconds = _profile_scope_seconds(scopes, lambda name: name in ("batch.decode_step.total", "decode_step.total"))
    prefill_layers_seconds = _profile_scope_seconds(scopes, lambda name: name == "layers_total")
    layers_seconds = _profile_scope_seconds(scopes, lambda name: name in ("batch.layers_total", "layers_total"))
    if decode_step_seconds > 0.0:
        return decode_step_seconds + prefill_layers_seconds
    if layers_seconds > 0.0:
        return layers_seconds
    return total_scope_seconds


def _profile_layer_total_seconds(scopes: dict[object, object], suffix: str) -> float:
    return _profile_scope_seconds(scopes, lambda name: name.startswith("layer.") and name.endswith(suffix))


def _profile_group_seconds(scopes: dict[object, object], *, layer_suffix: str, fallback_prefixes: tuple[str, ...]) -> float:
    layer_seconds = _profile_layer_total_seconds(scopes, layer_suffix)
    if layer_seconds > 0.0:
        return layer_seconds
    return _profile_scope_seconds(scopes, lambda name: name.startswith(fallback_prefixes))


def _format_profile_summary_lines(prefix: str, scopes: dict[object, object]) -> list[str]:
    total_seconds = _profile_scope_seconds(scopes, lambda _name: True)
    denominator_seconds = _profile_summary_denominator_seconds(scopes, total_seconds)
    collective_seconds = _profile_scope_seconds(scopes, lambda name: name.startswith("collective."))
    moe_seconds = _profile_group_seconds(scopes, layer_suffix=".moe.total", fallback_prefixes=("moe.",))
    full_attention_seconds = _profile_group_seconds(
        scopes,
        layer_suffix=".full_attention.total",
        fallback_prefixes=("full_attention.", "batch.full_attention."),
    )
    linear_attention_seconds = _profile_group_seconds(
        scopes,
        layer_suffix=".linear_attention.total",
        fallback_prefixes=("linear_attention.",),
    )
    dense_attention_seconds = _profile_scope_seconds(scopes, lambda name: name.endswith(".dense_fallback"))
    batch_kv_seconds = _profile_scope_seconds(scopes, lambda name: name.endswith(".batch_kv_tensors"))

    def pct(seconds: float) -> float:
        return (seconds / denominator_seconds * 100.0) if denominator_seconds > 0.0 else 0.0

    return [
        f"{prefix}_profile_total_scope_seconds: {total_seconds:.6f}",
        f"{prefix}_profile_summary_denominator_seconds: {denominator_seconds:.6f}",
        f"{prefix}_profile_collective_total_seconds: {collective_seconds:.6f}",
        f"{prefix}_profile_collective_percent: {pct(collective_seconds):.2f}",
        f"{prefix}_profile_moe_total_seconds: {moe_seconds:.6f}",
        f"{prefix}_profile_moe_percent: {pct(moe_seconds):.2f}",
        f"{prefix}_profile_full_attention_total_seconds: {full_attention_seconds:.6f}",
        f"{prefix}_profile_full_attention_percent: {pct(full_attention_seconds):.2f}",
        f"{prefix}_profile_linear_attention_total_seconds: {linear_attention_seconds:.6f}",
        f"{prefix}_profile_linear_attention_percent: {pct(linear_attention_seconds):.2f}",
        f"{prefix}_profile_dense_attention_fallback_total_seconds: {dense_attention_seconds:.6f}",
        f"{prefix}_profile_dense_attention_fallback_percent: {pct(dense_attention_seconds):.2f}",
        f"{prefix}_profile_batch_kv_tensors_total_seconds: {batch_kv_seconds:.6f}",
        f"{prefix}_profile_batch_kv_tensors_percent: {pct(batch_kv_seconds):.2f}",
    ]


def _format_profile_lines(prefix: str, profile: dict[str, object], *, top_n: int = 20) -> list[str]:
    enabled = bool(profile.get("enabled", False))
    sync_cuda = bool(profile.get("sync_cuda", False))
    lines = [
        f"{prefix}_profile_enabled: {enabled}",
        f"{prefix}_profile_sync_cuda: {sync_cuda}",
    ]
    scopes = profile.get("scopes", {})
    if not enabled or not isinstance(scopes, dict):
        return lines
    lines.extend(_format_profile_summary_lines(prefix, scopes))
    ranked = sorted(
        ((str(name), data) for name, data in scopes.items() if isinstance(data, dict)),
        key=lambda item: float(item[1].get("total_seconds", 0.0)),
        reverse=True,
    )
    for index, (name, data) in enumerate(ranked[:top_n]):
        lines.extend(
            [
                f"{prefix}_profile_scope_{index}_name: {name}",
                f"{prefix}_profile_scope_{index}_calls: {data.get('calls')}",
                f"{prefix}_profile_scope_{index}_total_seconds: {float(data.get('total_seconds', 0.0)):.6f}",
                f"{prefix}_profile_scope_{index}_avg_seconds: {float(data.get('avg_seconds', 0.0)):.6f}",
                f"{prefix}_profile_scope_{index}_max_seconds: {float(data.get('max_seconds', 0.0)):.6f}",
                f"{prefix}_profile_scope_{index}_bytes: {data.get('bytes')}",
            ]
        )
    return lines


def _profile_config_from_args(args: argparse.Namespace) -> RuntimeProfileConfig:
    mode = getattr(args, "tp_profile", "off")
    return RuntimeProfileConfig(
        enabled=mode != "off",
        sync_cuda=mode == "sync",
        layer_detail=bool(getattr(args, "tp_profile_layer_detail", False)),
    )


def _native_paged_attention_override_from_args(args: argparse.Namespace) -> bool | None:
    mode = getattr(args, "tp_native_paged_attention", "config")
    if mode == "config":
        return None
    return mode == "on"


def _apply_native_paged_attention_override(config: RuntimeConfig, override: bool | None) -> RuntimeConfig:
    if override is None:
        return config
    return replace(config, full_attention=replace(config.full_attention, native_paged_attention=override))


def _runtime_config_from_args(config: dict, args: argparse.Namespace) -> RuntimeConfig:
    return _apply_native_paged_attention_override(
        parse_runtime_config(config),
        _native_paged_attention_override_from_args(args),
    )


def _benchmark_prompt_from_args(manifest: Manifest, args: argparse.Namespace) -> str:
    target = args.tp_benchmark_prompt_tokens
    if target is None:
        return args.prompt
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(manifest.model_dir, local_files_only=True, trust_remote_code=True)
    prompt = args.prompt or " "
    tokenized = tokenizer(prompt, add_special_tokens=True)
    token_ids = tokenized["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    token_ids = list(token_ids)
    if not token_ids:
        token_ids = [0]
    repeated = (token_ids * ((target + len(token_ids) - 1) // len(token_ids)))[:target]
    return tokenizer.decode(repeated, skip_special_tokens=False)


def _format_tp_benchmark_result(result: GenerateResult) -> list[str]:
    return ["tp_benchmark_iterations: 1", *_format_tp_generate_result(result)]


def _run_tp_concurrent_benchmark_once(
    session: TpModelSession,
    prompt: str,
    max_new_tokens: int,
    concurrency: int,
    *,
    fast_decode: bool = False,
) -> TpConcurrentBenchmarkRun:
    states = []
    results: list[GenerateResult] = []
    total_start = time.perf_counter()
    prefill_start = total_start
    try:
        for _ in range(concurrency):
            states.append(session.start_generation(prompt, max_new_tokens, fast_decode=fast_decode))
        prefill_end = time.perf_counter()
        decode_start = prefill_end
        batch_step_calls = 0
        active = [state for state in states if not state.completed]
        while active:
            batch_step_calls += 1
            session.step_generations_batch(active)
            active = [state for state in active if not state.completed]
        decode_end = time.perf_counter()
        for state in states:
            results.append(session.finish_generation(state))
        total_end = time.perf_counter()
    except Exception:
        for state in states:
            if not state.result_built:
                state.decode_state.release()
        raise
    generated_tokens = sum(result.max_new_tokens for result in results)
    return TpConcurrentBenchmarkRun(
        results=results,
        concurrency=concurrency,
        prefill_wall_seconds=max(0.0, prefill_end - prefill_start),
        decode_wall_seconds=max(0.0, decode_end - decode_start),
        total_wall_seconds=max(0.0, total_end - total_start),
        batch_step_calls=batch_step_calls,
        generated_tokens=generated_tokens,
        all_finite=all(result.all_finite for result in results),
    )


def _format_tp_concurrent_benchmark_results(
    runs: list[TpConcurrentBenchmarkRun],
    *,
    warmup_iterations: int,
    prompt_target_tokens: int | None,
    print_runs: bool,
) -> list[str]:
    if not runs:
        raise CliError("concurrent benchmark requires at least one measured run")
    all_results = [result for run in runs for result in run.results]
    concurrency = runs[0].concurrency
    requests_total = sum(len(run.results) for run in runs)
    tokens_total = sum(run.generated_tokens for run in runs)
    batch_step_calls_total = sum(run.batch_step_calls for run in runs)
    decode_tps = [
        run.generated_tokens / run.decode_wall_seconds if run.decode_wall_seconds > 0 else float("inf")
        for run in runs
    ]
    wall_tps = [
        run.generated_tokens / run.total_wall_seconds if run.total_wall_seconds > 0 else float("inf")
        for run in runs
    ]
    effective_batch_sizes = [
        run.generated_tokens / run.batch_step_calls if run.batch_step_calls else 0.0
        for run in runs
    ]
    lines = [
        f"tp_benchmark_warmup_iterations: {warmup_iterations}",
        f"tp_benchmark_iterations: {len(runs)}",
        f"tp_benchmark_concurrency: {concurrency}",
        f"tp_benchmark_prompt_target_tokens: {prompt_target_tokens}",
        f"tp_benchmark_prompt_tokens_min: {min(result.prompt_tokens for result in all_results)}",
        f"tp_benchmark_prompt_tokens_max: {max(result.prompt_tokens for result in all_results)}",
        f"tp_benchmark_max_new_tokens: {all_results[0].max_new_tokens}",
        f"tp_benchmark_concurrent_iterations: {len(runs)}",
        f"tp_benchmark_concurrent_requests_total: {requests_total}",
        f"tp_benchmark_concurrent_tokens_total: {tokens_total}",
        f"tp_benchmark_concurrent_batch_step_calls_total: {batch_step_calls_total}",
        f"tp_benchmark_concurrent_prefill_wall_seconds_avg: {_avg(run.prefill_wall_seconds for run in runs):.6f}",
        f"tp_benchmark_concurrent_prefill_wall_seconds_min: {min(run.prefill_wall_seconds for run in runs):.6f}",
        f"tp_benchmark_concurrent_prefill_wall_seconds_max: {max(run.prefill_wall_seconds for run in runs):.6f}",
        f"tp_benchmark_concurrent_decode_wall_seconds_avg: {_avg(run.decode_wall_seconds for run in runs):.6f}",
        f"tp_benchmark_concurrent_decode_wall_seconds_min: {min(run.decode_wall_seconds for run in runs):.6f}",
        f"tp_benchmark_concurrent_decode_wall_seconds_max: {max(run.decode_wall_seconds for run in runs):.6f}",
        f"tp_benchmark_concurrent_total_wall_seconds_avg: {_avg(run.total_wall_seconds for run in runs):.6f}",
        f"tp_benchmark_concurrent_decode_tokens_per_second_avg: {_avg(decode_tps):.6f}",
        f"tp_benchmark_concurrent_wall_tokens_per_second_avg: {_avg(wall_tps):.6f}",
        f"tp_benchmark_concurrent_effective_batch_size_avg: {_avg(effective_batch_sizes):.6f}",
        f"tp_benchmark_concurrent_seconds_per_decode_token_avg: {_avg((run.decode_wall_seconds / run.generated_tokens if run.generated_tokens else 0.0) for run in runs):.6f}",
        f"tp_benchmark_concurrent_seconds_per_batch_step_avg: {_avg((run.decode_wall_seconds / run.batch_step_calls if run.batch_step_calls else 0.0) for run in runs):.6f}",
        f"tp_benchmark_concurrent_all_finite: {all(run.all_finite for run in runs)}",
        f"tp_benchmark_concurrent_per_request_decode_seconds_avg: {_avg(result.decode_seconds for result in all_results):.6f}",
        f"tp_benchmark_concurrent_per_request_decode_tokens_per_second_avg: {_avg(result.decode_tokens_per_second for result in all_results):.6f}",
        f"tp_benchmark_kv_estimated_total_bytes_max: {max(result.kv_cache.estimated_total_bytes for result in all_results)}",
        f"tp_benchmark_cuda_max_allocated_max: {max((result.cuda_memory.max_allocated or 0) for result in all_results)}",
        f"tp_benchmark_cuda_max_reserved_max: {max((result.cuda_memory.max_reserved or 0) for result in all_results)}",
    ]
    profile_results = [run.results[-1] for run in runs if run.results]
    lines.extend(_format_profile_lines("tp_benchmark", _merge_profile_results(profile_results), top_n=10))
    if print_runs:
        for index, run in enumerate(runs):
            run_decode_tps = run.generated_tokens / run.decode_wall_seconds if run.decode_wall_seconds > 0 else float("inf")
            lines.extend(
                [
                    f"tp_benchmark_run_{index}_concurrent_prefill_wall_seconds: {run.prefill_wall_seconds:.6f}",
                    f"tp_benchmark_run_{index}_concurrent_decode_wall_seconds: {run.decode_wall_seconds:.6f}",
                    f"tp_benchmark_run_{index}_concurrent_total_wall_seconds: {run.total_wall_seconds:.6f}",
                    f"tp_benchmark_run_{index}_concurrent_decode_tokens_per_second: {run_decode_tps:.6f}",
                    f"tp_benchmark_run_{index}_concurrent_batch_step_calls: {run.batch_step_calls}",
                    f"tp_benchmark_run_{index}_concurrent_generated_tokens: {run.generated_tokens}",
                ]
            )
    lines.extend(_format_tp_generate_result(all_results[-1]))
    return lines



def _format_tp_benchmark_results(
    results: list[GenerateResult],
    *,
    warmup_iterations: int,
    prompt_target_tokens: int | None,
    print_runs: bool,
) -> list[str]:
    if not results:
        raise CliError("benchmark requires at least one measured result")
    lines = [
        f"tp_benchmark_warmup_iterations: {warmup_iterations}",
        f"tp_benchmark_iterations: {len(results)}",
        f"tp_benchmark_prompt_target_tokens: {prompt_target_tokens}",
        f"tp_benchmark_prompt_tokens_min: {min(result.prompt_tokens for result in results)}",
        f"tp_benchmark_prompt_tokens_max: {max(result.prompt_tokens for result in results)}",
        f"tp_benchmark_max_new_tokens: {results[0].max_new_tokens}",
        f"tp_benchmark_prefill_seconds_avg: {_avg(result.prefill_seconds for result in results):.6f}",
        f"tp_benchmark_prefill_seconds_min: {min(result.prefill_seconds for result in results):.6f}",
        f"tp_benchmark_prefill_seconds_max: {max(result.prefill_seconds for result in results):.6f}",
        f"tp_benchmark_decode_seconds_avg: {_avg(result.decode_seconds for result in results):.6f}",
        f"tp_benchmark_decode_seconds_min: {min(result.decode_seconds for result in results):.6f}",
        f"tp_benchmark_decode_seconds_max: {max(result.decode_seconds for result in results):.6f}",
        f"tp_benchmark_total_seconds_avg: {_avg(result.total_seconds for result in results):.6f}",
        f"tp_benchmark_decode_tokens_per_second_avg: {_avg(result.decode_tokens_per_second for result in results):.6f}",
        f"tp_benchmark_total_tokens_per_second_avg: {_avg(result.total_tokens_per_second for result in results):.6f}",
        f"tp_benchmark_kv_estimated_total_bytes_max: {max(result.kv_cache.estimated_total_bytes for result in results)}",
        f"tp_benchmark_cuda_max_allocated_max: {max((result.cuda_memory.max_allocated or 0) for result in results)}",
        f"tp_benchmark_cuda_max_reserved_max: {max((result.cuda_memory.max_reserved or 0) for result in results)}",
    ]
    lines.extend(_format_profile_lines("tp_benchmark", _merge_profile_results(results), top_n=10))
    if print_runs:
        for index, result in enumerate(results):
            lines.extend(
                [
                    f"tp_benchmark_run_{index}_prefill_seconds: {result.prefill_seconds:.6f}",
                    f"tp_benchmark_run_{index}_decode_seconds: {result.decode_seconds:.6f}",
                    f"tp_benchmark_run_{index}_total_seconds: {result.total_seconds:.6f}",
                    f"tp_benchmark_run_{index}_decode_tokens_per_second: {result.decode_tokens_per_second:.6f}",
                    f"tp_benchmark_run_{index}_kv_estimated_total_bytes: {result.kv_cache.estimated_total_bytes}",
                ]
            )
    lines.extend(_format_tp_generate_result(results[-1]))
    return lines


def _merge_profile_results(results: list[GenerateResult]) -> dict[str, object]:
    enabled = any(result.profile.enabled for result in results)
    sync_cuda = any(result.profile.sync_cuda for result in results)
    totals: dict[str, dict[str, float | int | None]] = {}
    for result in results:
        for name, data in result.profile.to_dict()["scopes"].items():
            scope = totals.setdefault(
                name,
                {
                    "calls": 0,
                    "total_seconds": 0.0,
                    "max_seconds": 0.0,
                    "min_seconds": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "bytes": 0,
                },
            )
            scope["calls"] = int(scope["calls"] or 0) + int(data["calls"])
            scope["total_seconds"] = float(scope["total_seconds"] or 0.0) + float(data["total_seconds"])
            scope["max_seconds"] = max(float(scope["max_seconds"] or 0.0), float(data["max_seconds"] or 0.0))
            min_seconds = data.get("min_seconds")
            if min_seconds is not None:
                scope["min_seconds"] = float(min_seconds) if scope["min_seconds"] is None else min(float(scope["min_seconds"]), float(min_seconds))
            scope["input_tokens"] = int(scope["input_tokens"] or 0) + int(data.get("input_tokens", 0))
            scope["output_tokens"] = int(scope["output_tokens"] or 0) + int(data.get("output_tokens", 0))
            scope["bytes"] = int(scope["bytes"] or 0) + int(data.get("bytes", 0))
    for scope in totals.values():
        calls = int(scope["calls"] or 0)
        scope["avg_seconds"] = float(scope["total_seconds"] or 0.0) / calls if calls else 0.0
    return {"enabled": enabled, "sync_cuda": sync_cuda, "scopes": totals}


def _avg(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0



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
            run_worker_protocol_loop(
                WorkerState(
                    launch,
                    profile_config=_profile_config_from_args(args),
                    native_paged_attention_override=_native_paged_attention_override_from_args(args),
                    fast_decode=args.tp_fast_decode,
                ),
                runtime,
            )
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
                    profile_config=_profile_config_from_args(args),
                    native_paged_attention_override=_native_paged_attention_override_from_args(args),
                    fast_decode=args.tp_fast_decode,
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
            runtime_config = _runtime_config_from_args(config, args)
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
            runtime_config = _runtime_config_from_args(config, args)
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
            runtime_config = _runtime_config_from_args(config, args)
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
            runtime_config = _runtime_config_from_args(config, args)
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
                _profile_config_from_args(args),
                fast_decode=args.tp_fast_decode,
                cuda_graph_probe=args.tp_cuda_graph_probe,
            ):
                print(line)
        except (ConfigError, EngineError, MappingError, LoaderError, TpRuntimeError, CliError) as exc:
            raise CliError(str(exc)) from exc
        except Exception as exc:
            raise CliError(f"TP resident generation failed: {exc}") from exc
        return 0
    if args.tp_benchmark:
        try:
            runtime_config = _runtime_config_from_args(config, args)
            concurrent = args.tp_benchmark_concurrency > 1
            print("TP resident concurrent generation benchmark" if concurrent else "TP resident generation benchmark")
            launch = _tp_launch_from_args(
                args.tp_world_size,
                args.tp_rank,
                args.tp_local_rank,
                args.tp_backend,
                args.tp_init_method,
                args.tp_device,
            )
            benchmark_prompt = _benchmark_prompt_from_args(manifest, args)
            with TpModelSession(manifest, runtime_config, launch, profile_config=_profile_config_from_args(args)) as session:
                if concurrent:
                    runs: list[TpConcurrentBenchmarkRun] = []
                    for _ in range(args.tp_benchmark_warmup):
                        _run_tp_concurrent_benchmark_once(
                            session,
                            benchmark_prompt,
                            args.max_new_tokens,
                            args.tp_benchmark_concurrency,
                            fast_decode=args.tp_fast_decode,
                        )
                    for _ in range(args.tp_benchmark_iterations):
                        runs.append(
                            _run_tp_concurrent_benchmark_once(
                                session,
                                benchmark_prompt,
                                args.max_new_tokens,
                                args.tp_benchmark_concurrency,
                                fast_decode=args.tp_fast_decode,
                            )
                        )
                    lines = _format_tp_concurrent_benchmark_results(
                        runs,
                        warmup_iterations=args.tp_benchmark_warmup,
                        prompt_target_tokens=args.tp_benchmark_prompt_tokens,
                        print_runs=args.tp_benchmark_print_runs,
                    )
                else:
                    results: list[GenerateResult] = []
                    for _ in range(args.tp_benchmark_warmup):
                        session.generate(benchmark_prompt, args.max_new_tokens, fast_decode=args.tp_fast_decode)
                    for _ in range(args.tp_benchmark_iterations):
                        results.append(
                            session.generate(
                                benchmark_prompt,
                                args.max_new_tokens,
                                fast_decode=args.tp_fast_decode,
                                cuda_graph_probe=args.tp_cuda_graph_probe,
                            )
                        )
                    lines = _format_tp_benchmark_results(
                        results,
                        warmup_iterations=args.tp_benchmark_warmup,
                        prompt_target_tokens=args.tp_benchmark_prompt_tokens,
                        print_runs=args.tp_benchmark_print_runs,
                    )
            for line in lines:
                print(line)
        except (ConfigError, EngineError, MappingError, LoaderError, TpRuntimeError, CliError) as exc:
            raise CliError(str(exc)) from exc
        except Exception as exc:
            raise CliError(f"TP benchmark failed: {exc}") from exc
        return 0
    if args.reference_prefill:
        try:
            runtime_config = _runtime_config_from_args(config, args)
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
            runtime_config = _runtime_config_from_args(config, args)
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
            runtime_config = _runtime_config_from_args(config, args)
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
        "--tp-profile",
        choices=("off", "cpu", "sync"),
        default="off",
        help="Enable TP profiling: off, CPU-side scope timing, or synchronized CUDA scope timing.",
    )
    parser.add_argument(
        "--tp-profile-layer-detail",
        action="store_true",
        help="Include per-layer profile scope names when --tp-profile is enabled.",
    )
    parser.add_argument(
        "--tp-native-paged-attention",
        choices=("config", "on", "off"),
        default="config",
        help="Override text_config.native_paged_attention for TP generate/benchmark/service runs.",
    )
    parser.add_argument(
        "--tp-fast-decode",
        action="store_true",
        help="Skip avoidable per-token CUDA synchronizations and diagnostics in TP generate/benchmark/service decode loops.",
    )
    parser.add_argument(
        "--tp-cuda-graph-probe",
        action="store_true",
        help="Report CUDA graph decode-readiness blockers without attempting capture or replay.",
    )
    parser.add_argument(
        "--tp-benchmark-iterations",
        type=int,
        default=1,
        help="Measured resident benchmark iterations for --tp-benchmark.",
    )
    parser.add_argument(
        "--tp-benchmark-warmup",
        type=int,
        default=0,
        help="Warmup generations to run before measured --tp-benchmark iterations.",
    )
    parser.add_argument(
        "--tp-benchmark-prompt-tokens",
        type=int,
        help="Approximate prompt token target for --tp-benchmark by repeating/truncating the prompt.",
    )
    parser.add_argument(
        "--tp-benchmark-concurrency",
        type=int,
        default=1,
        help="Concurrent generation requests to advance with batched decode during --tp-benchmark.",
    )
    parser.add_argument(
        "--tp-benchmark-print-runs",
        action="store_true",
        help="Print per-run benchmark timing lines in addition to aggregate metrics.",
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
    if args.tp_benchmark_iterations <= 0:
        parser.error("--tp-benchmark-iterations must be positive")
    if args.tp_benchmark_warmup < 0:
        parser.error("--tp-benchmark-warmup must be non-negative")
    if args.tp_benchmark_prompt_tokens is not None and args.tp_benchmark_prompt_tokens <= 0:
        parser.error("--tp-benchmark-prompt-tokens must be positive")
    if args.tp_benchmark_concurrency <= 0:
        parser.error("--tp-benchmark-concurrency must be positive")
    if args.inspect_tp is not None and args.inspect_tp <= 0:
        parser.error("--inspect-tp must be positive")
    try:
        return run(args)
    except CliError as exc:
        parser.exit(2, f"qwen36-run: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
