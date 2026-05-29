from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from reference_ops import ReferenceWeights, embedding, full_attention, linear_attention, rms_norm
from runtime_config import RuntimeConfig
from weight_mapping import LanguageModelMapping, LayerMapping


class TpRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class TpLaunchConfig:
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"
    init_method: str | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise TpRuntimeError(f"tp world size must be >= 1, got {self.world_size}")
        if not 0 <= self.rank < self.world_size:
            raise TpRuntimeError(f"tp rank {self.rank} out of range for world size {self.world_size}")
        if self.local_rank < 0:
            raise TpRuntimeError(f"tp local rank must be >= 0, got {self.local_rank}")
        if self.backend not in ("nccl", "gloo"):
            raise TpRuntimeError(f"unsupported tp backend: {self.backend}")

    @classmethod
    def from_env(
        cls,
        *,
        world_size: int | None = None,
        rank: int | None = None,
        local_rank: int | None = None,
        backend: str = "nccl",
        init_method: str | None = None,
        device: str | None = None,
    ) -> TpLaunchConfig:
        resolved_world = world_size if world_size is not None else int(os.environ.get("WORLD_SIZE", "1"))
        resolved_rank = rank if rank is not None else int(os.environ.get("RANK", "0"))
        resolved_local = local_rank if local_rank is not None else int(os.environ.get("LOCAL_RANK", str(resolved_rank)))
        return cls(
            world_size=resolved_world,
            rank=resolved_rank,
            local_rank=resolved_local,
            backend=backend,
            init_method=init_method,
            device=device,
        )

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    def torch_device(self) -> Any:
        import torch

        if self.device is not None:
            return torch.device(self.device)
        if self.backend == "nccl":
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")


class TpRuntime:
    def __init__(self, config: TpLaunchConfig) -> None:
        self.config = config
        self.device: Any | None = None
        self._owns_process_group = False

    def __enter__(self) -> TpRuntime:
        import torch
        import torch.distributed as dist

        self.device = self.config.torch_device()
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise TpRuntimeError("CUDA is required for NCCL TP runtime")
            torch.cuda.set_device(self.device)
        if self.config.is_distributed:
            if not dist.is_available():
                raise TpRuntimeError("torch.distributed is not available")
            if self.config.backend == "nccl" and not dist.is_nccl_available():
                raise TpRuntimeError("torch.distributed NCCL backend is not available")
            if self.config.backend == "gloo" and not dist.is_gloo_available():
                raise TpRuntimeError("torch.distributed gloo backend is not available")
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.config.backend,
                    init_method=self.config.init_method or "env://",
                    rank=self.config.rank,
                    world_size=self.config.world_size,
                )
                self._owns_process_group = True
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._owns_process_group:
            return
        import torch.distributed as dist

        dist.destroy_process_group()
        self._owns_process_group = False

    def all_reduce_sum(self, tensor: Any) -> Any:
        if not self.config.is_distributed:
            return tensor
        import torch.distributed as dist

        if self.config.backend == "nccl" and tensor.device.type != "cuda":
            raise TpRuntimeError("NCCL all-reduce requires a CUDA tensor")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def barrier(self) -> None:
        if not self.config.is_distributed:
            return
        import torch.distributed as dist

        dist.barrier()


def tp_decoder_layer(
    hidden_states: Any,
    mapping: LayerMapping,
    config: RuntimeConfig,
    weights: ReferenceWeights,
    runtime: TpRuntime,
) -> Any:
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights.tensor(mapping.input_layernorm.name), config.rms_norm_eps)
    if mapping.layer_type == "full_attention":
        hidden_states = residual + full_attention(hidden_states, mapping.attention, config, weights)
    elif mapping.layer_type == "linear_attention":
        hidden_states = residual + linear_attention(hidden_states, mapping.attention, config, weights)
    else:
        raise ValueError(f"unsupported TP layer type: {mapping.layer_type}")
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights.tensor(mapping.post_attention_layernorm.name), config.rms_norm_eps)
    moe_output = weights.moe(hidden_states, mapping.mlp, config)
    runtime.all_reduce_sum(moe_output)
    return residual + moe_output


def tp_language_model(
    input_ids: Any,
    mapping: LanguageModelMapping,
    config: RuntimeConfig,
    weights: ReferenceWeights,
    runtime: TpRuntime,
) -> Any:
    from weight_mapping import LinearTensor

    hidden_states = embedding(input_ids, weights.tensor(mapping.embed_tokens.name))
    for layer in mapping.layers:
        hidden_states = tp_decoder_layer(hidden_states, layer, config, weights, runtime)
    hidden_states = rms_norm(hidden_states, weights.tensor(mapping.final_norm.name), config.rms_norm_eps)
    return weights.linear(hidden_states, LinearTensor(weight=mapping.lm_head, scale=None))


def mapped_tensor_bytes(mapping: LanguageModelMapping) -> int:
    return sum(tensor.nbytes for tensor in _mapped_tensor_infos(mapping))


def _mapped_tensor_infos(mapping: LanguageModelMapping):
    tensors = [mapping.embed_tokens, mapping.final_norm, mapping.lm_head]
    for layer in mapping.layers:
        tensors.extend((layer.input_layernorm, layer.post_attention_layernorm))
        attention = layer.attention
        if layer.layer_type == "full_attention":
            tensors.extend(
                (
                    attention.q_proj.weight,
                    attention.k_proj.weight,
                    attention.v_proj.weight,
                    attention.o_proj.weight,
                    attention.q_norm,
                    attention.k_norm,
                )
            )
            tensors.extend(t for t in (attention.q_proj.scale, attention.k_proj.scale, attention.v_proj.scale, attention.o_proj.scale) if t is not None)
        else:
            tensors.extend(
                (
                    attention.in_proj_qkv.weight,
                    attention.in_proj_z.weight,
                    attention.out_proj.weight,
                    attention.in_proj_a.weight,
                    attention.in_proj_b.weight,
                    attention.conv1d_weight,
                    attention.a_log,
                    attention.dt_bias,
                    attention.norm,
                )
            )
            tensors.extend(t for t in (attention.in_proj_qkv.scale, attention.in_proj_z.scale, attention.out_proj.scale) if t is not None)
        moe = layer.mlp
        tensors.extend((moe.gate, moe.shared_expert_gate))
        for expert in (*moe.experts, moe.shared_expert):
            for linear in (expert.gate_proj, expert.up_proj, expert.down_proj):
                tensors.append(linear.weight)
                if linear.scale is not None:
                    tensors.append(linear.scale)
    return tensors
