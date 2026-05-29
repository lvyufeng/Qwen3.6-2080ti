from __future__ import annotations

import argparse
from pathlib import Path

from checkpoint import CheckpointError, Manifest, build_manifest
from fp8_smoke import Fp8SmokeReport, inspect_fp8_checkpoint
from loader import LoaderError, TensorLoader
from weight_mapping import LanguageModelMapping, MappingError, build_language_model_mapping


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


def run(args: argparse.Namespace) -> int:
    model_dir = args.model.expanduser().resolve()
    if not model_dir.is_dir():
        raise CliError(f"model path is not a directory: {model_dir}")

    try:
        manifest = build_manifest(model_dir)
    except CheckpointError as exc:
        raise CliError(str(exc)) from exc
    config = manifest.config
    print("Loaded model config")
    print(f"model_dir: {model_dir}")
    for line in _summarize_config(config):
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
    if args.inspect_tensor:
        try:
            print("Loaded tensor payload")
            for line in _summarize_tensor_load(manifest, args.inspect_tensor, args.tensor_device):
                print(line)
        except LoaderError as exc:
            raise CliError(str(exc)) from exc
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
        "--inspect-tensor",
        help="Read one tensor payload by name and print its location and decoded shape.",
    )
    parser.add_argument(
        "--tensor-device",
        help="Optionally copy --inspect-tensor to this torch device, e.g. cpu or cuda:0.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    try:
        return run(args)
    except CliError as exc:
        parser.exit(2, f"qwen36-run: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
