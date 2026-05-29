from __future__ import annotations

import argparse
from pathlib import Path

from qwen36_2080ti.checkpoint import CheckpointError, Manifest, build_manifest


class CliError(RuntimeError):
    pass



def _summarize_config(config: dict[str, object]) -> list[str]:
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
