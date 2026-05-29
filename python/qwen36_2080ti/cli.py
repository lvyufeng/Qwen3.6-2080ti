from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class CliError(RuntimeError):
    pass


def _load_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise CliError(f"missing config.json under {model_dir}")
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as exc:
        raise CliError(f"invalid JSON in {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise CliError(f"expected object in {config_path}")
    return config


def _summarize_config(config: dict[str, Any]) -> list[str]:
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


def run(args: argparse.Namespace) -> int:
    model_dir = args.model.expanduser().resolve()
    if not model_dir.is_dir():
        raise CliError(f"model path is not a directory: {model_dir}")

    config = _load_config(model_dir)
    print("Loaded model config")
    print(f"model_dir: {model_dir}")
    for line in _summarize_config(config):
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
