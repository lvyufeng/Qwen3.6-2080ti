from __future__ import annotations

import json
from pathlib import Path

from qwen36_2080ti.cli import main


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

    rc = main(["--model", str(tmp_path), "--prompt", "hello", "--max-new-tokens", "1"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "Loaded model config" in out
    assert "model_type: qwen3_moe" in out
    assert "num_experts: 128" in out
    assert "inference: not implemented yet" in out
