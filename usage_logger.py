
from __future__ import annotations

import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from threading import Lock

_LOCK = Lock()
_BASE = Path(__file__).resolve().parent
_CSV = _BASE / "results" / "token_usage.csv"
_FIELDS = [
    "timestamp", "method", "manual", "pass",
    "input_tokens", "output_tokens", "total_tokens",
    "model",
]


def _detect_method() -> str:
    return Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else "unknown"


def _detect_manual() -> str:
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--manual" and i + 1 < len(argv):
            return Path(argv[i + 1]).name
    return "unknown"


def _extract_usage(response) -> dict:
    """Pull input/output/total tokens from a LangChain AIMessage."""
    # LangChain >= 0.2 normalises usage into response.usage_metadata
    usage = getattr(response, "usage_metadata", None)
    if usage:
        return {
            "input_tokens":  int(usage.get("input_tokens",  0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "total_tokens":  int(usage.get("total_tokens",  0) or 0),
        }
    # Fallback: response_metadata.token_usage (older LangChain / direct OpenAI shape)
    meta = getattr(response, "response_metadata", {}) or {}
    tu = meta.get("token_usage") or meta.get("usage") or {}
    return {
        "input_tokens":  int(tu.get("prompt_tokens", tu.get("input_tokens",  0)) or 0),
        "output_tokens": int(tu.get("completion_tokens", tu.get("output_tokens", 0)) or 0),
        "total_tokens":  int(tu.get("total_tokens", 0) or 0),
    }


def _extract_model(response) -> str:
    meta = getattr(response, "response_metadata", {}) or {}
    return str(meta.get("model_name") or meta.get("model") or "")


def log_usage(response, pass_name: str = "main") -> None:
    """Append one row to results/token_usage.csv from a LangChain AIMessage."""
    try:
        usage = _extract_usage(response)
        row = {
            "timestamp":     datetime.now().isoformat(timespec="seconds"),
            "method":        _detect_method(),
            "manual":        _detect_manual(),
            "pass":          pass_name,
            "input_tokens":  usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens":  usage["total_tokens"] or (usage["input_tokens"] + usage["output_tokens"]),
            "model":         _extract_model(response),
        }
        with _LOCK:
            _CSV.parent.mkdir(parents=True, exist_ok=True)
            new_file = not _CSV.exists()
            with open(_CSV, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=_FIELDS)
                if new_file:
                    w.writeheader()
                w.writerow(row)
    except Exception as e:
        # Never break the extraction pipeline because of logging
        print(f"[usage_logger] WARNING: failed to log token usage: {e}", file=sys.stderr)
