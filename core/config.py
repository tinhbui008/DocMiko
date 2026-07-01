"""
Central config loader. Reads .env from project root automatically.
All core modules import get_api_key() from here instead of os.environ directly.
"""

import os
from pathlib import Path

_env_loaded = False


def _load_env() -> None:
    global _env_loaded
    if _env_loaded:
        return
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
    _env_loaded = True


def get_api_key() -> str:
    _load_env()
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LLM_API_KEY")
    if not key:
        raise RuntimeError(
            "No API key found. Set ANTHROPIC_API_KEY or LLM_API_KEY in your .env file."
        )
    return key


def get_model(default: str = "claude-sonnet-4-6") -> str:
    _load_env()
    return os.environ.get("LLM_MODEL", default)
