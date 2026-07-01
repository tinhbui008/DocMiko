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


def get_provider() -> str:
    """
    Which LLM backend to use for translation. Default is the local **ollama**
    (OpenAI-compatible, free). Set LLM_PROVIDER=anthropic in .env only if you
    deliberately want the paid Claude backend.
    """
    _load_env()
    return os.environ.get("LLM_PROVIDER", "ollama").strip().lower()


def get_model(default: str | None = None) -> str:
    _load_env()
    if default is None:
        default = "qwen2.5:14b" if get_provider() in ("ollama", "openai") else "claude-sonnet-4-6"
    return os.environ.get("LLM_MODEL", default)


def get_base_url() -> str | None:
    """Optional custom endpoint (proxy/gateway). None => SDK default."""
    _load_env()
    return os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("LLM_BASE_URL") or None


def make_client():
    """Build an Anthropic client honoring key + optional base_url."""
    import anthropic

    kwargs = {"api_key": get_api_key()}
    base = get_base_url()
    if base:
        kwargs["base_url"] = base
    return anthropic.Anthropic(**kwargs)


def make_openai_client():
    """
    Build an OpenAI-compatible client, used for Ollama (and any OpenAI-style
    gateway). Ollama serves this at http://localhost:11434/v1 and ignores the
    api key. Configure via LLM_BASE_URL / LLM_API_KEY in .env.
    """
    from openai import OpenAI

    _load_env()
    base = os.environ.get("LLM_BASE_URL") or "http://localhost:11434/v1"
    key = os.environ.get("LLM_API_KEY") or "ollama"
    return OpenAI(base_url=base, api_key=key)
