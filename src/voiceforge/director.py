"""A bounded local-LLM adviser for interpreting production language."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class VibeProfile:
    name: str
    strength: float
    retune_ms: float
    reason: str


PROFILES = {
    "intimate": VibeProfile("intimate", 0.50, 120.0, "keeps slides and slow vibrato"),
    "warm": VibeProfile("warm", 0.62, 95.0, "settles sustained notes without locking them"),
    "polished": VibeProfile("polished", 0.75, 65.0, "noticeably clean but still human"),
    "pop": VibeProfile("pop", 0.90, 28.0, "tight contemporary note-centering"),
    "robotic": VibeProfile("robotic", 1.0, 8.0, "deliberately obvious hard retune"),
}


def _load_local_env() -> None:
    """Load a tiny local .env file without adding a runtime dependency."""
    path = Path(".env")
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key and not key.lstrip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"'))


def profile_from_text(vibe: str) -> VibeProfile:
    words = vibe.lower()
    if any(word in words for word in ("robot", "t-pain", "hard tune", "hyperpop")):
        return PROFILES["robotic"]
    if any(word in words for word in ("pop", "glossy", "radio", "tight")):
        return PROFILES["pop"]
    if any(word in words for word in ("warm", "soul", "rich")):
        return PROFILES["warm"]
    if any(word in words for word in ("clean", "polished", "modern")):
        return PROFILES["polished"]
    return PROFILES["intimate"]


def local_llm_profile(
    vibe: str,
    vocal_facts: dict[str, object],
    endpoint: str | None = None,
    model: str | None = None,
) -> VibeProfile | None:
    """Ask an OpenAI-compatible *local* server to select one safe preset name.

    No audio is sent. The LLM receives only the user's words and aggregate pitch facts,
    and can choose a name from a closed set. A failed/offline server simply returns None.
    """
    _load_local_env()
    endpoint = endpoint or os.getenv("VOICEFORGE_LLM_URL")
    model = model or os.getenv("VOICEFORGE_LLM_MODEL")
    if not endpoint or not model:
        return None
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return None
    prompt = (
        "You are a conservative vocal producer. Select exactly one profile name from "
        "[intimate, warm, polished, pop, robotic]. Do not explain.\n"
        f"Vibe: {vibe}\nVocal facts: {json.dumps(vocal_facts, sort_keys=True)}"
    )
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 12,
        }
    ).encode()
    request = Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=8) as response:
            answer = json.loads(response.read())
        selected = str(answer["choices"][0]["message"]["content"]).strip().lower().split()[0]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError, URLError, TimeoutError):
        return None
    return PROFILES.get(selected.rstrip(".,!"))
