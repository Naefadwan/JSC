"""Local Ollama client. Report text is sent only to 127.0.0.1."""

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from jsc_review.review.findings import Requirement


class LocalModelError(RuntimeError):
    """Raised when the local model does not return a usable judgment."""


@dataclass(frozen=True)
class ModelSettings:
    """Versioned local-model settings."""

    model: str
    temperature: float
    host: str
    num_ctx: int


@dataclass(frozen=True)
class ModelJudgment:
    """One model judgment before evidence verification."""

    status: str
    reason: str
    quote: str
    confidence: float


def load_model_settings() -> ModelSettings:
    """Read the versioned model name and temperature."""

    path = Path(__file__).with_name("model.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ModelSettings(
        model=str(payload["model"]),
        temperature=float(payload["temperature"]),
        host=str(payload["host"]).rstrip("/"),
        num_ctx=int(payload["num_ctx"]),
    )


def judge_passage(
    requirement: Requirement,
    passage: str,
    *,
    passage_is_ocr: bool,
    settings: ModelSettings | None = None,
    timeout_seconds: float = 180,
) -> ModelJudgment:
    """Ask the local model to judge one requirement against one passage."""

    chosen = settings or load_model_settings()
    if not chosen.host.startswith("http://127.0.0.1"):
        raise LocalModelError("The model host must stay on 127.0.0.1")
    prompt_path = Path(__file__).with_name("prompt_requirement.txt")
    instructions = prompt_path.read_text(encoding="utf-8").strip()
    source = "OCR text" if passage_is_ocr else "embedded report text"
    user = (
        f"Requirement {requirement.requirement_id}: {requirement.text_ar}\n"
        f"English: {requirement.text_en}\n"
        f"Passage source: {source}\n"
        f"Passage:\n{passage}"
    )
    body = {
        "model": chosen.model,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "format": _judgment_schema(),
        "options": {"temperature": chosen.temperature, "num_ctx": chosen.num_ctx},
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user},
        ],
    }
    request = Request(
        f"{chosen.host}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LocalModelError("The local model request failed") from exc
    content = payload.get("message", {}).get("content", "")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LocalModelError("The local model did not return JSON") from exc
    return _parse_judgment(parsed)


def _parse_judgment(parsed: object) -> ModelJudgment:
    if not isinstance(parsed, dict):
        raise LocalModelError("The local model judgment was not an object")
    status = str(parsed.get("status", ""))
    if status not in {"Met", "Not Met", "Needs Review"}:
        raise LocalModelError("The local model returned an unknown status")
    reason = str(parsed.get("reason", "")).strip()
    if not reason:
        raise LocalModelError("The local model omitted the reason")
    confidence = parsed.get("confidence", 0.0)
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        raise LocalModelError("The local model confidence was not a number")
    return ModelJudgment(
        status=status,
        reason=reason,
        quote=str(parsed.get("quote", "")).strip(),
        confidence=min(1.0, max(0.0, float(confidence))),
    )


def _judgment_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["status", "reason", "quote", "confidence"],
        "properties": {
            "status": {"type": "string", "enum": ["Met", "Not Met", "Needs Review"]},
            "reason": {"type": "string"},
            "quote": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }
