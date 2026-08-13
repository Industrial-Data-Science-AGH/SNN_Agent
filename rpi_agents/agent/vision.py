"""Vision classification: Gemini API + failsafe (P3).

Top-level imports remain hardware-free; google.genai and cv2 are imported
lazily inside _call_gemini().
"""

import json
import logging
from collections.abc import Callable

import numpy as np

from agent import config
from agent.types import VisionVerdict

logger = logging.getLogger(__name__)

_PROMPT = (
    "You are a home-security analyst reviewing a single frame captured after "
    "a glass-break sensor triggered. Make two independent judgments:\n"
    "1. window_broken: does this frame show visible evidence the window or "
    "glass itself is broken (a shattered pane, a hole, cracks, glass shards "
    "on the sill or floor)? Judge the window's physical state only -- not "
    "whether anyone is present.\n"
    "2. is_intrusion: does this frame show a real break-in or intruder, as "
    "opposed to a false alarm (pet, headlights, curtain, empty scene, or a "
    "window that looks broken but with no other sign of intrusion)?\n"
    'Return JSON {"window_broken": bool, "is_intrusion": bool, "confidence": 0..1, '
    '"reason": short string}. When uncertain on either judgment, prefer the '
    "value that leads to an alert (window_broken=true or is_intrusion=true)."
)

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "window_broken": {"type": "boolean"},
        "is_intrusion": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["window_broken", "is_intrusion", "confidence", "reason"],
}


def _encode_jpeg(snapshot: np.ndarray) -> bytes:
    """Encode an RGB snapshot to JPEG bytes (flips to BGR for cv2)."""
    import cv2  # type: ignore[import-untyped]

    ok, buf = cv2.imencode(".jpg", snapshot[..., ::-1])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _call_gemini(snapshot: np.ndarray) -> str:
    """Send snapshot to Gemini and return raw JSON response text.

    Never imported at module top — tests monkeypatch this symbol directly.
    """
    from google import genai  # type: ignore[import-untyped]
    from google.genai import types  # type: ignore[import-untyped]

    settings = config.load_settings()
    client = genai.Client(
        api_key=settings.gemini_api_key,
        http_options=types.HttpOptions(timeout=int(config.GEMINI_TIMEOUT_S * 1000)),
    )
    jpeg = _encode_jpeg(snapshot)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=[
            _PROMPT,
            types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_VERDICT_SCHEMA,
            max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
        ),
    )
    return response.text


def _parse_verdict(raw: str) -> VisionVerdict:
    """Parse a raw Gemini JSON string into a VisionVerdict.

    Strips markdown code fences if present (Gemini sometimes wraps JSON).
    Raises ValueError or KeyError on missing/mistyped fields.
    """
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        end = len(lines) - 1 if lines[-1].startswith("```") else len(lines)
        stripped = "\n".join(lines[1:end])

    data = json.loads(stripped)

    if "window_broken" not in data:
        raise ValueError("Missing field: window_broken")
    if "is_intrusion" not in data:
        raise ValueError("Missing field: is_intrusion")
    if "confidence" not in data:
        raise ValueError("Missing field: confidence")
    if "reason" not in data:
        raise ValueError("Missing field: reason")

    confidence = max(0.0, min(1.0, float(data["confidence"])))

    return VisionVerdict(
        is_intrusion=bool(data["is_intrusion"]),
        confidence=confidence,
        reason=str(data["reason"])[:200],
        source="gemini",
        window_broken=bool(data["window_broken"]),
    )


def verdict(
    snapshot: np.ndarray,
    *,
    generate: Callable[[np.ndarray], str] | None = None,
) -> VisionVerdict:
    """Classify snapshot as intrusion or false alarm via Gemini.

    Args:
        snapshot: RGB image array, shape (height, width, 3) uint8.
        generate: Optional seam replacing _call_gemini for replay/testing.
            P4 callers use the single-arg form (no generate).

    Returns:
        VisionVerdict. On any failure, returns a failsafe ALARM verdict
        (is_intrusion=True, source="failsafe") — never lets an exception escape.
    """
    gen = generate or _call_gemini
    try:
        return _parse_verdict(gen(snapshot))
    except Exception as exc:
        logger.warning("Vision failed (%s); failing open to ALARM.", exc)
        return VisionVerdict(
            is_intrusion=True,
            confidence=1.0,
            reason=f"failsafe: {type(exc).__name__}",
            source="failsafe",
            window_broken=True,
        )
