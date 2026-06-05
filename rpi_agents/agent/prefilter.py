"""Local motion + person prefilter: gates vision API calls (P2).

Top-level imports remain hardware-free; cv2 is imported lazily inside run().
Gate is intentionally permissive: escalate = motion OR person.
A missing/disabled person model degrades gracefully to motion-only — recall
must not be traded for efficiency.
"""

import logging
from typing import Any

import numpy as np

from agent import config
from agent.types import PrefilterResult

logger = logging.getLogger(__name__)

# Module-level net cache: loaded once on first run() call, not on every call.
_net: Any = None
_net_loaded: bool = False  # True once a load attempt has been made

# Latest bounding-box/frame info from the last escalating detection.
# Written when escalate=True; read by the P3 vision stage.
last_detection: dict[str, Any] = {}


def _load_net() -> Any:
    """Return the cached MobileNet-SSD net, loading it on first call.

    Returns None if person detection is disabled, model files are absent,
    or the load raises — the caller falls back to motion-only gate.
    """
    global _net, _net_loaded
    if _net_loaded:
        return _net
    _net_loaded = True

    if not config.PREFILTER_PERSON_ENABLED:
        logger.info("Person detection disabled (PREFILTER_PERSON_ENABLED=False).")
        return None

    import cv2  # type: ignore[import-untyped]

    prototxt = config.PERSON_MODEL_DIR / "MobileNetSSD_deploy.prototxt"
    caffemodel = config.PERSON_MODEL_DIR / "MobileNetSSD_deploy.caffemodel"
    if not prototxt.exists() or not caffemodel.exists():
        logger.warning(
            "Person model files not found at %s; falling back to motion-only gate.",
            config.PERSON_MODEL_DIR,
        )
        return None

    try:
        _net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
        return _net
    except Exception as exc:
        logger.warning(
            "Failed to load person model: %s; falling back to motion-only gate.", exc
        )
        return None


def run(frames: np.ndarray) -> PrefilterResult:
    """Run motion + person detection on a frame sequence.

    Uses frame-diff motion detection (normalized mean absolute diff between
    consecutive grayscale frames) and optional MobileNet-SSD (cv2.dnn) person
    detection on the last frame.  Gate: escalate = motion OR person.

    Args:
        frames: Array of shape (n_frames, H, W, 3) in BGR uint8.

    Returns:
        PrefilterResult with motion, person, escalate, and confidence score.
    """
    import cv2  # type: ignore[import-untyped]

    n = len(frames)

    # Motion detection: max normalised mean-absolute-diff across pairs

    motion = False
    max_diff = 0.0

    if n >= 2:
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        diffs = [
            np.mean(
                np.abs(grays[i].astype(np.float32) - grays[i - 1].astype(np.float32))
            ) / 255.0
            for i in range(1, n)
        ]
        max_diff = float(max(diffs))
        motion = max_diff >= config.PREFILTER_MOTION_THRESHOLD

    score = max_diff

    # Person detection: MobileNet-SSD on last frame (graceful degrade)
    
    person = False
    net = _load_net()

    if net is not None and n > 0:
        try:
            last_frame = frames[-1]
            h, w = last_frame.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(last_frame, (300, 300)),
                0.007843,
                (300, 300),
                127.5,
            )
            net.setInput(blob)
            detections = net.forward()
            # MobileNet-SSD class index 15 = person
            for i in range(detections.shape[2]):
                class_id = int(detections[0, 0, i, 1])
                confidence = float(detections[0, 0, i, 2])
                if class_id == 15 and confidence >= config.PREFILTER_PERSON_CONF:
                    person = True
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    last_detection["bbox"] = box.astype(int).tolist()
                    last_detection["frame_idx"] = n - 1
                    last_detection["confidence"] = confidence
                    break
        except Exception as exc:
            logger.warning("Person detection failed: %s; continuing without.", exc)

    # Permissive gate
    
    escalate = motion or person

    if escalate and not person:
        # Motion-only escalation: record frame index; no bbox available.
        last_detection["frame_idx"] = n - 1
        last_detection["bbox"] = None

    return PrefilterResult(motion=motion, person=person, escalate=escalate, score=score)
