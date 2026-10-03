"""Select one transformed GT frame as a clip's reference image."""

from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np


def sharpness_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_reference(
    frames: Sequence[np.ndarray],
    frame_indices: Sequence[int],
    strategy: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int, int]:
    if len(frames) != len(frame_indices) or not frames:
        raise ValueError("Reference selection requires aligned, non-empty frames and indices")
    if strategy == "center":
        position = len(frames) // 2
    elif strategy == "first":
        position = 0
    elif strategy == "last":
        position = len(frames) - 1
    elif strategy == "random":
        position = int(rng.integers(0, len(frames)))
    elif strategy == "sharpest":
        position = max(range(len(frames)), key=lambda index: sharpness_score(frames[index]))
    else:
        raise ValueError("Unsupported reference strategy: %s" % strategy)
    return frames[position].copy(), int(frame_indices[position]), position
