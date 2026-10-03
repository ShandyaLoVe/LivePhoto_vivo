"""Apply aligned spatial transforms and Unicode-safe PNG input/output."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import cv2
import numpy as np

from .config import output_size


INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


@dataclass(frozen=True)
class SpatialPlan:
    source_height: int
    source_width: int
    resized_height: int
    resized_width: int
    crop_top: int
    crop_left: int
    output_height: int
    output_width: int
    interpolation: str
    keep_aspect_ratio: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_spatial_plan(frame: np.ndarray, config: Mapping[str, Any], rng: np.random.Generator) -> SpatialPlan:
    source_height, source_width = frame.shape[:2]
    if bool(config.get("preserve_source_resolution", False)):
        target_height, target_width = source_height, source_width
    else:
        target_height, target_width = output_size(config)
    keep_aspect = bool(config.get("keep_aspect_ratio", True))
    interpolation = str(config.get("interpolation", "lanczos")).lower()
    if interpolation not in INTERPOLATIONS:
        raise ValueError("Unsupported image interpolation: %s" % interpolation)

    if keep_aspect:
        scale = max(target_width / source_width, target_height / source_height)
        resized_width = max(target_width, int(round(source_width * scale)))
        resized_height = max(target_height, int(round(source_height * scale)))
    else:
        resized_width, resized_height = target_width, target_height

    max_left = resized_width - target_width
    max_top = resized_height - target_height
    if bool(config.get("random_crop", False)):
        crop_left = int(rng.integers(0, max_left + 1)) if max_left else 0
        crop_top = int(rng.integers(0, max_top + 1)) if max_top else 0
    else:
        crop_left, crop_top = max_left // 2, max_top // 2

    return SpatialPlan(
        source_height=source_height,
        source_width=source_width,
        resized_height=resized_height,
        resized_width=resized_width,
        crop_top=crop_top,
        crop_left=crop_left,
        output_height=target_height,
        output_width=target_width,
        interpolation=interpolation,
        keep_aspect_ratio=keep_aspect,
    )


def apply_spatial_plan(frame: np.ndarray, plan: SpatialPlan) -> np.ndarray:
    if frame.shape[:2] != (plan.source_height, plan.source_width):
        raise ValueError("Resolution changed inside one source video")
    if (
        plan.resized_height == plan.source_height
        and plan.resized_width == plan.source_width
        and plan.crop_top == 0
        and plan.crop_left == 0
        and plan.output_height == plan.source_height
        and plan.output_width == plan.source_width
    ):
        return np.ascontiguousarray(frame)
    resized = cv2.resize(
        frame,
        (plan.resized_width, plan.resized_height),
        interpolation=INTERPOLATIONS[plan.interpolation],
    )
    top, left = plan.crop_top, plan.crop_left
    cropped = resized[top:top + plan.output_height, left:left + plan.output_width]
    if cropped.shape[:2] != (plan.output_height, plan.output_width):
        raise RuntimeError("Spatial transform produced the wrong dimensions")
    return np.ascontiguousarray(cropped)


def transform_clip(frames: Sequence[np.ndarray], config: Mapping[str, Any], rng: np.random.Generator) -> Tuple[list, SpatialPlan]:
    if not frames:
        raise ValueError("Cannot transform an empty clip")
    plan = make_spatial_plan(frames[0], config, rng)
    return [apply_spatial_plan(frame, plan) for frame in frames], plan


def encode_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("Could not PNG-encode %s" % path)
    path.write_bytes(encoded.tobytes())


def decode_image(path: Path) -> np.ndarray:
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Corrupted or unsupported image: %s" % path)
    return image
