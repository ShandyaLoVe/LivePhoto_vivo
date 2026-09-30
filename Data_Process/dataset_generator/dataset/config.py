from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "input_dir": "./videos",
    "output_dir": "./dataset_output",
    "seed": 42,
    "num_workers": 1,
    "overwrite": False,
    "ffmpeg_bin": "ffmpeg",
    "ffprobe_bin": "ffprobe",
    "video": {
        "extensions": ["mp4", "mov", "mkv", "avi", "webm", "m4v", "mpeg", "mpg", "ts"],
        "scan_all_files": True,
        "allow_partial_decode": True,
    },
    "clip": {
        "num_frames": 7,
        "frame_interval": 1,
        "clip_stride": 7,
        "drop_last": True,
    },
    "image": {
        "width": 512,
        "height": 512,
        "gt_width": None,
        "gt_height": None,
        "crop_size": None,
        "keep_aspect_ratio": True,
        "random_crop": True,
        "center_crop": False,
        "interpolation": "lanczos",
    },
    "reference": {"strategy": "center"},
    "degradation": {
        "mode": "clip_consistent",
        "temporal_variation": 0.03,
        "blur": {"enabled": True, "sigma": [0.2, 3.0]},
        "motion_blur": {"enabled": False, "kernel_size": [3, 15], "angle": [-180.0, 180.0]},
        "downsample": {"enabled": True, "scale": [1.0, 4.0], "down_interpolation": "area", "up_interpolation": "cubic"},
        "gaussian_noise": {"enabled": True, "sigma": [0.0, 20.0]},
        "poisson_noise": {"enabled": False, "peak": [30.0, 60.0]},
        "color_distortion": {"enabled": False, "saturation": [0.85, 1.15], "hue_shift": [-5.0, 5.0]},
        "brightness_contrast": {"enabled": False, "brightness": [-10.0, 10.0], "contrast": [0.9, 1.1]},
        "gamma": {"enabled": False, "gamma": [0.85, 1.2]},
        "sharpen": {"enabled": False, "amount": [0.0, 1.5], "radius": [0.5, 1.5]},
        "jpeg": {"enabled": True, "quality": [30, 95]},
        "video_compression": {"enabled": False, "codec": "libx264", "crf": [18, 38], "preset": "medium", "keyint": 12},
    },
    "split": {"train": 0.8, "val": 0.1, "test": 0.1},
}


def _deep_merge(base: Dict[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Optional[Path]) -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("The YAML root must be a mapping")
        _deep_merge(config, loaded)
    validate_config(config)
    return config


def output_size(image_config: Mapping[str, Any]) -> Tuple[int, int]:
    """Return output (height, width). crop_size accepts int or [height, width]."""
    crop_size = image_config.get("crop_size")
    if crop_size is not None:
        if isinstance(crop_size, int):
            return crop_size, crop_size
        if isinstance(crop_size, (list, tuple)) and len(crop_size) == 2:
            return int(crop_size[0]), int(crop_size[1])
        raise ValueError("image.crop_size must be an integer or [height, width]")

    width = image_config.get("gt_width") or image_config.get("width")
    height = image_config.get("gt_height") or image_config.get("height")
    if width is None or height is None:
        raise ValueError("image.width/image.height (or gt_width/gt_height) are required")
    return int(height), int(width)


def validate_config(config: Mapping[str, Any]) -> None:
    clip = config["clip"]
    for key in ("num_frames", "frame_interval", "clip_stride"):
        if int(clip[key]) <= 0:
            raise ValueError("clip.%s must be > 0" % key)

    height, width = output_size(config["image"])
    if height <= 0 or width <= 0:
        raise ValueError("Output image dimensions must be > 0")
    if config["image"].get("random_crop") and config["image"].get("center_crop"):
        raise ValueError("image.random_crop and image.center_crop cannot both be true")

    if config["reference"]["strategy"] not in {"center", "first", "last", "random", "sharpest"}:
        raise ValueError("Unsupported reference.strategy")
    if config["degradation"]["mode"] not in {"frame_independent", "clip_consistent"}:
        raise ValueError("degradation.mode must be frame_independent or clip_consistent")

    split = config["split"]
    expected = {"train", "val", "test"}
    if set(split) != expected:
        raise ValueError("split must contain exactly train, val and test")
    if any(float(split[name]) < 0 for name in expected):
        raise ValueError("Split ratios cannot be negative")
    total = sum(float(split[name]) for name in expected)
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    if int(config.get("num_workers", 1)) <= 0:
        raise ValueError("num_workers must be > 0")

