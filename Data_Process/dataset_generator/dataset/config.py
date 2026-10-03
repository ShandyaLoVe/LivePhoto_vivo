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
        "extensions": ["mp4", "mov", "mkv", "avi", "webm", "m4v", "mpeg", "mpg", "ts", "yuv"],
        "scan_all_files": True,
        "allow_partial_decode": True,
        "exclude_dir_names": [],
        "raw_yuv": {
            "enabled": True,
            "width": None,
            "height": None,
            "pix_fmt": "yuv420p",
            "fps": 30.0,
            "rotation": 0,
            "use_sidecar": True,
            "sidecar_suffix": ".json",
            "files": {},
        },
    },
    "clip": {
        "num_frames": 7,
        "frame_interval": 1,
        "clip_stride": 7,
        "drop_last": True,
        "sampling_fps": None,
        "clip_duration_seconds": None,
        "clip_stride_seconds": None,
    },
    "image": {
        "width": 512,
        "height": 512,
        "gt_width": None,
        "gt_height": None,
        "preserve_source_resolution": False,
        "lq_width": None,
        "lq_height": None,
        "lq_interpolation": "area",
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
        "device_style": {"enabled": False, "profile": "random", "profiles": {}},
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


def lq_output_size(image_config: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    """Return configured LQ (height, width), or None to match GT dimensions."""
    width = image_config.get("lq_width")
    height = image_config.get("lq_height")
    if width is None and height is None:
        return None
    if width is None or height is None:
        raise ValueError("image.lq_width and image.lq_height must both be set or both be null")
    return int(height), int(width)


def validate_config(config: Mapping[str, Any]) -> None:
    clip = config["clip"]
    for key in ("num_frames", "frame_interval", "clip_stride"):
        if int(clip[key]) <= 0:
            raise ValueError("clip.%s must be > 0" % key)
    sampling_fps = clip.get("sampling_fps")
    duration = clip.get("clip_duration_seconds")
    if (sampling_fps is None) != (duration is None):
        raise ValueError("clip.sampling_fps and clip.clip_duration_seconds must both be set or both be null")
    if sampling_fps is not None:
        sampling_fps, duration = float(sampling_fps), float(duration)
        stride_seconds = float(clip.get("clip_stride_seconds") or duration)
        if sampling_fps <= 0 or duration <= 0 or stride_seconds <= 0:
            raise ValueError("Time-based clip sampling values must be > 0")
        if abs(int(clip["num_frames"]) - sampling_fps * duration) > 1e-6:
            raise ValueError("clip.num_frames must equal sampling_fps * clip_duration_seconds")

    image = config["image"]
    if not bool(image.get("preserve_source_resolution", False)):
        height, width = output_size(image)
        if height <= 0 or width <= 0:
            raise ValueError("GT output dimensions must be > 0")
    lq_size = lq_output_size(image)
    if lq_size is not None and (lq_size[0] <= 0 or lq_size[1] <= 0):
        raise ValueError("LQ output dimensions must be > 0")
    if str(image.get("lq_interpolation", "area")).lower() not in {"nearest", "linear", "cubic", "area", "lanczos"}:
        raise ValueError("Unsupported image.lq_interpolation")
    if image.get("random_crop") and image.get("center_crop"):
        raise ValueError("image.random_crop and image.center_crop cannot both be true")

    if config["reference"]["strategy"] not in {"center", "first", "last", "random", "sharpest"}:
        raise ValueError("Unsupported reference.strategy")
    if config["degradation"]["mode"] not in {"frame_independent", "clip_consistent"}:
        raise ValueError("degradation.mode must be frame_independent or clip_consistent")
    device = config["degradation"].get("device_style", {})
    if device.get("enabled"):
        profiles = device.get("profiles", {})
        if not isinstance(profiles, Mapping) or not profiles:
            raise ValueError("degradation.device_style.profiles must be a non-empty mapping")
        requested = str(device.get("profile", "random"))
        if requested != "random" and requested not in profiles:
            raise ValueError("degradation.device_style.profile is not present in profiles")

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

    raw_yuv = config.get("video", {}).get("raw_yuv", {})
    if raw_yuv.get("enabled", True):
        width, height = raw_yuv.get("width"), raw_yuv.get("height")
        if (width is None) != (height is None):
            raise ValueError("video.raw_yuv.width and height must both be set or both be null")
        if width is not None and (int(width) <= 0 or int(height) <= 0):
            raise ValueError("video.raw_yuv width and height must be > 0")
        if float(raw_yuv.get("fps", 30.0)) <= 0:
            raise ValueError("video.raw_yuv.fps must be > 0")
        if not str(raw_yuv.get("pix_fmt", "")).strip():
            raise ValueError("video.raw_yuv.pix_fmt cannot be empty")
        if int(raw_yuv.get("rotation", 0)) % 360 not in {0, 90, 180, 270}:
            raise ValueError("video.raw_yuv.rotation must be 0, 90, 180 or 270")
        if not isinstance(raw_yuv.get("files", {}), Mapping):
            raise ValueError("video.raw_yuv.files must be a mapping")
        if raw_yuv.get("use_sidecar", True) and not str(raw_yuv.get("sidecar_suffix", "")).strip():
            raise ValueError("video.raw_yuv.sidecar_suffix cannot be empty when use_sidecar is true")
