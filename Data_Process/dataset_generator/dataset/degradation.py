"""Create reproducible low-quality frames with configurable degradations."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .ffmpeg import passthrough_fps_args
from .images import INTERPOLATIONS


def _bounds(spec: Any) -> Tuple[float, float]:
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return float(spec[0]), float(spec[1])
    value = float(spec)
    return value, value


def _sample(spec: Any, rng: np.random.Generator, integer: bool = False, odd: bool = False) -> Any:
    low, high = _bounds(spec)
    if low > high:
        raise ValueError("Invalid degradation range: %s" % (spec,))
    if integer:
        value = int(rng.integers(int(np.ceil(low)), int(np.floor(high)) + 1))
        if odd and value % 2 == 0:
            value = value + 1 if value < int(np.floor(high)) else value - 1
        return max(1, value)
    return float(rng.uniform(low, high)) if high > low else low


def _jitter(base: Any, spec: Any, amount: float, rng: np.random.Generator, integer: bool = False, odd: bool = False) -> Any:
    low, high = _bounds(spec)
    if high == low or amount <= 0:
        value = base
    else:
        value = float(np.clip(float(base) + rng.uniform(-1.0, 1.0) * amount * (high - low), low, high))
    if integer:
        value = int(round(value))
        if odd and value % 2 == 0:
            value = value + 1 if value < int(high) else value - 1
        value = max(1, value)
    return value


def _operation_params(config: Mapping[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    blur = config.get("blur", {})
    if blur.get("enabled"):
        params["blur_sigma"] = _sample(blur.get("sigma", [0.2, 3.0]), rng)
    motion = config.get("motion_blur", {})
    if motion.get("enabled"):
        params["motion_blur_kernel_size"] = _sample(motion.get("kernel_size", [3, 15]), rng, integer=True, odd=True)
        params["motion_blur_angle"] = _sample(motion.get("angle", [-180, 180]), rng)
    down = config.get("downsample", {})
    if down.get("enabled"):
        params["downsample_scale"] = max(1.0, _sample(down.get("scale", [1.0, 4.0]), rng))
    gaussian = config.get("gaussian_noise", {})
    if gaussian.get("enabled"):
        params["gaussian_noise_sigma"] = max(0.0, _sample(gaussian.get("sigma", [0, 20]), rng))
    poisson = config.get("poisson_noise", {})
    if poisson.get("enabled"):
        params["poisson_noise_peak"] = max(1e-6, _sample(poisson.get("peak", [30, 60]), rng))
    color = config.get("color_distortion", {})
    if color.get("enabled"):
        params["saturation"] = max(0.0, _sample(color.get("saturation", [0.85, 1.15]), rng))
        params["hue_shift"] = _sample(color.get("hue_shift", [-5, 5]), rng)
    bc = config.get("brightness_contrast", {})
    if bc.get("enabled"):
        params["brightness"] = _sample(bc.get("brightness", [-10, 10]), rng)
        params["contrast"] = max(0.0, _sample(bc.get("contrast", [0.9, 1.1]), rng))
    gamma = config.get("gamma", {})
    if gamma.get("enabled"):
        params["gamma"] = max(1e-6, _sample(gamma.get("gamma", [0.85, 1.2]), rng))
    sharpen = config.get("sharpen", {})
    if sharpen.get("enabled"):
        params["sharpen_amount"] = max(0.0, _sample(sharpen.get("amount", [0, 1.5]), rng))
        params["sharpen_radius"] = max(1e-6, _sample(sharpen.get("radius", [0.5, 1.5]), rng))
    jpeg = config.get("jpeg", {})
    if jpeg.get("enabled"):
        params["jpeg_quality"] = int(np.clip(_sample(jpeg.get("quality", [30, 95]), rng, integer=True), 1, 100))

    device = config.get("device_style", {})
    if device.get("enabled"):
        profiles = device.get("profiles", {})
        if not isinstance(profiles, Mapping) or not profiles:
            raise ValueError("degradation.device_style.profiles must be a non-empty mapping")
        requested = str(device.get("profile", "random"))
        names = sorted(str(name) for name in profiles)
        if requested == "random":
            profile_name = names[int(rng.integers(len(names)))]
        elif requested in profiles:
            profile_name = requested
        else:
            raise ValueError("Unknown device style profile: %s" % requested)
        profile = profiles[profile_name]
        if not isinstance(profile, Mapping):
            raise ValueError("Device style profile must be a mapping: %s" % profile_name)

        def sample_rgb(key: str, defaults: Sequence[Any]) -> List[float]:
            spec = profile.get(key, defaults)
            if isinstance(spec, Mapping):
                values = [spec.get(channel, defaults[index]) for index, channel in enumerate(("r", "g", "b"))]
            elif isinstance(spec, (list, tuple)) and len(spec) == 3:
                values = list(spec)
            else:
                raise ValueError("device_style.%s.%s must define R/G/B ranges" % (profile_name, key))
            return [float(_sample(value, rng)) for value in values]

        params["device_style_profile"] = profile_name
        params["device_rgb_gains"] = sample_rgb("rgb_gain", ([1.0, 1.0],) * 3)
        params["device_rgb_biases"] = sample_rgb("rgb_bias", ([0.0, 0.0],) * 3)
        params["device_contrast_multiplier"] = _sample(profile.get("contrast_multiplier", [1.0, 1.0]), rng)
        params["device_brightness_offset"] = _sample(profile.get("brightness_offset", [0.0, 0.0]), rng)
        params["device_chroma_multiplier"] = max(
            0.0,
            _sample(profile.get("chroma_multiplier", [1.0, 1.0]), rng),
        )
    return params


def _jitter_params(base: Mapping[str, Any], config: Mapping[str, Any], amount: float, rng: np.random.Generator) -> Dict[str, Any]:
    if amount <= 0:
        return dict(base)
    result = dict(base)
    specs = {
        "blur_sigma": (config.get("blur", {}).get("sigma", [0.2, 3.0]), False, False),
        "motion_blur_kernel_size": (config.get("motion_blur", {}).get("kernel_size", [3, 15]), True, True),
        "motion_blur_angle": (config.get("motion_blur", {}).get("angle", [-180, 180]), False, False),
        "downsample_scale": (config.get("downsample", {}).get("scale", [1, 4]), False, False),
        "gaussian_noise_sigma": (config.get("gaussian_noise", {}).get("sigma", [0, 20]), False, False),
        "poisson_noise_peak": (config.get("poisson_noise", {}).get("peak", [30, 60]), False, False),
        "saturation": (config.get("color_distortion", {}).get("saturation", [0.85, 1.15]), False, False),
        "hue_shift": (config.get("color_distortion", {}).get("hue_shift", [-5, 5]), False, False),
        "brightness": (config.get("brightness_contrast", {}).get("brightness", [-10, 10]), False, False),
        "contrast": (config.get("brightness_contrast", {}).get("contrast", [0.9, 1.1]), False, False),
        "gamma": (config.get("gamma", {}).get("gamma", [0.85, 1.2]), False, False),
        "sharpen_amount": (config.get("sharpen", {}).get("amount", [0, 1.5]), False, False),
        "sharpen_radius": (config.get("sharpen", {}).get("radius", [0.5, 1.5]), False, False),
        "jpeg_quality": (config.get("jpeg", {}).get("quality", [30, 95]), True, False),
    }
    for key in list(result):
        # Device-style parameters intentionally remain exactly clip-consistent.
        if key not in specs:
            continue
        spec, integer, odd = specs[key]
        result[key] = _jitter(result[key], spec, amount, rng, integer=integer, odd=odd)
    if "downsample_scale" in result:
        result["downsample_scale"] = max(1.0, float(result["downsample_scale"]))
    if "jpeg_quality" in result:
        result["jpeg_quality"] = int(np.clip(result["jpeg_quality"], 1, 100))
    return result


def _motion_kernel(size: int, angle: float) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D((size / 2.0 - 0.5, size / 2.0 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (size, size))
    total = float(kernel.sum())
    return kernel / total if total > 0 else kernel


def _apply_device_style(image: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    if "device_style_profile" not in params:
        return image
    rgb = image[:, :, ::-1].astype(np.float32) / 255.0
    gains = np.asarray(params["device_rgb_gains"], dtype=np.float32).reshape(1, 1, 3)
    biases = np.asarray(params["device_rgb_biases"], dtype=np.float32).reshape(1, 1, 3) / 255.0
    rgb = np.clip(rgb * gains + biases, 0.0, 1.0)

    luma_weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32).reshape(1, 1, 3)
    luma = np.sum(rgb * luma_weights, axis=2, keepdims=True)
    pivot = float(np.median(luma))
    contrast = float(params.get("device_contrast_multiplier", 1.0))
    brightness = float(params.get("device_brightness_offset", 0.0)) / 255.0
    rgb = np.clip((rgb - pivot) * contrast + pivot + brightness, 0.0, 1.0)

    luma = np.sum(rgb * luma_weights, axis=2, keepdims=True)
    chroma = float(params.get("device_chroma_multiplier", 1.0))
    rgb = np.clip(luma + chroma * (rgb - luma), 0.0, 1.0)
    return np.ascontiguousarray(rgb[:, :, ::-1] * 255.0, dtype=np.float32)


def _apply_frame(
    image: np.ndarray,
    params: Mapping[str, Any],
    config: Mapping[str, Any],
    rng: np.random.Generator,
    output_size: Optional[Tuple[int, int]] = None,
    output_interpolation: str = "area",
) -> np.ndarray:
    value = image.astype(np.float32)
    sigma = float(params.get("blur_sigma", 0.0))
    if sigma > 1e-6:
        value = cv2.GaussianBlur(value, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if "motion_blur_kernel_size" in params:
        kernel = _motion_kernel(int(params["motion_blur_kernel_size"]), float(params["motion_blur_angle"]))
        value = cv2.filter2D(value, -1, kernel)

    scale = float(params.get("downsample_scale", 1.0))
    if scale > 1.000001:
        height, width = value.shape[:2]
        small_width, small_height = max(1, int(round(width / scale))), max(1, int(round(height / scale)))
        down_cfg = config.get("downsample", {})
        down_name, up_name = down_cfg.get("down_interpolation", "area"), down_cfg.get("up_interpolation", "cubic")
        if down_name not in INTERPOLATIONS or up_name not in INTERPOLATIONS:
            raise ValueError("Unsupported downsample interpolation")
        value = cv2.resize(value, (small_width, small_height), interpolation=INTERPOLATIONS[down_name])
        value = cv2.resize(value, (width, height), interpolation=INTERPOLATIONS[up_name])

    if output_size is not None:
        output_height, output_width = output_size
        if output_height <= 0 or output_width <= 0:
            raise ValueError("LQ output dimensions must be > 0")
        interpolation_name = str(output_interpolation).lower()
        if interpolation_name not in INTERPOLATIONS:
            raise ValueError("Unsupported LQ resize interpolation: %s" % interpolation_name)
        if value.shape[:2] != (output_height, output_width):
            value = cv2.resize(
                value,
                (output_width, output_height),
                interpolation=INTERPOLATIONS[interpolation_name],
            )

    value = _apply_device_style(value, params)

    noise_sigma = float(params.get("gaussian_noise_sigma", 0.0))
    if noise_sigma > 0:
        value += rng.normal(0.0, noise_sigma, value.shape).astype(np.float32)
    if "poisson_noise_peak" in params:
        peak = float(params["poisson_noise_peak"])
        value = rng.poisson(np.clip(value, 0, 255) / 255.0 * peak).astype(np.float32) * (255.0 / peak)

    if "saturation" in params or "hue_shift" in params:
        hsv = cv2.cvtColor(np.clip(value, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= float(params.get("saturation", 1.0))
        hsv[:, :, 0] = np.mod(hsv[:, :, 0] + float(params.get("hue_shift", 0.0)), 180.0)
        hsv[:, :, 1:] = np.clip(hsv[:, :, 1:], 0, 255)
        value = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    value = value * float(params.get("contrast", 1.0)) + float(params.get("brightness", 0.0))
    if "gamma" in params:
        value = 255.0 * np.power(np.clip(value, 0, 255) / 255.0, float(params["gamma"]))
    if "sharpen_amount" in params and float(params["sharpen_amount"]) > 0:
        blurred = cv2.GaussianBlur(value, (0, 0), float(params["sharpen_radius"]))
        value = value + float(params["sharpen_amount"]) * (value - blurred)

    output = np.clip(value, 0, 255).round().astype(np.uint8)
    if "jpeg_quality" in params:
        ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, int(params["jpeg_quality"])])
        if not ok:
            raise RuntimeError("JPEG degradation encode failed")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("JPEG degradation decode failed")
        output = decoded
    return np.ascontiguousarray(output)


def _video_compress(
    frames: Sequence[np.ndarray],
    fps: float,
    config: Mapping[str, Any],
    rng: np.random.Generator,
    ffmpeg_bin: str,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    height, width = frames[0].shape[:2]
    codec = str(config.get("codec", "libx264"))
    crf = int(_sample(config.get("crf", [18, 38]), rng, integer=True))
    preset = str(config.get("preset", "medium"))
    keyint = max(1, int(config.get("keyint", 12)))
    pixel_format = "yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p"
    with tempfile.TemporaryDirectory(prefix="dataset-video-compression-") as temp_dir:
        encoded_path = Path(temp_dir) / "clip.mkv"
        encode_command = [
            ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (width, height),
            "-r", "%.8f" % max(float(fps), 1e-3), "-i", "pipe:0",
            "-an", "-c:v", codec, "-preset", preset, "-crf", str(crf),
            "-g", str(keyint), "-pix_fmt", pixel_format, "-frames:v", str(len(frames)),
            str(encoded_path),
        ]
        raw_input = b"".join(np.ascontiguousarray(frame).tobytes() for frame in frames)
        encoded = subprocess.run(encode_command, input=raw_input, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if encoded.returncode != 0:
            raise RuntimeError("Video compression encode failed: %s" % encoded.stderr.decode("utf-8", errors="replace"))
        decode_command = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-i", str(encoded_path),
            "-an",
        ] + passthrough_fps_args(ffmpeg_bin) + ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        decoded = subprocess.run(decode_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if decoded.returncode != 0:
            raise RuntimeError("Video compression decode failed: %s" % decoded.stderr.decode("utf-8", errors="replace"))
    frame_size = height * width * 3
    if len(decoded.stdout) != len(frames) * frame_size:
        raise RuntimeError("Video compression changed the frame count or dimensions")
    output = []
    for index in range(len(frames)):
        start = index * frame_size
        output.append(np.frombuffer(decoded.stdout[start:start + frame_size], dtype=np.uint8).reshape(height, width, 3).copy())
    return output, {"codec": codec, "crf": crf, "preset": preset, "keyint": keyint, "pixel_format": pixel_format}


def degrade_clip(
    gt_frames: Sequence[np.ndarray],
    config: Mapping[str, Any],
    rng: np.random.Generator,
    fps: float,
    ffmpeg_bin: str = "ffmpeg",
    output_size: Optional[Tuple[int, int]] = None,
    output_interpolation: str = "area",
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    if not gt_frames:
        raise ValueError("Cannot degrade an empty clip")
    mode = str(config.get("mode", "clip_consistent"))
    variation = float(config.get("temporal_variation", 0.0))
    base = _operation_params(config, rng) if mode == "clip_consistent" else None
    output: List[np.ndarray] = []
    frame_params: List[Dict[str, Any]] = []
    for frame in gt_frames:
        params = _jitter_params(base, config, variation, rng) if base is not None else _operation_params(config, rng)
        output.append(_apply_frame(frame, params, config, rng, output_size, output_interpolation))
        frame_params.append(params)

    video_meta = None
    video_config = config.get("video_compression", {})
    if video_config.get("enabled"):
        output, video_meta = _video_compress(output, fps, video_config, rng, ffmpeg_bin)

    metadata: Dict[str, Any] = {
        "mode": mode,
        "temporal_variation": variation if mode == "clip_consistent" else None,
        "base_parameters": base,
        "frame_parameters": frame_params,
        "video_compression": video_meta,
        "lq_resize": {
            "height": int(output[0].shape[0]),
            "width": int(output[0].shape[1]),
            "interpolation": str(output_interpolation),
        },
        "pipeline_order": [
            "gaussian_blur", "motion_blur", "downsample_upsample", "lq_resize", "device_style", "noise",
            "color", "brightness_contrast", "gamma", "sharpen", "jpeg", "video_compression",
        ],
    }
    return output, metadata
