from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


_FFMPEG_MAJOR_CACHE: Dict[str, int] = {}


class ProbeError(RuntimeError):
    pass


def passthrough_fps_args(ffmpeg_bin: str) -> List[str]:
    """Use the modern option on FFmpeg 5+, retaining compatibility with 4.x."""
    if ffmpeg_bin not in _FFMPEG_MAJOR_CACHE:
        try:
            completed = subprocess.run(
                [ffmpeg_bin, "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=10,
                check=False,
            )
            first_line = completed.stdout.decode("utf-8", errors="replace").splitlines()[0]
            match = re.search(r"ffmpeg version\s+(?:n)?(\d+)", first_line)
            _FFMPEG_MAJOR_CACHE[ffmpeg_bin] = int(match.group(1)) if match else 5
        except (OSError, subprocess.TimeoutExpired, IndexError):
            _FFMPEG_MAJOR_CACHE[ffmpeg_bin] = 5
    return ["-fps_mode", "passthrough"] if _FFMPEG_MAJOR_CACHE[ffmpeg_bin] >= 5 else ["-vsync", "0"]


@dataclass(frozen=True)
class VideoInfo:
    path: str
    stream_index: int
    codec_name: str
    width: int
    height: int
    display_width: int
    display_height: int
    rotation: int
    fps: float
    avg_frame_rate: str
    r_frame_rate: str
    duration: Optional[float]
    nb_frames: Optional[int]
    pix_fmt: Optional[str]
    is_raw_yuv: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VideoInfo":
        return cls(**dict(value))


def _rate_to_float(rate: Any) -> float:
    if not rate or rate in {"0/0", "N/A"}:
        return 0.0
    try:
        return float(Fraction(str(rate)))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _rotation(stream: Mapping[str, Any]) -> int:
    rotation: Any = (stream.get("tags") or {}).get("rotate", 0)
    for item in stream.get("side_data_list") or []:
        if "rotation" in item:
            rotation = item["rotation"]
            break
    try:
        value = int(round(float(rotation))) % 360
    except (TypeError, ValueError):
        value = 0
    return min(
        (0, 90, 180, 270),
        key=lambda candidate: min(abs(candidate - value), 360 - abs(candidate - value)),
    )


def raw_yuv_sidecar_path(path: Path, raw_config: Mapping[str, Any]) -> Path:
    suffix = str(raw_config.get("sidecar_suffix", ".json"))
    return Path(str(path) + suffix)


def _raw_yuv_options(path: Path, raw_config: Mapping[str, Any], source_key: Optional[str]) -> Dict[str, Any]:
    options: Dict[str, Any] = {
        "width": raw_config.get("width"),
        "height": raw_config.get("height"),
        "pix_fmt": raw_config.get("pix_fmt", "yuv420p"),
        "fps": raw_config.get("fps", 30.0),
        "rotation": raw_config.get("rotation", 0),
    }
    file_overrides = raw_config.get("files") or {}
    if not isinstance(file_overrides, Mapping):
        raise ProbeError("video.raw_yuv.files must be a mapping")
    override = file_overrides.get(source_key) if source_key is not None else None
    if override is None:
        override = file_overrides.get(path.name)
    if override is not None:
        if not isinstance(override, Mapping):
            raise ProbeError("RAW YUV override for %s must be a mapping" % (source_key or path.name))
        options.update(override)

    sidecar_path = raw_yuv_sidecar_path(path, raw_config)
    if bool(raw_config.get("use_sidecar", True)) and sidecar_path.is_file():
        try:
            with sidecar_path.open("r", encoding="utf-8") as handle:
                sidecar = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ProbeError("Invalid RAW YUV sidecar %s: %s" % (sidecar_path, exc)) from exc
        if not isinstance(sidecar, Mapping):
            raise ProbeError("RAW YUV sidecar must contain a JSON object: %s" % sidecar_path)
        options.update(sidecar)

    if "pixel_format" in options:
        options["pix_fmt"] = options["pixel_format"]
    try:
        width = int(options["width"])
        height = int(options["height"])
        fps = float(options["fps"])
        rotation = int(options.get("rotation", 0)) % 360
    except (KeyError, TypeError, ValueError) as exc:
        raise ProbeError(
            "RAW YUV requires numeric width, height and fps in video.raw_yuv, a per-file override, or %s"
            % sidecar_path
        ) from exc
    pix_fmt = str(options.get("pix_fmt") or "").strip()
    if width <= 0 or height <= 0 or fps <= 0 or not pix_fmt:
        raise ProbeError("RAW YUV width, height, fps and pix_fmt must be valid positive values")
    if rotation not in {0, 90, 180, 270}:
        raise ProbeError("RAW YUV rotation must be 0, 90, 180 or 270")
    return {"width": width, "height": height, "fps": fps, "pix_fmt": pix_fmt, "rotation": rotation}


def probe_raw_yuv(
    path: Path,
    raw_config: Mapping[str, Any],
    ffprobe_bin: str = "ffprobe",
    timeout: int = 60,
    source_key: Optional[str] = None,
) -> VideoInfo:
    if not bool(raw_config.get("enabled", True)):
        raise ProbeError("RAW YUV support is disabled by video.raw_yuv.enabled")
    if not path.is_file() or path.stat().st_size <= 0:
        raise ProbeError("RAW YUV file is empty or missing")
    options = _raw_yuv_options(path, raw_config, source_key)
    width, height = int(options["width"]), int(options["height"])
    fps, pix_fmt, rotation = float(options["fps"]), str(options["pix_fmt"]), int(options["rotation"])
    command = [
        ffprobe_bin,
        "-v", "error",
        "-f", "rawvideo",
        "-pixel_format", pix_fmt,
        "-video_size", "%dx%d" % (width, height),
        "-framerate", "%.12g" % fps,
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProbeError("ffprobe failed for RAW YUV %s: %s" % (path, exc)) from exc
    stderr = completed.stderr.decode("utf-8", errors="replace").strip()
    if completed.returncode != 0:
        raise ProbeError(stderr or "ffprobe returned code %d for RAW YUV" % completed.returncode)
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
        stream = next(item for item in payload.get("streams", []) if item.get("codec_type") == "video")
    except (UnicodeDecodeError, json.JSONDecodeError, StopIteration) as exc:
        raise ProbeError("Invalid ffprobe result for RAW YUV %s" % path) from exc
    duration_value = stream.get("duration") or (payload.get("format") or {}).get("duration")
    try:
        duration = float(duration_value) if duration_value not in (None, "N/A") else None
    except (TypeError, ValueError):
        duration = None
    file_size = path.stat().st_size
    nb_frames: Optional[int] = None
    try:
        bit_rate = int(stream["bit_rate"])
        frame_bytes_value = bit_rate / (8.0 * fps)
        frame_bytes = int(round(frame_bytes_value))
        if frame_bytes <= 0 or abs(frame_bytes - frame_bytes_value) > 1e-6:
            raise ValueError("invalid raw frame size")
        if file_size % frame_bytes != 0:
            raise ProbeError(
                "RAW YUV byte size %d is not a whole number of %d-byte frames; check width/height/pix_fmt or file integrity"
                % (file_size, frame_bytes)
            )
        nb_frames = file_size // frame_bytes
    except ProbeError:
        raise
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        try:
            nb_frames = int(stream["duration_ts"]) if stream.get("duration_ts") not in (None, "N/A") else None
        except (TypeError, ValueError):
            nb_frames = None
    rate = Fraction(str(fps)).limit_denominator(100000)
    rate_text = "%d/%d" % (rate.numerator, rate.denominator)
    display_width, display_height = (height, width) if rotation in {90, 270} else (width, height)
    return VideoInfo(
        path=str(path),
        stream_index=int(stream.get("index", 0)),
        codec_name="rawvideo",
        width=width,
        height=height,
        display_width=display_width,
        display_height=display_height,
        rotation=rotation,
        fps=fps,
        avg_frame_rate=rate_text,
        r_frame_rate=rate_text,
        duration=duration,
        nb_frames=nb_frames,
        pix_fmt=str(stream.get("pix_fmt") or pix_fmt),
        is_raw_yuv=True,
    )


def probe_video(
    path: Path,
    ffprobe_bin: str = "ffprobe",
    timeout: int = 60,
    raw_yuv_config: Optional[Mapping[str, Any]] = None,
    source_key: Optional[str] = None,
) -> VideoInfo:
    if path.suffix.lower() == ".yuv":
        if raw_yuv_config is None:
            raise ProbeError("RAW .yuv input requires video.raw_yuv configuration")
        return probe_raw_yuv(path, raw_yuv_config, ffprobe_bin, timeout, source_key)
    command = [
        ffprobe_bin,
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProbeError("ffprobe failed for %s: %s" % (path, exc)) from exc
    stderr = completed.stderr.decode("utf-8", errors="replace").strip()
    if completed.returncode != 0:
        raise ProbeError(stderr or "ffprobe returned code %d" % completed.returncode)
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError("Invalid ffprobe JSON for %s" % path) from exc

    video_streams = [stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"]
    if not video_streams:
        raise ProbeError("No video stream")
    stream = video_streams[0]
    width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ProbeError("Video stream has invalid dimensions")
    rotation = _rotation(stream)
    display_width, display_height = (height, width) if rotation in {90, 270} else (width, height)
    avg_rate = str(stream.get("avg_frame_rate") or "0/0")
    nominal_rate = str(stream.get("r_frame_rate") or "0/0")
    fps = _rate_to_float(avg_rate) or _rate_to_float(nominal_rate)
    if fps <= 0:
        fps = 1.0

    duration_value = stream.get("duration") or (payload.get("format") or {}).get("duration")
    try:
        duration = float(duration_value) if duration_value not in (None, "N/A") else None
    except (TypeError, ValueError):
        duration = None
    frames_value = stream.get("nb_frames")
    try:
        nb_frames = int(frames_value) if frames_value not in (None, "N/A") else None
    except (TypeError, ValueError):
        nb_frames = None

    return VideoInfo(
        path=str(path),
        stream_index=int(stream["index"]),
        codec_name=str(stream.get("codec_name") or "unknown"),
        width=width,
        height=height,
        display_width=display_width,
        display_height=display_height,
        rotation=rotation,
        fps=fps,
        avg_frame_rate=avg_rate,
        r_frame_rate=nominal_rate,
        duration=duration,
        nb_frames=nb_frames,
        pix_fmt=stream.get("pix_fmt"),
    )


def discover_candidates(
    input_dir: Path,
    extensions: Iterable[str],
    scan_all_files: bool,
    excluded_dir: Optional[Path] = None,
    raw_yuv_sidecar_suffix: str = ".json",
    excluded_dir_names: Iterable[str] = (),
) -> List[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError("Input directory does not exist: %s" % input_dir)
    allowed = {"." + str(ext).lower().lstrip(".") for ext in extensions}
    excluded = excluded_dir.resolve() if excluded_dir is not None else None
    excluded_names = {str(name) for name in excluded_dir_names}
    candidates: List[Path] = []
    for root_value, dir_names, file_names in os.walk(str(input_dir), topdown=True, followlinks=False):
        root = Path(root_value)
        retained_dirs = []
        for name in dir_names:
            if name.startswith(".") or name in excluded_names:
                continue
            child = (root / name).resolve()
            if excluded is not None and (child == excluded or excluded in child.parents):
                continue
            retained_dirs.append(name)
        dir_names[:] = retained_dirs
        for name in file_names:
            if name.startswith("."):
                continue
            item = root / name
            if raw_yuv_sidecar_suffix and item.name.lower().endswith((".yuv" + raw_yuv_sidecar_suffix).lower()):
                raw_path = Path(str(item)[:-len(raw_yuv_sidecar_suffix)])
                if raw_path.is_file():
                    continue
            resolved = item.resolve()
            if excluded is not None and (resolved == excluded or excluded in resolved.parents):
                continue
            if scan_all_files or item.suffix.lower() in allowed:
                candidates.append(item)
    return sorted(candidates, key=lambda path: path.relative_to(input_dir).as_posix())
