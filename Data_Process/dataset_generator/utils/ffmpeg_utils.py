from __future__ import annotations

import json
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


def probe_video(path: Path, ffprobe_bin: str = "ffprobe", timeout: int = 60) -> VideoInfo:
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
) -> List[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError("Input directory does not exist: %s" % input_dir)
    allowed = {"." + str(ext).lower().lstrip(".") for ext in extensions}
    excluded = excluded_dir.resolve() if excluded_dir is not None else None
    candidates: List[Path] = []
    for item in input_dir.rglob("*"):
        if not item.is_file() or item.name.startswith("."):
            continue
        resolved = item.resolve()
        if excluded is not None and (resolved == excluded or excluded in resolved.parents):
            continue
        if scan_all_files or item.suffix.lower() in allowed:
            candidates.append(item)
    return sorted(candidates, key=lambda path: path.relative_to(input_dir).as_posix())
