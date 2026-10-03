from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from utils.ffmpeg_utils import VideoInfo, passthrough_fps_args, probe_video


class VideoDecodeError(RuntimeError):
    def __init__(self, message: str, decoded_frames: int = 0):
        super().__init__(message)
        self.decoded_frames = decoded_frames


def _read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


class FFmpegVideoReader:
    """Stream decoded 8-bit BGR frames in presentation order from one video."""

    def __init__(self, path: Path, ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe", info: Optional[VideoInfo] = None):
        self.path = Path(path)
        self.ffmpeg_bin = ffmpeg_bin
        self.info = info or probe_video(self.path, ffprobe_bin)
        self.decoded_frames = 0

    def iter_frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        info = self.info
        frame_bytes = info.width * info.height * 3
        command = [
            self.ffmpeg_bin,
            "-hide_banner", "-loglevel", "error",
        ]
        if info.is_raw_yuv:
            command += [
                "-f", "rawvideo",
                "-pixel_format", str(info.pix_fmt),
                "-video_size", "%dx%d" % (info.width, info.height),
                "-framerate", "%.12g" % info.fps,
                "-i", str(self.path),
            ]
        else:
            command += ["-noautorotate", "-i", str(self.path)]
        command += [
            "-map", "0:%d" % info.stream_index,
            "-an", "-sn", "-dn",
        ] + passthrough_fps_args(self.ffmpeg_bin) + [
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "pipe:1",
        ]
        stderr_file = tempfile.TemporaryFile()
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_file)
        except OSError as exc:
            stderr_file.close()
            raise VideoDecodeError("Could not start FFmpeg: %s" % exc) from exc

        try:
            assert process.stdout is not None
            index = 0
            while True:
                raw = _read_exact(process.stdout, frame_bytes)
                if not raw:
                    break
                if len(raw) != frame_bytes:
                    raise VideoDecodeError(
                        "Truncated raw frame: expected %d bytes, got %d" % (frame_bytes, len(raw)),
                        decoded_frames=index,
                    )
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(info.height, info.width, 3).copy()
                frame = np.ascontiguousarray(_apply_rotation(frame, info.rotation))
                self.decoded_frames = index + 1
                yield index, frame
                index += 1

            return_code = process.wait()
            if return_code != 0:
                stderr_file.seek(0)
                message = stderr_file.read().decode("utf-8", errors="replace").strip()
                raise VideoDecodeError(
                    message or "FFmpeg returned code %d" % return_code,
                    decoded_frames=index,
                )
        finally:
            if process.stdout is not None:
                process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            stderr_file.close()
