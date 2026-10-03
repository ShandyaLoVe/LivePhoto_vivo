from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Iterator, List, Optional, Tuple

import numpy as np


@dataclass
class Clip:
    frames: List[np.ndarray]
    frame_indices: List[int]
    window_start_frame: Optional[int] = None
    window_end_frame: Optional[int] = None

    @property
    def start_frame(self) -> int:
        return self.frame_indices[0]

    @property
    def end_frame(self) -> int:
        return self.frame_indices[-1]


def iter_clips(
    frame_stream: Iterable[Tuple[int, np.ndarray]],
    num_frames: int,
    frame_interval: int,
    clip_stride: int,
    drop_last: bool = True,
) -> Iterator[Clip]:
    """Sample clips in one streaming pass without decoding any source frame twice.

    With drop_last=False, a final full clip is aligned to the last decoded frame.
    Videos shorter than one full temporal span still yield no samples so the strict
    frame-count and strictly-increasing-index invariants are never weakened.
    """
    span = (num_frames - 1) * frame_interval + 1
    buffer: Deque[Tuple[int, np.ndarray]] = deque()
    tail: Deque[Tuple[int, np.ndarray]] = deque(maxlen=span)
    next_start = 0
    last_yielded_start: Optional[int] = None
    last_index = -1

    for index, frame in frame_stream:
        if index != last_index + 1:
            raise ValueError("Decoded frame indices must be contiguous")
        last_index = index
        buffer.append((index, frame))
        tail.append((index, frame))

        while index >= next_start + span - 1:
            by_index = {item_index: item_frame for item_index, item_frame in buffer}
            indices = [next_start + offset * frame_interval for offset in range(num_frames)]
            if all(item_index in by_index for item_index in indices):
                yield Clip(
                    [by_index[item_index] for item_index in indices],
                    indices,
                    window_start_frame=next_start,
                    window_end_frame=next_start + span - 1,
                )
                last_yielded_start = next_start
            next_start += clip_stride
            while buffer and buffer[0][0] < next_start:
                buffer.popleft()

    if not drop_last and last_index + 1 >= span:
        tail_start = last_index - span + 1
        if tail_start != last_yielded_start:
            by_index = {item_index: item_frame for item_index, item_frame in tail}
            indices = [tail_start + offset * frame_interval for offset in range(num_frames)]
            if all(item_index in by_index for item_index in indices):
                yield Clip(
                    [by_index[item_index] for item_index in indices],
                    indices,
                    window_start_frame=tail_start,
                    window_end_frame=tail_start + span - 1,
                )


def iter_time_clips(
    frame_stream: Iterable[Tuple[int, np.ndarray]],
    num_frames: int,
    source_fps: float,
    target_fps: float,
    clip_duration_seconds: float,
    clip_stride_seconds: float,
    drop_last: bool = True,
) -> Iterator[Clip]:
    """Sample a fixed number of frames from fixed-duration CFR source windows.

    Source indices are nearest-neighbour samples of an exact target-FPS grid.
    Window completion is checked independently of the last selected frame, so a
    3-second clip always requires the full 3 seconds of source material.
    """
    if source_fps <= 0 or target_fps <= 0 or clip_duration_seconds <= 0 or clip_stride_seconds <= 0:
        raise ValueError("Time-based sampling rates and durations must be > 0")
    if abs(num_frames - target_fps * clip_duration_seconds) > 1e-6:
        raise ValueError("num_frames must equal target_fps * clip_duration_seconds")
    window_frames = int(round(source_fps * clip_duration_seconds))
    stride_frames = int(round(source_fps * clip_stride_seconds))
    if window_frames <= 0 or stride_frames <= 0:
        raise ValueError("Time-based source window and stride must contain at least one frame")
    offsets = [int(np.floor(index * source_fps / target_fps + 0.5)) for index in range(num_frames)]
    if len(set(offsets)) != len(offsets):
        raise ValueError("target_fps cannot exceed source_fps without duplicating frames")
    if offsets[-1] >= window_frames:
        raise ValueError("The target sampling grid exceeds the requested clip duration")

    buffer: Deque[Tuple[int, np.ndarray]] = deque()
    tail: Deque[Tuple[int, np.ndarray]] = deque(maxlen=window_frames)
    next_start = 0
    last_yielded_start: Optional[int] = None
    last_index = -1

    for index, frame in frame_stream:
        if index != last_index + 1:
            raise ValueError("Decoded frame indices must be contiguous")
        last_index = index
        buffer.append((index, frame))
        tail.append((index, frame))

        while index >= next_start + window_frames - 1:
            by_index = {item_index: item_frame for item_index, item_frame in buffer}
            indices = [next_start + offset for offset in offsets]
            if all(item_index in by_index for item_index in indices):
                yield Clip(
                    [by_index[item_index] for item_index in indices],
                    indices,
                    window_start_frame=next_start,
                    window_end_frame=next_start + window_frames - 1,
                )
                last_yielded_start = next_start
            next_start += stride_frames
            while buffer and buffer[0][0] < next_start:
                buffer.popleft()

    if not drop_last and last_index + 1 >= window_frames:
        tail_start = last_index - window_frames + 1
        if tail_start != last_yielded_start:
            by_index = {item_index: item_frame for item_index, item_frame in tail}
            indices = [tail_start + offset for offset in offsets]
            if all(item_index in by_index for item_index in indices):
                yield Clip(
                    [by_index[item_index] for item_index in indices],
                    indices,
                    window_start_frame=tail_start,
                    window_end_frame=tail_start + window_frames - 1,
                )
