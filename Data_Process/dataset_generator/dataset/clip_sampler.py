from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Iterator, List, Optional, Tuple

import numpy as np


@dataclass
class Clip:
    frames: List[np.ndarray]
    frame_indices: List[int]

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
                yield Clip([by_index[item_index] for item_index in indices], indices)
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
                yield Clip([by_index[item_index] for item_index in indices], indices)

