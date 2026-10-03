"""Atomically write aligned GT, reference, LQ, and metadata clip assets."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dataset.metadata import write_json
from .images import encode_png


def write_clip(
    parent_dir: Path,
    local_name: str,
    gt_frames: Sequence[np.ndarray],
    reference: np.ndarray,
    lq_frames: Sequence[np.ndarray],
    metadata: Mapping[str, Any],
) -> Path:
    if not gt_frames or len(gt_frames) != len(lq_frames):
        raise ValueError("GT and LQ must be non-empty and have exactly the same frame count")
    final_dir = parent_dir / local_name
    if final_dir.exists():
        raise FileExistsError("Clip output already exists: %s" % final_dir)
    temp_dir = parent_dir / (".%s.tmp-%s" % (local_name, uuid.uuid4().hex))
    try:
        gt_dir, ref_dir, lq_dir = temp_dir / "GT", temp_dir / "REF", temp_dir / "LQ"
        gt_dir.mkdir(parents=True)
        ref_dir.mkdir()
        lq_dir.mkdir()
        for index, (gt, lq) in enumerate(zip(gt_frames, lq_frames)):
            filename = "%03d.png" % index
            encode_png(gt_dir / filename, gt)
            encode_png(lq_dir / filename, lq)
        encode_png(ref_dir / "ref.png", reference)
        write_json(temp_dir / "meta.json", metadata)
        os.replace(str(temp_dir), str(final_dir))
        return final_dir
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(str(temp_dir), ignore_errors=True)
        raise
