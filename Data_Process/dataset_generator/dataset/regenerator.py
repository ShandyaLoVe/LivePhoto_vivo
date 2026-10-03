"""Regenerate LQ frames from existing GT frames with rollback safety."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .config import lq_output_size, output_size
from .degradation import degrade_clip
from .images import decode_image, encode_png
from .metadata import read_json, write_json
from .seeding import derive_seed
from .validator import validate_dataset


PreparedClip = Tuple[Path, Path, Path, Dict[str, Any], Dict[str, Any]]


def _clip_dirs(dataset_root: Path) -> List[Path]:
    clips: List[Path] = []
    for split in ("train", "val", "test"):
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError("Missing split directory: %s" % split_dir)
        clips.extend(sorted(path for path in split_dir.glob("clip_*") if path.is_dir()))
    return clips


def _prepare_clip(clip_dir: Path, config: Dict[str, Any]) -> PreparedClip:
    meta_path = clip_dir / "meta.json"
    old_meta = read_json(meta_path)
    gt_files = sorted(path for path in (clip_dir / "GT").glob("*.png") if not path.name.startswith("."))
    expected_count = int(old_meta.get("num_frames", -1))
    if len(gt_files) != expected_count:
        raise ValueError("%s: expected %d GT frames, found %d" % (clip_dir, expected_count, len(gt_files)))

    temp_lq = clip_dir / ".LQ.regenerated"
    backup_lq = clip_dir / ".LQ.previous"
    if temp_lq.exists() or backup_lq.exists():
        raise FileExistsError("Stale LQ regeneration directory exists in %s" % clip_dir)
    temp_lq.mkdir()

    try:
        gt_frames = [decode_image(path) for path in gt_files]
        regeneration_seed = derive_seed(int(old_meta["seed"]), "regenerate_lq_v1")
        rng = np.random.default_rng(regeneration_seed)
        lq_frames, degradation = degrade_clip(
            gt_frames,
            config["degradation"],
            rng,
            fps=float(old_meta["fps"]),
            ffmpeg_bin=str(config["ffmpeg_bin"]),
            output_size=lq_output_size(config["image"]),
            output_interpolation=str(config["image"].get("lq_interpolation", "area")),
        )
        for index, frame in enumerate(lq_frames):
            encode_png(temp_lq / ("%03d.png" % index), frame)

        new_meta = dict(old_meta)
        new_meta["schema_version"] = max(2, int(old_meta.get("schema_version", 1)))
        new_meta["lq_resolution"] = [int(lq_frames[0].shape[0]), int(lq_frames[0].shape[1])]
        new_meta["lq_regeneration_seed"] = regeneration_seed
        degradation["source"] = "regenerated_from_existing_gt"
        degradation["regeneration_seed"] = regeneration_seed
        new_meta["degradation"] = degradation
        return clip_dir, temp_lq, backup_lq, old_meta, new_meta
    except Exception:
        shutil.rmtree(str(temp_lq), ignore_errors=True)
        raise


def regenerate_lq(dataset_root: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    dataset_root = dataset_root.resolve()
    clips = _clip_dirs(dataset_root)
    if not clips:
        raise ValueError("Dataset contains no clip directories: %s" % dataset_root)

    cv2.setNumThreads(1)
    prepared: List[PreparedClip] = []
    try:
        for clip_dir in clips:
            prepared.append(_prepare_clip(clip_dir, config))
    except Exception:
        for _, temp_lq, _, _, _ in prepared:
            shutil.rmtree(str(temp_lq), ignore_errors=True)
        raise

    swapped: List[PreparedClip] = []
    try:
        for item in prepared:
            clip_dir, temp_lq, backup_lq, _, new_meta = item
            lq_dir = clip_dir / "LQ"
            if not lq_dir.is_dir():
                raise FileNotFoundError("Missing LQ directory: %s" % lq_dir)
            os.replace(str(lq_dir), str(backup_lq))
            swapped.append(item)
            os.replace(str(temp_lq), str(lq_dir))
            write_json(clip_dir / "meta.json", new_meta)

        expected_gt = (
            None
            if config["image"].get("preserve_source_resolution", False)
            else output_size(config["image"])
        )
        report = validate_dataset(
            dataset_root,
            expected_size=expected_gt,
            expected_lq_size=lq_output_size(config["image"]),
            expected_num_frames=int(config["clip"]["num_frames"]),
        )
        if not report["valid"]:
            raise RuntimeError("Regenerated LQ validation failed: %s" % report["errors"][:10])

        summary_path = dataset_root / "dataset_summary.json"
        summary = read_json(summary_path) if summary_path.is_file() else {}
        summary["validation"] = report
        summary["lq_regeneration"] = {
            "num_clips": len(clips),
            "source": "existing_gt",
            "device_style_enabled": bool(config["degradation"].get("device_style", {}).get("enabled")),
        }
        write_json(summary_path, summary)
        for _, _, backup_lq, _, _ in swapped:
            shutil.rmtree(str(backup_lq))
        return report
    except Exception:
        for clip_dir, temp_lq, backup_lq, old_meta, _ in reversed(swapped):
            lq_dir = clip_dir / "LQ"
            shutil.rmtree(str(lq_dir), ignore_errors=True)
            if backup_lq.exists():
                os.replace(str(backup_lq), str(lq_dir))
            write_json(clip_dir / "meta.json", old_meta)
            shutil.rmtree(str(temp_lq), ignore_errors=True)
        for _, temp_lq, backup_lq, _, _ in prepared[len(swapped):]:
            shutil.rmtree(str(temp_lq), ignore_errors=True)
            shutil.rmtree(str(backup_lq), ignore_errors=True)
        raise
