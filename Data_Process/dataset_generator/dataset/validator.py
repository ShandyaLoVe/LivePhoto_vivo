from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from dataset.metadata import read_json
from utils.image_utils import decode_image


class ErrorCollector:
    def __init__(self, limit: int = 1000):
        self.limit = limit
        self.total = 0
        self.messages: List[str] = []

    def add(self, message: str) -> None:
        self.total += 1
        if len(self.messages) < self.limit:
            self.messages.append(message)


def _png_files(path: Path) -> List[Path]:
    return sorted([
        item for item in path.iterdir()
        if item.is_file() and not item.name.startswith(".") and item.suffix.lower() == ".png"
    ]) if path.is_dir() else []


def _entries(path: Path) -> List[Path]:
    return sorted(path.iterdir()) if path.is_dir() else []


def validate_dataset(
    dataset_root: Path,
    expected_size: Optional[Tuple[int, int]] = None,
    expected_lq_size: Optional[Tuple[int, int]] = None,
    expected_num_frames: Optional[int] = None,
) -> Dict[str, Any]:
    errors = ErrorCollector()
    source_by_split: Dict[str, Set[str]] = {name: set() for name in ("train", "val", "test")}
    clip_counts: Dict[str, int] = {}

    for split in ("train", "val", "test"):
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            errors.add("Missing split directory: %s" % split)
            clip_counts[split] = 0
            continue
        clips = sorted([item for item in split_dir.iterdir() if item.is_dir() and item.name.startswith("clip_")])
        clip_counts[split] = len(clips)
        expected_clip_names = ["clip_%06d" % index for index in range(len(clips))]
        if [item.name for item in clips] != expected_clip_names:
            errors.add("%s: clip directories are not a contiguous zero-based sequence" % split)
        for clip_dir in clips:
            prefix = "%s/%s" % (split, clip_dir.name)
            meta_path = clip_dir / "meta.json"
            if not meta_path.is_file():
                errors.add("%s: meta.json is missing" % prefix)
                continue
            try:
                meta = read_json(meta_path)
            except Exception as exc:
                errors.add("%s: invalid meta.json: %s" % (prefix, exc))
                continue
            if meta.get("clip_id") != clip_dir.name:
                errors.add("%s: clip_id does not match directory name" % prefix)
            source = meta.get("source_video")
            if not isinstance(source, str) or not source:
                errors.add("%s: source_video is missing" % prefix)
            else:
                source_by_split[split].add(source)

            gt_files = _png_files(clip_dir / "GT")
            lq_files = _png_files(clip_dir / "LQ")
            ref_files = _png_files(clip_dir / "REF")
            num_frames = int(meta.get("num_frames", -1))
            if expected_num_frames is not None and num_frames != expected_num_frames:
                errors.add("%s: num_frames=%d does not match configured %d" % (prefix, num_frames, expected_num_frames))
            if len(gt_files) != num_frames or len(lq_files) != num_frames:
                errors.add("%s: GT/LQ count mismatch (GT=%d, LQ=%d, expected=%d)" % (prefix, len(gt_files), len(lq_files), num_frames))
            for kind in ("GT", "LQ"):
                entries = _entries(clip_dir / kind)
                unexpected = [
                    item.name for item in entries
                    if not item.is_file() or item.name.startswith(".") or item.suffix.lower() != ".png"
                ]
                if unexpected:
                    errors.add("%s: %s contains unexpected entries: %s" % (prefix, kind, unexpected))
            ref_entries = _entries(clip_dir / "REF")
            if len(ref_entries) != 1 or ref_entries[0].name != "ref.png" or not ref_entries[0].is_file():
                errors.add("%s: REF must contain exactly ref.png" % prefix)
            if [item.name for item in gt_files] != [item.name for item in lq_files]:
                errors.add("%s: GT and LQ filenames are not aligned" % prefix)
            expected_names = ["%03d.png" % index for index in range(max(0, num_frames))]
            if [item.name for item in gt_files] != expected_names:
                errors.add("%s: frame filenames are not a contiguous zero-based sequence" % prefix)

            indices = meta.get("frame_indices")
            if not isinstance(indices, list) or len(indices) != num_frames:
                errors.add("%s: frame_indices count is invalid" % prefix)
            elif any(not isinstance(value, int) for value in indices) or any(a >= b for a, b in zip(indices, indices[1:])):
                errors.add("%s: frame_indices must be strictly increasing integers" % prefix)
            elif meta.get("source_start_frame") != indices[0] or meta.get("source_end_frame") != indices[-1]:
                errors.add("%s: source start/end do not match frame_indices" % prefix)
            elif meta.get("sampling_mode", "frame_interval") == "time":
                try:
                    source_fps = float(meta.get("source_fps", meta["fps"]))
                    target_fps = float(meta["target_fps"])
                    duration = float(meta["clip_duration_seconds"])
                    window_start = int(meta["source_window_start_frame"])
                    window_end = int(meta["source_window_end_frame"])
                    expected_window_frames = int(round(source_fps * duration))
                    expected_indices = [
                        window_start + int(np.floor(index * source_fps / target_fps + 0.5))
                        for index in range(num_frames)
                    ]
                    if abs(num_frames - target_fps * duration) > 1e-6:
                        errors.add("%s: num_frames does not equal target_fps * duration" % prefix)
                    if window_end - window_start + 1 != expected_window_frames:
                        errors.add("%s: source window does not match clip duration" % prefix)
                    if indices != expected_indices:
                        errors.add("%s: frame_indices do not match the target-FPS time grid" % prefix)
                    actual_duration = float(meta["clip_time_end_seconds"]) - float(meta["clip_time_start_seconds"])
                    if abs(actual_duration - duration) > 1e-6:
                        errors.add("%s: clip timestamps do not span the requested duration" % prefix)
                    timestamps = meta.get("source_sample_timestamps_seconds")
                    if not isinstance(timestamps, list) or len(timestamps) != num_frames:
                        errors.add("%s: source sample timestamps are invalid" % prefix)
                except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
                    errors.add("%s: invalid time-sampling metadata: %s" % (prefix, exc))
            elif len(indices) > 1 and any(
                current - previous != int(meta.get("frame_interval", -1))
                for previous, current in zip(indices, indices[1:])
            ):
                errors.add("%s: frame_indices do not match frame_interval" % prefix)
            if isinstance(indices, list) and meta.get("reference_frame_index") not in indices:
                errors.add("%s: reference_frame_index is not a GT frame" % prefix)

            resolution = meta.get("gt_resolution", meta.get("resolution"))
            if not (isinstance(resolution, list) and len(resolution) == 2):
                errors.add("%s: GT resolution metadata is invalid" % prefix)
                gt_resolution = None
            else:
                gt_resolution = (int(resolution[0]), int(resolution[1]))
                if expected_size is not None and gt_resolution != expected_size:
                    errors.add("%s: GT resolution %s does not match configured %s" % (prefix, gt_resolution, expected_size))
                legacy_resolution = meta.get("resolution")
                if legacy_resolution is not None and list(gt_resolution) != legacy_resolution:
                    errors.add("%s: resolution and gt_resolution disagree" % prefix)

            ref_value = meta.get("ref_resolution", resolution)
            if not (isinstance(ref_value, list) and len(ref_value) == 2):
                errors.add("%s: REF resolution metadata is invalid" % prefix)
                ref_resolution = None
            else:
                ref_resolution = (int(ref_value[0]), int(ref_value[1]))
                if gt_resolution is not None and ref_resolution != gt_resolution:
                    errors.add("%s: REF resolution must equal GT resolution" % prefix)

            lq_value = meta.get("lq_resolution", resolution)
            if not (isinstance(lq_value, list) and len(lq_value) == 2):
                errors.add("%s: LQ resolution metadata is invalid" % prefix)
                lq_resolution = None
            else:
                lq_resolution = (int(lq_value[0]), int(lq_value[1]))
                if expected_lq_size is not None and lq_resolution != expected_lq_size:
                    errors.add("%s: LQ resolution %s does not match configured %s" % (prefix, lq_resolution, expected_lq_size))

            if meta.get("gt_preserves_source_resolution"):
                source_resolution = meta.get("source_resolution")
                if not (isinstance(source_resolution, list) and len(source_resolution) == 2):
                    errors.add("%s: source_resolution metadata is invalid" % prefix)
                elif gt_resolution != (int(source_resolution[0]), int(source_resolution[1])):
                    errors.add("%s: GT does not preserve source resolution" % prefix)

            decoded_gt = []
            for kind, files in (("GT", gt_files), ("LQ", lq_files), ("REF", ref_files)):
                kind_resolution = {
                    "GT": gt_resolution,
                    "REF": ref_resolution,
                    "LQ": lq_resolution,
                }[kind]
                for image_path in files:
                    try:
                        image = decode_image(image_path)
                        if kind_resolution is not None and image.shape[:2] != kind_resolution:
                            errors.add("%s: %s/%s has resolution %s" % (prefix, kind, image_path.name, image.shape[:2]))
                        if kind == "GT":
                            decoded_gt.append(image)
                    except Exception as exc:
                        errors.add("%s: corrupted %s/%s: %s" % (prefix, kind, image_path.name, exc))
            reference_position = meta.get("reference_position")
            if len(ref_files) == 1 and isinstance(reference_position, int) and 0 <= reference_position < len(decoded_gt):
                if isinstance(indices, list) and len(indices) > reference_position:
                    if meta.get("reference_frame_index") != indices[reference_position]:
                        errors.add("%s: reference position and source index disagree" % prefix)
                try:
                    reference = decode_image(ref_files[0])
                    if not np.array_equal(reference, decoded_gt[reference_position]):
                        errors.add("%s: REF pixels are not identical to the selected GT frame" % prefix)
                except Exception:
                    pass
            degradation = meta.get("degradation")
            if not isinstance(degradation, dict):
                errors.add("%s: degradation metadata is missing" % prefix)
            elif len(degradation.get("frame_parameters") or []) != num_frames:
                errors.add("%s: degradation frame_parameters count is invalid" % prefix)

    split_names = ("train", "val", "test")
    for first_index, first in enumerate(split_names):
        for second in split_names[first_index + 1:]:
            overlap = source_by_split[first] & source_by_split[second]
            if overlap:
                errors.add("Source videos overlap between %s and %s: %s" % (first, second, sorted(overlap)))

    return {
        "valid": errors.total == 0,
        "num_errors": errors.total,
        "errors": errors.messages,
        "errors_truncated": errors.total > len(errors.messages),
        "clip_counts": clip_counts,
        "source_video_counts": {name: len(values) for name, values in source_by_split.items()},
    }
