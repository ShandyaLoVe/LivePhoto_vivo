"""Orchestrate deterministic video-to-dataset generation."""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .clip_sampler import iter_clips, iter_time_clips
from .config import lq_output_size, output_size, validate_config
from .degradation import degrade_clip
from .ffmpeg import VideoInfo, discover_candidates, probe_video
from .images import transform_clip
from .metadata import read_json, write_json
from .reference_selector import select_reference
from .seeding import derive_seed, seed_everything
from .splitting import assign_splits
from .validator import validate_dataset
from .video_reader import FFmpegVideoReader, VideoDecodeError
from .writer import write_clip


def _process_video(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Worker entry point. One process owns one source and decodes it once."""
    source_path = Path(task["source_path"])
    source_video = str(task["source_video"])
    split = str(task["split"])
    config = task["config"]
    video_seed = int(task["video_seed"])
    stage_root = Path(task["stage_root"])
    video_key = str(task["video_key"])
    video_stage = stage_root / split / video_key
    seed_everything(video_seed)
    cv2.setNumThreads(1)
    reader: Optional[FFmpegVideoReader] = None
    clip_paths: List[str] = []
    warning = None
    try:
        video_stage.mkdir(parents=True, exist_ok=False)
        info = VideoInfo.from_dict(task["video_info"])
        reader = FFmpegVideoReader(source_path, config["ffmpeg_bin"], config["ffprobe_bin"], info=info)
        clip_cfg = config["clip"]
        stream = reader.iter_frames()
        try:
            time_sampling = clip_cfg.get("sampling_fps") is not None
            if time_sampling:
                duration = float(clip_cfg["clip_duration_seconds"])
                sampled = iter_time_clips(
                    stream,
                    num_frames=int(clip_cfg["num_frames"]),
                    source_fps=info.fps,
                    target_fps=float(clip_cfg["sampling_fps"]),
                    clip_duration_seconds=duration,
                    clip_stride_seconds=float(clip_cfg.get("clip_stride_seconds") or duration),
                    drop_last=bool(clip_cfg.get("drop_last", True)),
                )
            else:
                sampled = iter_clips(
                    stream,
                    num_frames=int(clip_cfg["num_frames"]),
                    frame_interval=int(clip_cfg["frame_interval"]),
                    clip_stride=int(clip_cfg["clip_stride"]),
                    drop_last=bool(clip_cfg.get("drop_last", True)),
                )
            for local_index, clip in enumerate(sampled):
                clip_seed = derive_seed(video_seed, clip.start_frame, local_index)
                rng = np.random.default_rng(clip_seed)
                gt_frames, spatial_plan = transform_clip(clip.frames, config["image"], rng)
                reference, reference_frame_index, reference_position = select_reference(
                    gt_frames,
                    clip.frame_indices,
                    str(config["reference"]["strategy"]),
                    rng,
                )
                lq_frames, degradation_meta = degrade_clip(
                    gt_frames,
                    config["degradation"],
                    rng,
                    fps=float(clip_cfg["sampling_fps"]) if time_sampling else info.fps,
                    ffmpeg_bin=str(config["ffmpeg_bin"]),
                    output_size=lq_output_size(config["image"]),
                    output_interpolation=str(config["image"].get("lq_interpolation", "area")),
                )
                local_name = "clip_local_%06d" % local_index
                metadata = {
                    "schema_version": 2,
                    "clip_id": local_name,
                    "source_video": source_video,
                    "source_start_frame": clip.start_frame,
                    "source_end_frame": clip.end_frame,
                    "frame_indices": clip.frame_indices,
                    "sampling_mode": "time" if time_sampling else "frame_interval",
                    "frame_interval": None if time_sampling else int(clip_cfg["frame_interval"]),
                    "fps": float(clip_cfg["sampling_fps"]) if time_sampling else info.fps,
                    "source_fps": info.fps,
                    "avg_frame_rate": info.avg_frame_rate,
                    "nominal_frame_rate": info.r_frame_rate,
                    "num_frames": len(gt_frames),
                    "source_window_start_frame": clip.window_start_frame,
                    "source_window_end_frame": clip.window_end_frame,
                    "reference_frame_index": reference_frame_index,
                    "reference_position": reference_position,
                    "reference_strategy": str(config["reference"]["strategy"]),
                    "resolution": [int(gt_frames[0].shape[0]), int(gt_frames[0].shape[1])],
                    "gt_resolution": [int(gt_frames[0].shape[0]), int(gt_frames[0].shape[1])],
                    "ref_resolution": [int(reference.shape[0]), int(reference.shape[1])],
                    "lq_resolution": [int(lq_frames[0].shape[0]), int(lq_frames[0].shape[1])],
                    "gt_preserves_source_resolution": bool(config["image"].get("preserve_source_resolution", False)),
                    "source_resolution": [info.display_height, info.display_width],
                    "source_codec": info.codec_name,
                    "source_pixel_format": info.pix_fmt,
                    "source_is_raw_yuv": info.is_raw_yuv,
                    "source_rotation": info.rotation,
                    "decode_pixel_format": "bgr24",
                    "spatial_transform": spatial_plan.to_dict(),
                    "degradation": degradation_meta,
                    "seed": clip_seed,
                }
                if time_sampling:
                    metadata.update({
                        "target_fps": float(clip_cfg["sampling_fps"]),
                        "clip_duration_seconds": float(clip_cfg["clip_duration_seconds"]),
                        "clip_stride_seconds": float(clip_cfg.get("clip_stride_seconds") or clip_cfg["clip_duration_seconds"]),
                        "clip_time_start_seconds": float(clip.window_start_frame) / info.fps,
                        "clip_time_end_seconds": float(clip.window_end_frame + 1) / info.fps,
                        "source_sample_timestamps_seconds": [float(index) / info.fps for index in clip.frame_indices],
                    })
                clip_path = write_clip(video_stage, local_name, gt_frames, reference, lq_frames, metadata)
                clip_paths.append(str(clip_path))
            if info.is_raw_yuv and info.nb_frames is not None and reader.decoded_frames != info.nb_frames:
                raise RuntimeError(
                    "RAW YUV decode ended at %d/%d frames; refusing an incomplete dataset"
                    % (reader.decoded_frames, info.nb_frames)
                )
        except VideoDecodeError as exc:
            if not bool(config["video"].get("allow_partial_decode", True)) or not clip_paths:
                raise
            warning = "Partial decode accepted after %d frames: %s" % (exc.decoded_frames, exc)

        return {
            "ok": True,
            "source_video": source_video,
            "split": split,
            "clips": clip_paths,
            "num_clips": len(clip_paths),
            "decoded_frames": reader.decoded_frames if reader else 0,
            "warning": warning,
        }
    except Exception as exc:
        if video_stage.exists():
            shutil.rmtree(str(video_stage), ignore_errors=True)
        return {
            "ok": False,
            "source_video": source_video,
            "split": split,
            "clips": [],
            "num_clips": 0,
            "decoded_frames": reader.decoded_frames if reader else 0,
            "error": "%s: %s" % (type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
        }


def _safe_output_root(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    if resolved == Path(resolved.anchor) or resolved == Path.home().resolve():
        raise ValueError("Refusing to use a filesystem root or home directory as output_dir")


def _prepare_output(output_dir: Path, overwrite: bool) -> None:
    _safe_output_root(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    managed = [output_dir / name for name in ("train", "val", "test", "logs")]
    managed.append(output_dir / "dataset_summary.json")
    existing = [path for path in managed if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output already contains generated data; use --overwrite to replace it: %s" % ", ".join(str(path) for path in existing)
        )
    if overwrite:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(str(path))
            else:
                path.unlink()
    for name in ("train", "val", "test", "logs"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")


def _finalize_clips(output_dir: Path, results: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counters = {"train": 0, "val": 0, "test": 0}
    successful = sorted(
        [result for result in results if result.get("ok")],
        key=lambda result: (str(result["split"]), str(result["source_video"])),
    )
    for result in successful:
        split = str(result["split"])
        for staged_value in sorted(result["clips"]):
            staged = Path(staged_value)
            clip_id = "clip_%06d" % counters[split]
            metadata = read_json(staged / "meta.json")
            metadata["clip_id"] = clip_id
            write_json(staged / "meta.json", metadata)
            destination = output_dir / split / clip_id
            if destination.exists():
                raise FileExistsError("Final clip already exists: %s" % destination)
            os.replace(str(staged), str(destination))
            counters[split] += 1
    return counters


def generate_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    validate_config(config)
    input_dir = Path(config["input_dir"]).expanduser().resolve()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    if input_dir == output_dir:
        raise ValueError("input_dir and output_dir must be different")
    if not input_dir.is_dir():
        raise FileNotFoundError("Input directory does not exist: %s" % input_dir)
    if output_dir in input_dir.parents:
        raise ValueError("input_dir cannot be nested inside output_dir")
    for binary_key in ("ffmpeg_bin", "ffprobe_bin"):
        binary = str(config[binary_key])
        if shutil.which(binary) is None and not Path(binary).is_file():
            raise FileNotFoundError("Required executable not found: %s" % binary)

    _prepare_output(output_dir, bool(config.get("overwrite", False)))
    video_cfg = config["video"]
    candidates = discover_candidates(
        input_dir,
        video_cfg.get("extensions", []),
        bool(video_cfg.get("scan_all_files", True)),
        excluded_dir=output_dir,
        raw_yuv_sidecar_suffix=str(video_cfg.get("raw_yuv", {}).get("sidecar_suffix", ".json")),
        excluded_dir_names=video_cfg.get("exclude_dir_names", []),
    )
    probe_failures: List[Dict[str, Any]] = []
    probed: List[Tuple[str, Path, VideoInfo]] = []
    for path in candidates:
        source_video = path.relative_to(input_dir).as_posix()
        try:
            info = probe_video(
                path,
                str(config["ffprobe_bin"]),
                raw_yuv_config=video_cfg.get("raw_yuv", {}),
                source_key=source_video,
            )
            probed.append((source_video, path, info))
        except Exception as exc:
            probe_failures.append({"stage": "probe", "source_video": source_video, "error": str(exc)})

    assignments = assign_splits([item[0] for item in probed], config["split"], int(config["seed"]))
    stage_root = Path(tempfile.mkdtemp(prefix=".staging-", dir=str(output_dir)))
    tasks: List[Dict[str, Any]] = []
    for source_video, source_path, info in probed:
        tasks.append({
            "source_path": str(source_path),
            "source_video": source_video,
            "split": assignments[source_video],
            "config": config,
            "video_seed": derive_seed(int(config["seed"]), source_video),
            "stage_root": str(stage_root),
            "video_key": "%08x" % derive_seed(0, source_video),
            "video_info": info.to_dict(),
        })

    results: List[Dict[str, Any]] = []
    try:
        workers = min(int(config.get("num_workers", 1)), max(1, len(tasks)))
        if workers == 1:
            results = [_process_video(task) for task in tasks]
        elif tasks:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_task = {executor.submit(_process_video, task): task for task in tasks}
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        results.append({
                            "ok": False,
                            "source_video": task["source_video"],
                            "split": task["split"],
                            "clips": [],
                            "num_clips": 0,
                            "decoded_frames": 0,
                            "error": "Worker process failed: %s" % exc,
                        })

        clip_counts = _finalize_clips(output_dir, results)
    finally:
        shutil.rmtree(str(stage_root), ignore_errors=True)

    successful = [result for result in results if result.get("ok")]
    process_failures = [result for result in results if not result.get("ok")]
    success_records = [{key: value for key, value in result.items() if key not in {"clips", "traceback"}} for result in successful]
    failure_records: List[Mapping[str, Any]] = list(probe_failures)
    failure_records.extend({key: value for key, value in result.items() if key != "clips"} for result in process_failures)
    _write_jsonl(output_dir / "logs" / "success.log", success_records)
    _write_jsonl(output_dir / "logs" / "failed.log", failure_records)

    validation = validate_dataset(
        output_dir,
        expected_size=None if config["image"].get("preserve_source_resolution", False) else output_size(config["image"]),
        expected_lq_size=lq_output_size(config["image"]),
        expected_num_frames=int(config["clip"]["num_frames"]),
    )
    summary = {
        "num_source_videos": len(candidates),
        "num_valid_source_videos": len(probed),
        "num_successful_source_videos": len(successful),
        "num_failed_videos": len(probe_failures) + len(process_failures),
        "num_short_videos": sum(1 for result in successful if int(result["num_clips"]) == 0),
        "num_partial_decode_videos": sum(1 for result in successful if result.get("warning")),
        "num_train_clips": clip_counts["train"],
        "num_val_clips": clip_counts["val"],
        "num_test_clips": clip_counts["test"],
        "split_source_videos": {
            split: sorted(source for source, assigned in assignments.items() if assigned == split)
            for split in ("train", "val", "test")
        },
        "seed": int(config["seed"]),
        "validation": validation,
    }
    write_json(output_dir / "dataset_summary.json", summary)
    if not validation["valid"]:
        raise RuntimeError(
            "Dataset generation finished but validation found %d error(s); see dataset_summary.json" % validation["num_errors"]
        )
    return summary
