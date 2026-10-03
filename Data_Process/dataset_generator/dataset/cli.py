"""Provide command-line interfaces for generation, regeneration, and validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .config import load_config, lq_output_size, output_size, validate_config
from .generator import generate_dataset
from .regenerator import regenerate_lq
from .validator import validate_dataset


def _generation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate aligned GT/REF/LQ clips from source videos"
    )
    parser.add_argument("--config", type=Path, help="YAML configuration file")
    parser.add_argument("--input", dest="input_dir", help="Override input_dir")
    parser.add_argument("--output", dest="output_dir", help="Override output_dir")
    parser.add_argument("--num-frames", type=int, help="Override clip.num_frames")
    parser.add_argument("--frame-interval", type=int, help="Override clip.frame_interval")
    parser.add_argument("--clip-stride", type=int, help="Override clip.clip_stride")
    parser.add_argument("--num-workers", type=int, help="Override num_workers")
    parser.add_argument("--seed", type=int, help="Override seed")
    parser.add_argument("--yuv-width", type=int, help="Default width for raw .yuv inputs")
    parser.add_argument("--yuv-height", type=int, help="Default height for raw .yuv inputs")
    parser.add_argument("--yuv-pix-fmt", help="Default FFmpeg pixel format for raw .yuv inputs")
    parser.add_argument("--yuv-fps", type=float, help="Default FPS for raw .yuv inputs")
    parser.add_argument(
        "--yuv-rotation",
        type=int,
        choices=(0, 90, 180, 270),
        help="Default rotation for raw .yuv inputs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace train/val/test/logs in output_dir",
    )
    drop_group = parser.add_mutually_exclusive_group()
    drop_group.add_argument(
        "--drop-last",
        dest="drop_last",
        action="store_true",
        help="Drop an incomplete tail clip",
    )
    drop_group.add_argument(
        "--keep-last",
        dest="drop_last",
        action="store_false",
        help="Add a tail-aligned full clip",
    )
    parser.set_defaults(drop_last=None)
    return parser


def _apply_generation_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    for key in ("input_dir", "output_dir", "num_workers", "seed"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    for key in ("num_frames", "frame_interval", "clip_stride"):
        value = getattr(args, key)
        if value is not None:
            config["clip"][key] = value
    if args.drop_last is not None:
        config["clip"]["drop_last"] = args.drop_last
    if args.overwrite:
        config["overwrite"] = True

    raw_yuv = config["video"]["raw_yuv"]
    for argument, key in (
        ("yuv_width", "width"),
        ("yuv_height", "height"),
        ("yuv_pix_fmt", "pix_fmt"),
        ("yuv_fps", "fps"),
        ("yuv_rotation", "rotation"),
    ):
        value = getattr(args, argument)
        if value is not None:
            raw_yuv[key] = value


def generate_main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the dataset generation command."""
    args = _generation_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        _apply_generation_overrides(config, args)
        validate_config(config)
        summary = generate_dataset(config)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1


def regenerate_main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the atomic LQ regeneration command."""
    parser = argparse.ArgumentParser(
        description="Atomically regenerate LQ frames from an existing dataset's GT"
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        report = regenerate_lq(args.dataset, config)
        print(
            "Regenerated LQ for %d clips; validation errors: %d"
            % (sum(report["clip_counts"].values()), report["num_errors"])
        )
        return 0
    except Exception as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1


def validate_main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the standalone dataset validation command."""
    parser = argparse.ArgumentParser(
        description="Validate a generated GT/REF/LQ video dataset"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Dataset root containing train/val/test",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional generation config for expected resolution",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config) if args.config else None
    expected_size = (
        None
        if not config or config["image"].get("preserve_source_resolution", False)
        else output_size(config["image"])
    )
    report = validate_dataset(
        args.dataset.resolve(),
        expected_size=expected_size,
        expected_lq_size=lq_output_size(config["image"]) if config else None,
        expected_num_frames=int(config["clip"]["num_frames"]) if config else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1
