#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from dataset.config import load_config, lq_output_size, output_size
from dataset.validator import validate_dataset


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a generated GT/REF/LQ video dataset")
    parser.add_argument("--dataset", required=True, type=Path, help="Dataset root containing train/val/test")
    parser.add_argument("--config", type=Path, help="Optional generation config for expected resolution")
    args = parser.parse_args(argv)
    config = load_config(args.config) if args.config else None
    expected = (
        None
        if not config or config["image"].get("preserve_source_resolution", False)
        else output_size(config["image"])
    )
    expected_lq = lq_output_size(config["image"]) if config else None
    expected_num_frames = int(config["clip"]["num_frames"]) if config else None
    report = validate_dataset(
        args.dataset.resolve(),
        expected_size=expected,
        expected_lq_size=expected_lq,
        expected_num_frames=expected_num_frames,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
