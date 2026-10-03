#!/usr/bin/env python3
"""Compatibility entry point for generating a video training dataset."""

from dataset.cli import generate_main
from dataset.generator import generate_dataset
from dataset.splitting import assign_splits

# Preserve imports used by earlier callers while the public API lives in dataset.
_split_assignments = assign_splits
main = generate_main

__all__ = ["generate_dataset", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
