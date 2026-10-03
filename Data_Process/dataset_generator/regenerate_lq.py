#!/usr/bin/env python3
"""Compatibility entry point for rebuilding LQ frames from existing GT frames."""

from dataset.cli import regenerate_main
from dataset.regenerator import regenerate_lq

main = regenerate_main

__all__ = ["regenerate_lq", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
