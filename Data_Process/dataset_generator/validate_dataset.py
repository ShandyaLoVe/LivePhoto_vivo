#!/usr/bin/env python3
"""Compatibility entry point for validating a generated dataset."""

from dataset.cli import validate_main
from dataset.validator import validate_dataset

main = validate_main

__all__ = ["validate_dataset", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
