"""Derive stable random seeds and initialize supported random generators."""

from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np


def derive_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join([str(int(base_seed))] + [str(part) for part in parts])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2 ** 32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
