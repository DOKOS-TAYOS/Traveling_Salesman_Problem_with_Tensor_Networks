from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TauSchedule:
    by_size: dict[int, float]


def tau_for_size(
    n: int,
    default_tau: float,
    schedule: TauSchedule | None,
) -> float:
    """Return the explicitly calibrated tau for n, or the default tau."""
    if schedule is None:
        return default_tau
    exact_tau = schedule.by_size.get(int(n))
    if exact_tau is not None:
        return exact_tau
    return default_tau


def load_tau_schedule(path: Path | None) -> TauSchedule | None:
    """Load selected tau values from a tau-by-size CSV file."""
    if path is None:
        return None

    by_size: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n = _optional_int(row.get("n"))
            tau = _optional_float(row.get("selected_tau"))
            if n is not None and tau is not None:
                by_size[n] = tau

    return TauSchedule(by_size=by_size)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    return value_float if math.isfinite(value_float) else None


def _optional_int(value: Any) -> int | None:
    value_float = _optional_float(value)
    if value_float is None:
        return None
    return int(value_float)
