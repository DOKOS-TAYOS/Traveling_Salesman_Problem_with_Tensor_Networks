from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.config import DEFAULT_TAU_SOURCE


@dataclass(frozen=True)
class TauScheduleEntry:
    selected_tau: float
    tau_source: str
    instance_type: str | None = None
    n: int | None = None


@dataclass(frozen=True)
class TauSchedule:
    by_size: dict[int, float]
    by_instance_type_and_size: dict[tuple[str, int], TauScheduleEntry] = field(
        default_factory=dict
    )
    source_by_size: dict[int, str] = field(default_factory=dict)
    calibration_seeds: set[int] = field(default_factory=set)


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


def tau_for_instance(
    instance_type: str,
    n: int,
    default_tau: float,
    schedule: TauSchedule | None,
) -> float:
    """Return calibrated tau for an instance type and size when available."""
    entry = tau_schedule_entry_for_instance(instance_type, n, schedule)
    if entry is not None:
        return entry.selected_tau
    return tau_for_size(n, default_tau, schedule)


def tau_source_for_instance(
    instance_type: str,
    n: int,
    schedule: TauSchedule | None,
) -> str:
    """Return the source label for the selected tau value."""
    entry = tau_schedule_entry_for_instance(instance_type, n, schedule)
    if entry is not None:
        return entry.tau_source
    if schedule is not None and int(n) in schedule.by_size:
        return schedule.source_by_size.get(int(n), "calibrated_by_size_legacy")
    return DEFAULT_TAU_SOURCE


def tau_schedule_entry_for_instance(
    instance_type: str,
    n: int,
    schedule: TauSchedule | None,
) -> TauScheduleEntry | None:
    """Return schedule entry for exact instance type/size, if present."""
    if schedule is None:
        return None
    return schedule.by_instance_type_and_size.get((instance_type, int(n)))


def load_tau_schedule(path: Path | None) -> TauSchedule | None:
    """Load selected tau values from a tau-by-size CSV file."""
    if path is None:
        return None

    by_size: dict[int, float] = {}
    by_instance_type_and_size: dict[tuple[str, int], TauScheduleEntry] = {}
    source_by_size: dict[int, str] = {}
    calibration_seeds: set[int] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n = _optional_int(row.get("n"))
            tau = _optional_float(row.get("selected_tau"))
            if n is not None and tau is not None:
                by_size[n] = tau
                instance_type = _optional_str(row.get("instance_type"))
                tau_source = _optional_str(row.get("tau_source")) or DEFAULT_TAU_SOURCE
                source_by_size[n] = tau_source
                if instance_type is not None:
                    by_instance_type_and_size[(instance_type, n)] = TauScheduleEntry(
                        selected_tau=tau,
                        tau_source=tau_source,
                        instance_type=instance_type,
                        n=n,
                    )
                calibration_seeds.update(_parse_seed_list(row.get("calibration_seeds")))

    return TauSchedule(
        by_size=by_size,
        by_instance_type_and_size=by_instance_type_and_size,
        source_by_size=source_by_size,
        calibration_seeds=calibration_seeds,
    )


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


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _parse_seed_list(value: Any) -> set[int]:
    if value in (None, ""):
        return set()
    text = str(value)
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = [part.strip() for part in text.split(",") if part.strip()]
    if not isinstance(decoded, list):
        return set()
    seeds: set[int] = set()
    for seed in decoded:
        try:
            seeds.add(int(seed))
        except (TypeError, ValueError):
            continue
    return seeds
