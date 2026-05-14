from __future__ import annotations

from collections.abc import Iterable

DEFAULT_CALIBRATION_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_EVALUATION_SEEDS = [5, 6, 7, 8, 9]

CALIBRATION_SPLIT = "calibration"
EVALUATION_SPLIT = "evaluation"
CALIBRATED_TAU_SOURCE = "calibrated_on_calibration_split"
DEFAULT_TAU_SOURCE = "manual_default_tau"


def validate_disjoint_splits(
    calibration_seeds: Iterable[int],
    evaluation_seeds: Iterable[int],
) -> None:
    """Raise ValueError when calibration and evaluation seeds overlap."""
    calibration = {int(seed) for seed in calibration_seeds}
    evaluation = {int(seed) for seed in evaluation_seeds}
    overlap = sorted(calibration & evaluation)
    if overlap:
        raise ValueError(
            "Calibration and evaluation seed splits overlap: "
            + ", ".join(str(seed) for seed in overlap)
        )
