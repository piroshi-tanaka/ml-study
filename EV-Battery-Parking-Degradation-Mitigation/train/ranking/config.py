"""Configuration module for ranking training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class RankingConfig:
    """Collection of parameters used for the ranking pipeline."""

    # Recent history weighting
    window_days: int = 90
    halflife_days: int = 30
    use_decay_weight: bool = True

    # Laplace smoothing
    alpha_smooth: float = 0.5

    # Candidate generation thresholds
    k_candidates: int = 12
    m_routine: int = 8
    n_charge_prior: int = 8
    l_nearby: int = 4
    nearby_radius_km: float = 1.0

    # Candidate scoring weights
    lambda_start: float = 0.7
    w_routine: float = 1.0
    w_charge: float = 1.0
    gamma_distance: float = 0.05
    kernel_sigma_hour: float = 2.0

    # Evaluation settings
    topk_eval: List[int] = field(default_factory=lambda: [1, 3, 5])

    # AutoGluon training parameters
    random_seed: int = 42
    time_limit: int = 300
    ag_presets: str = "medium_quality_faster_train"

    # Directory where evaluation artifacts are persisted. None disables saving.
    result_root: Optional[Path] = Path("result")


# Shared default configuration for quick use from notebooks
DEFAULT_CONFIG = RankingConfig()
