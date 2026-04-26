from __future__ import annotations

from typing import Dict

from group3_bayesian.cpt import (
    PRIOR_HEALTH,
    PRIOR_RISK,
    P_CAMERA_GIVEN_RISK_HEALTH,
    P_PEOPLE_GIVEN_DETECTION,
)
from shared.utils import normalize_probabilities


def infer_risk(camera_reading: str) -> Dict[str, float]:
    """
    Infer P(Risk | CameraReading) using:
    P(R | C) proportional to sum_H P(C | R, H) * P(H) * P(R)
    """
    unnormalized = {}

    for risk_state, risk_prior in PRIOR_RISK.items():
        total = 0.0
        for health_state, health_prior in PRIOR_HEALTH.items():
            likelihood = P_CAMERA_GIVEN_RISK_HEALTH[(risk_state, health_state)][camera_reading]
            total += likelihood * health_prior * risk_prior
        unnormalized[risk_state] = total

    return normalize_probabilities(unnormalized)


def infer_people(detection: str) -> Dict[str, float]:
    #Return P(PeoplePresence | Detection)
    if detection not in P_PEOPLE_GIVEN_DETECTION:
        raise ValueError(f"Unknown detection state: {detection}")
    return normalize_probabilities(P_PEOPLE_GIVEN_DETECTION[detection])
