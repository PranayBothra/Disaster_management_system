#Conditional probability tables for the Bayesian model.

PRIOR_RISK = {
    "High": 0.35,
    "Low": 0.65,
}

PRIOR_HEALTH = {
    "Working": 0.85,
    "Faulty": 0.15,
}

P_CAMERA_GIVEN_RISK_HEALTH = {
    ("High", "Working"): {"High": 0.88, "Low": 0.12},
    ("High", "Faulty"): {"High": 0.60, "Low": 0.40},
    ("Low", "Working"): {"High": 0.20, "Low": 0.80},
    ("Low", "Faulty"): {"High": 0.35, "Low": 0.65},
}

P_PEOPLE_GIVEN_DETECTION = {
    "Detected": {"Present": 0.90, "Absent": 0.10},
    "NotDetected": {"Present": 0.25, "Absent": 0.75},
}
