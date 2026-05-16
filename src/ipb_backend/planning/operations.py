from __future__ import annotations

from ipb_backend.planning.force_model import Operation, OperationType, Priority


SCORING_CRITERIA = (
    "concealment",
    "observation",
    "road_access",
    "civilian_avoidance",
    "comms_coverage",
    "logistics_proximity",
)


OPERATION_PROFILES: dict[OperationType, dict[str, float]] = {
    OperationType.DEFENSIVE: {
        "concealment": 0.30,
        "observation": 0.25,
        "road_access": 0.10,
        "civilian_avoidance": 0.15,
        "comms_coverage": 0.10,
        "logistics_proximity": 0.10,
    },
    OperationType.OFFENSIVE: {
        "concealment": 0.15,
        "observation": 0.15,
        "road_access": 0.30,
        "civilian_avoidance": 0.10,
        "comms_coverage": 0.15,
        "logistics_proximity": 0.15,
    },
    OperationType.RECON: {
        "concealment": 0.35,
        "observation": 0.30,
        "road_access": 0.05,
        "civilian_avoidance": 0.15,
        "comms_coverage": 0.10,
        "logistics_proximity": 0.05,
    },
    OperationType.SCREEN: {
        "concealment": 0.25,
        "observation": 0.30,
        "road_access": 0.15,
        "civilian_avoidance": 0.10,
        "comms_coverage": 0.15,
        "logistics_proximity": 0.05,
    },
    OperationType.LOGISTICS_HUB: {
        "concealment": 0.15,
        "observation": 0.05,
        "road_access": 0.35,
        "civilian_avoidance": 0.10,
        "comms_coverage": 0.10,
        "logistics_proximity": 0.25,
    },
    OperationType.WITHDRAWAL: {
        "concealment": 0.25,
        "observation": 0.10,
        "road_access": 0.35,
        "civilian_avoidance": 0.10,
        "comms_coverage": 0.10,
        "logistics_proximity": 0.10,
    },
}


PRIORITY_NUDGE = {
    Priority.LOW: 0.8,
    Priority.MEDIUM: 1.0,
    Priority.HIGH: 1.25,
}


def get_operation_profile(operation: Operation) -> dict[str, float]:
    base = dict(OPERATION_PROFILES[operation.type])

    base["concealment"] *= PRIORITY_NUDGE[operation.concealment_priority]
    base["road_access"] *= PRIORITY_NUDGE[operation.speed_priority]
    base["civilian_avoidance"] *= PRIORITY_NUDGE[operation.civilian_avoidance]
    base["comms_coverage"] *= PRIORITY_NUDGE[operation.comms_priority]

    total = sum(base.values())
    if total <= 0:
        return base
    return {key: value / total for key, value in base.items()}
