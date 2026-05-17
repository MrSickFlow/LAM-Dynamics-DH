from __future__ import annotations

from ipb_backend.planning.force_model import Operation, OperationType, Priority


SCORING_CRITERIA = (
    "terrain_fit",
    "concealment",
    "drone_cover",
    "observation",
    "route_resilience",
    "civilian_avoidance",
    "comms_coverage",
    "logistics_proximity",
)


OPERATION_PROFILES: dict[OperationType, dict[str, float]] = {
    OperationType.DEFENSIVE: {
        "terrain_fit": 0.28,
        "concealment": 0.18,
        "drone_cover": 0.17,
        "observation": 0.12,
        "route_resilience": 0.09,
        "civilian_avoidance": 0.06,
        "comms_coverage": 0.05,
        "logistics_proximity": 0.05,
    },
    OperationType.OFFENSIVE: {
        "terrain_fit": 0.24,
        "concealment": 0.10,
        "drone_cover": 0.08,
        "observation": 0.08,
        "route_resilience": 0.27,
        "civilian_avoidance": 0.05,
        "comms_coverage": 0.06,
        "logistics_proximity": 0.12,
    },
    OperationType.RECON: {
        "terrain_fit": 0.22,
        "concealment": 0.18,
        "drone_cover": 0.18,
        "observation": 0.14,
        "route_resilience": 0.08,
        "civilian_avoidance": 0.07,
        "comms_coverage": 0.08,
        "logistics_proximity": 0.05,
    },
    OperationType.SCREEN: {
        "terrain_fit": 0.22,
        "concealment": 0.16,
        "drone_cover": 0.15,
        "observation": 0.14,
        "route_resilience": 0.12,
        "civilian_avoidance": 0.06,
        "comms_coverage": 0.08,
        "logistics_proximity": 0.07,
    },
    OperationType.LOGISTICS_HUB: {
        "terrain_fit": 0.20,
        "concealment": 0.06,
        "drone_cover": 0.05,
        "observation": 0.04,
        "route_resilience": 0.32,
        "civilian_avoidance": 0.08,
        "comms_coverage": 0.10,
        "logistics_proximity": 0.15,
    },
    OperationType.WITHDRAWAL: {
        "terrain_fit": 0.18,
        "concealment": 0.12,
        "drone_cover": 0.10,
        "observation": 0.08,
        "route_resilience": 0.28,
        "civilian_avoidance": 0.07,
        "comms_coverage": 0.07,
        "logistics_proximity": 0.10,
    },
    OperationType.FIRE_SUPPORT: {
        "terrain_fit": 0.24,
        "concealment": 0.16,
        "drone_cover": 0.14,
        "observation": 0.12,
        "route_resilience": 0.15,
        "civilian_avoidance": 0.06,
        "comms_coverage": 0.06,
        "logistics_proximity": 0.07,
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
    base["drone_cover"] *= PRIORITY_NUDGE[operation.concealment_priority]
    base["route_resilience"] *= PRIORITY_NUDGE[operation.speed_priority]
    base["logistics_proximity"] *= PRIORITY_NUDGE[operation.speed_priority]
    base["civilian_avoidance"] *= PRIORITY_NUDGE[operation.civilian_avoidance]
    base["comms_coverage"] *= PRIORITY_NUDGE[operation.comms_priority]

    total = sum(base.values())
    if total <= 0:
        return base
    return {key: value / total for key, value in base.items()}
