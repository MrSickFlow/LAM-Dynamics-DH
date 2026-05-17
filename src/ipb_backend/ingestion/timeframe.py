"""Timeframe parsing utilities for IPB ingestion sources.

Supported formats
-----------------
* ``"24h"`` / ``"72h"``      — relative: past N hours ending now
* ``"7d"`` / ``"3d"``        — relative: past N days ending now
* ``"2024-05-15/2024-05-20"`` — absolute ISO date range (inclusive)
* ``"2024-05-15T06:00Z/2024-05-15T18:00Z"`` — absolute ISO datetime range

When a relative format is used, *end* is the current UTC hour (seconds truncated)
and *start* is calculated backward by the given duration.

When used for a *forecast* context (e.g. satellite passes, weather forecast),
callers should use *end* as the forward planning horizon and pass ``forward=True``
to ``parse_timeframe`` so that the window extends *ahead* of now instead.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_timeframe(timeframe: str, *, forward: bool = False) -> tuple[datetime, datetime]:
    """Return ``(start, end)`` as UTC datetimes for the given *timeframe* string.

    Parameters
    ----------
    timeframe:
        One of the formats described in the module docstring.
    forward:
        When ``True`` and a *relative* format is given, the window extends
        *forward* from now (``now`` → ``now + duration``) instead of backward.
        Useful for forecast and satellite pass windows.
    """
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    # Absolute ISO interval: "START/END"
    if "/" in timeframe:
        start_str, _, end_str = timeframe.partition("/")
        start = _parse_iso_dt(start_str.strip())
        end = _parse_iso_dt(end_str.strip())
        if start is not None and end is not None:
            if start > end:
                start, end = end, start
            return start, end

    # Relative hours: "24h", "72H"
    m = re.fullmatch(r"\s*(\d+)\s*h\s*", timeframe, re.IGNORECASE)
    if m:
        delta = timedelta(hours=int(m.group(1)))
        return (now, now + delta) if forward else (now - delta, now)

    # Relative days: "7d", "3D"
    m = re.fullmatch(r"\s*(\d+)\s*d\s*", timeframe, re.IGNORECASE)
    if m:
        delta = timedelta(days=int(m.group(1)))
        return (now, now + delta) if forward else (now - delta, now)

    # Fallback: past 24 hours
    return (now, now + timedelta(hours=24)) if forward else (now - timedelta(hours=24), now)


def timeframe_hours(timeframe: str) -> float:
    """Return the total span of the timeframe in hours (convenience helper)."""
    start, end = parse_timeframe(timeframe)
    return (end - start).total_seconds() / 3600.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ISO_FORMATS = (
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%MZ",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d",
)


def _parse_iso_dt(s: str) -> Optional[datetime]:
    for fmt in _ISO_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None
