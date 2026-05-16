from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Optional

from ipb_backend.config import settings
from ipb_backend.models import IngestionRequest

logger = logging.getLogger(__name__)


class RefreshScheduler:
    def __init__(self, ingestion_service) -> None:
        self._ingestion_service = ingestion_service
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if not settings.auto_refresh or self._task is not None:
            return
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _run_loop(self) -> None:
        while True:
            try:
                await self._ingestion_service.ingest(
                    IngestionRequest(area=settings.default_area, timeframe=settings.default_timeframe)
                )
            except Exception:
                logger.exception("Scheduled refresh failed")
            await asyncio.sleep(settings.refresh_interval_seconds)
