from __future__ import annotations

from abc import ABC, abstractmethod

from ipb_backend.models import DatasetRecord, LoadTarget, SourceDefinition


class SourceAdapter(ABC):
    def __init__(self, definition: SourceDefinition) -> None:
        self.definition = definition

    @abstractmethod
    async def fetch(self, area: str, timeframe: str, load_target: LoadTarget | None = None) -> DatasetRecord:
        """Fetch and normalize the latest record for a given area and timeframe."""
