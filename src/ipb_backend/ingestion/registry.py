from __future__ import annotations

from ipb_backend.models import SourceDefinition


class SourceRegistry:
    def __init__(self, definitions: list[SourceDefinition]) -> None:
        self._definitions = {definition.source_id: definition for definition in definitions}

    def list_sources(self) -> list[SourceDefinition]:
        return list(self._definitions.values())

    def get(self, source_id: str) -> SourceDefinition:
        return self._definitions[source_id]

    def missing_source_ids(self, source_ids: list[str]) -> list[str]:
        return [source_id for source_id in source_ids if source_id not in self._definitions]

    def enabled_source_ids(self) -> list[str]:
        return [source_id for source_id, definition in self._definitions.items() if definition.enabled]

    def update(self, definition: SourceDefinition) -> None:
        self._definitions[definition.source_id] = definition
