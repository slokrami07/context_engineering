"""Generic parser protocol and base classes for conversation export adapters."""

from typing import Any, Protocol
from context_engineer.types import IngestionPayload, Turn


class BaseAdapter(Protocol):
    """Protocol defining the adapter interface for ingesting external chat exports."""

    def parse(self, raw_data: Any) -> IngestionPayload:
        """Parses external export data structure into a normalized IngestionPayload."""
        ...

    def export(self, payload: IngestionPayload) -> Any:
        """Exports an IngestionPayload back to the external format."""
        ...
