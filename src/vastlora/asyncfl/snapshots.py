from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class VersionedSnapshot(Generic[T]):
    version: int
    payload: T


class SnapshotStore(Generic[T]):
    """Small in-memory versioned snapshot store for simulator prototypes."""

    def __init__(self) -> None:
        self._snapshots: dict[int, T] = {}

    def put(self, version: int, payload: T) -> VersionedSnapshot[T]:
        if version < 0:
            raise ValueError("version must be non-negative")
        if version in self._snapshots:
            raise ValueError(f"version {version} already exists")
        self._snapshots[version] = payload
        return VersionedSnapshot(version=version, payload=payload)

    def get(self, version: int) -> VersionedSnapshot[T]:
        try:
            payload = self._snapshots[version]
        except KeyError as exc:
            raise KeyError(f"version {version} is not available") from exc
        return VersionedSnapshot(version=version, payload=payload)

    def latest(self) -> VersionedSnapshot[T]:
        if not self._snapshots:
            raise ValueError("snapshot store is empty")
        version = max(self._snapshots)
        return VersionedSnapshot(version=version, payload=self._snapshots[version])

    def __contains__(self, version: object) -> bool:
        return version in self._snapshots

    def __len__(self) -> int:
        return len(self._snapshots)

