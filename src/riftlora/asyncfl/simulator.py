from __future__ import annotations

from dataclasses import dataclass
import heapq
import random
from collections import Counter


@dataclass(frozen=True)
class ClientProfile:
    client_id: str
    rank: int
    num_samples: int
    compute_time: float
    network_time: float = 0.0
    jitter: float = 0.0


@dataclass(frozen=True)
class _ReturnEvent:
    finish_time: float
    sequence_id: int
    client: ClientProfile
    base_version: int
    dispatch_time: float


@dataclass(frozen=True)
class ReturnRecord:
    client_id: str
    rank: int
    num_samples: int
    dispatch_time: float
    finish_time: float
    base_version: int
    arrival_version: int
    staleness: int
    aggregated: bool
    new_server_version: int


@dataclass(frozen=True)
class SimulationTrace:
    records: tuple[ReturnRecord, ...]

    @property
    def return_order(self) -> tuple[str, ...]:
        return tuple(record.client_id for record in self.records)

    @property
    def dispatch_versions(self) -> tuple[int, ...]:
        return tuple(record.base_version for record in self.records)

    @property
    def arrival_versions(self) -> tuple[int, ...]:
        return tuple(record.arrival_version for record in self.records)

    @property
    def staleness_values(self) -> tuple[int, ...]:
        return tuple(record.staleness for record in self.records)

    def staleness_histogram(self) -> dict[int, int]:
        return dict(Counter(self.staleness_values))


class AsyncEventSimulator:
    """Small deterministic simulator for version staleness and return ordering."""

    def __init__(
        self,
        clients: list[ClientProfile],
        *,
        seed: int = 0,
        buffer_size: int = 1,
    ) -> None:
        if not clients:
            raise ValueError("at least one client is required")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if any(client.rank <= 0 for client in clients):
            raise ValueError("client ranks must be positive")
        if any(client.num_samples <= 0 for client in clients):
            raise ValueError("client num_samples must be positive")

        self.clients = tuple(clients)
        self.seed = seed
        self.buffer_size = buffer_size

    def run(self, *, max_returns: int) -> SimulationTrace:
        if max_returns <= 0:
            raise ValueError("max_returns must be positive")

        rng = random.Random(self.seed)
        queue: list[tuple[float, int, _ReturnEvent]] = []
        server_version = 0
        next_sequence = 0
        buffered_returns = 0

        for client in self.clients:
            event = self._dispatch(
                client=client,
                now=0.0,
                base_version=server_version,
                sequence_id=next_sequence,
                rng=rng,
            )
            heapq.heappush(queue, (event.finish_time, event.sequence_id, event))
            next_sequence += 1

        records: list[ReturnRecord] = []
        while len(records) < max_returns:
            _, _, event = heapq.heappop(queue)
            arrival_version = server_version
            staleness = arrival_version - event.base_version

            buffered_returns += 1
            aggregated = buffered_returns >= self.buffer_size
            if aggregated:
                server_version += 1
                buffered_returns = 0

            records.append(
                ReturnRecord(
                    client_id=event.client.client_id,
                    rank=event.client.rank,
                    num_samples=event.client.num_samples,
                    dispatch_time=event.dispatch_time,
                    finish_time=event.finish_time,
                    base_version=event.base_version,
                    arrival_version=arrival_version,
                    staleness=staleness,
                    aggregated=aggregated,
                    new_server_version=server_version,
                )
            )

            next_event = self._dispatch(
                client=event.client,
                now=event.finish_time,
                base_version=server_version,
                sequence_id=next_sequence,
                rng=rng,
            )
            heapq.heappush(queue, (next_event.finish_time, next_event.sequence_id, next_event))
            next_sequence += 1

        return SimulationTrace(tuple(records))

    @staticmethod
    def _dispatch(
        *,
        client: ClientProfile,
        now: float,
        base_version: int,
        sequence_id: int,
        rng: random.Random,
    ) -> _ReturnEvent:
        if client.compute_time <= 0:
            raise ValueError("client compute_time must be positive")
        if client.network_time < 0 or client.jitter < 0:
            raise ValueError("network_time and jitter must be non-negative")

        delay = client.compute_time + client.network_time
        if client.jitter:
            delay += rng.uniform(0.0, client.jitter)

        return _ReturnEvent(
            finish_time=now + delay,
            sequence_id=sequence_id,
            client=client,
            base_version=base_version,
            dispatch_time=now,
        )

