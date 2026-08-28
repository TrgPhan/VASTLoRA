from __future__ import annotations

from dataclasses import dataclass
import heapq
import random
from collections import Counter
from typing import Literal


ScheduleMode = Literal["async", "cohort"]


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
    group_id: int = 0
    group_version: int = 0
    group_position: int = 1
    buffer_size: int = 1
    group_closed: bool = True


@dataclass(frozen=True)
class UpdateGroupRecord:
    group_id: int
    group_version: int
    server_version_after: int
    first_arrival_time: float
    aggregation_time: float
    closed: bool
    records: tuple[ReturnRecord, ...]

    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def client_ids(self) -> tuple[str, ...]:
        return tuple(record.client_id for record in self.records)

    @property
    def staleness_values(self) -> tuple[int, ...]:
        return tuple(record.staleness for record in self.records)


@dataclass(frozen=True)
class SimulationTrace:
    records: tuple[ReturnRecord, ...]
    groups: tuple[UpdateGroupRecord, ...] = ()

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

    @property
    def group_ids(self) -> tuple[int, ...]:
        return tuple(record.group_id for record in self.records)

    @property
    def group_versions(self) -> tuple[int, ...]:
        return tuple(record.group_version for record in self.records)

    @property
    def completed_groups(self) -> tuple[UpdateGroupRecord, ...]:
        return tuple(group for group in self.groups if group.closed)

    @property
    def pending_groups(self) -> tuple[UpdateGroupRecord, ...]:
        return tuple(group for group in self.groups if not group.closed)

    def staleness_histogram(self) -> dict[int, int]:
        return dict(Counter(self.staleness_values))

    def groups_by_version(self) -> dict[int, tuple[UpdateGroupRecord, ...]]:
        grouped: dict[int, list[UpdateGroupRecord]] = {}
        for group in self.groups:
            grouped.setdefault(group.group_version, []).append(group)
        return {version: tuple(groups) for version, groups in grouped.items()}


class AsyncEventSimulator:
    """Small deterministic simulator for version staleness and return ordering."""

    def __init__(
        self,
        clients: list[ClientProfile],
        *,
        seed: int = 0,
        buffer_size: int = 1,
        schedule_mode: ScheduleMode = "async",
    ) -> None:
        if not clients:
            raise ValueError("at least one client is required")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if schedule_mode not in {"async", "cohort"}:
            raise ValueError("schedule_mode must be 'async' or 'cohort'")
        if schedule_mode == "cohort" and buffer_size > len(clients):
            raise ValueError("cohort schedule requires buffer_size <= number of clients")
        if any(client.rank <= 0 for client in clients):
            raise ValueError("client ranks must be positive")
        if any(client.num_samples <= 0 for client in clients):
            raise ValueError("client num_samples must be positive")

        self.clients = tuple(clients)
        self.seed = seed
        self.buffer_size = buffer_size
        self.schedule_mode = schedule_mode

    def run(self, *, max_returns: int) -> SimulationTrace:
        if max_returns <= 0:
            raise ValueError("max_returns must be positive")
        if self.schedule_mode == "cohort":
            return self._run_cohort(max_returns=max_returns)
        return self._run_async(max_returns=max_returns)

    def _run_async(self, *, max_returns: int) -> SimulationTrace:
        rng = random.Random(self.seed)
        queue: list[tuple[float, int, _ReturnEvent]] = []
        server_version = 0
        next_sequence = 0
        buffered_returns = 0
        group_id = 0
        group_records: list[ReturnRecord] = []

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
        groups: list[UpdateGroupRecord] = []
        while len(records) < max_returns:
            _, _, event = heapq.heappop(queue)
            arrival_version = server_version
            staleness = arrival_version - event.base_version

            group_position = buffered_returns + 1
            aggregated = group_position >= self.buffer_size
            new_server_version = server_version + 1 if aggregated else server_version
            record = ReturnRecord(
                client_id=event.client.client_id,
                rank=event.client.rank,
                num_samples=event.client.num_samples,
                dispatch_time=event.dispatch_time,
                finish_time=event.finish_time,
                base_version=event.base_version,
                arrival_version=arrival_version,
                staleness=staleness,
                aggregated=aggregated,
                new_server_version=new_server_version,
                group_id=group_id,
                group_version=arrival_version,
                group_position=group_position,
                buffer_size=self.buffer_size,
                group_closed=aggregated,
            )
            records.append(record)
            group_records.append(record)

            if aggregated:
                server_version = new_server_version
                groups.append(_build_group_record(group_records, closed=True))
                group_records = []
                group_id += 1
                buffered_returns = 0
            else:
                buffered_returns = group_position

            next_event = self._dispatch(
                client=event.client,
                now=event.finish_time,
                base_version=server_version,
                sequence_id=next_sequence,
                rng=rng,
            )
            heapq.heappush(queue, (next_event.finish_time, next_event.sequence_id, next_event))
            next_sequence += 1

        if group_records:
            groups.append(_build_group_record(group_records, closed=False))

        return SimulationTrace(tuple(records), tuple(groups))

    def _run_cohort(self, *, max_returns: int) -> SimulationTrace:
        rng = random.Random(self.seed)
        records: list[ReturnRecord] = []
        groups: list[UpdateGroupRecord] = []
        server_version = 0
        group_id = 0
        next_sequence = 0
        next_client_index = 0
        now = 0.0

        while len(records) < max_returns:
            remaining = max_returns - len(records)
            target_size = min(self.buffer_size, remaining)
            cohort_clients = [
                self.clients[(next_client_index + offset) % len(self.clients)]
                for offset in range(target_size)
            ]
            events = [
                self._dispatch(
                    client=client,
                    now=now,
                    base_version=server_version,
                    sequence_id=next_sequence + offset,
                    rng=rng,
                )
                for offset, client in enumerate(cohort_clients)
            ]
            next_sequence += len(events)
            events.sort(key=lambda event: (event.finish_time, event.sequence_id))

            group_records: list[ReturnRecord] = []
            for position, event in enumerate(events, start=1):
                aggregated = target_size == self.buffer_size and position == target_size
                new_server_version = server_version + 1 if aggregated else server_version
                record = ReturnRecord(
                    client_id=event.client.client_id,
                    rank=event.client.rank,
                    num_samples=event.client.num_samples,
                    dispatch_time=event.dispatch_time,
                    finish_time=event.finish_time,
                    base_version=event.base_version,
                    arrival_version=server_version,
                    staleness=0,
                    aggregated=aggregated,
                    new_server_version=new_server_version,
                    group_id=group_id,
                    group_version=server_version,
                    group_position=position,
                    buffer_size=self.buffer_size,
                    group_closed=aggregated,
                )
                records.append(record)
                group_records.append(record)

            closed = len(group_records) == self.buffer_size
            if closed:
                server_version += 1
            groups.append(_build_group_record(group_records, closed=closed))
            now = max(event.finish_time for event in events)
            next_client_index = (next_client_index + target_size) % len(self.clients)
            group_id += 1

        return SimulationTrace(tuple(records), tuple(groups))


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


def _build_group_record(records: list[ReturnRecord], *, closed: bool) -> UpdateGroupRecord:
    first = records[0]
    last = records[-1]
    return UpdateGroupRecord(
        group_id=first.group_id,
        group_version=first.group_version,
        server_version_after=last.new_server_version,
        first_arrival_time=first.finish_time,
        aggregation_time=last.finish_time,
        closed=closed,
        records=tuple(records),
    )
