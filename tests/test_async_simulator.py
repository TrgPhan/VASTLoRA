import pytest

from riftlora.asyncfl import AsyncEventSimulator, ClientProfile, SnapshotStore


def _clients() -> list[ClientProfile]:
    return [
        ClientProfile("c_fast", rank=4, num_samples=20, compute_time=1.0, network_time=0.1),
        ClientProfile("c_mid", rank=8, num_samples=30, compute_time=2.0, network_time=0.2),
        ClientProfile("c_slow", rank=16, num_samples=40, compute_time=4.0, network_time=0.3),
    ]


def test_async_simulator_is_deterministic_for_same_seed() -> None:
    sim_a = AsyncEventSimulator(_clients(), seed=17, buffer_size=1)
    sim_b = AsyncEventSimulator(_clients(), seed=17, buffer_size=1)

    trace_a = sim_a.run(max_returns=12)
    trace_b = sim_b.run(max_returns=12)

    assert trace_a == trace_b
    assert trace_a.return_order == trace_b.return_order
    assert trace_a.dispatch_versions == trace_b.dispatch_versions
    assert trace_a.staleness_histogram() == trace_b.staleness_histogram()


def test_async_simulator_records_stale_returns() -> None:
    sim = AsyncEventSimulator(_clients(), seed=0, buffer_size=1)

    trace = sim.run(max_returns=9)

    assert max(trace.staleness_values) > 0
    assert trace.return_order[:3] == ("c_fast", "c_mid", "c_fast")
    assert trace.records[0].aggregated is True


def test_buffered_async_increments_version_only_when_buffer_is_full() -> None:
    sim = AsyncEventSimulator(_clients(), seed=0, buffer_size=2)

    trace = sim.run(max_returns=6)

    aggregated_flags = [record.aggregated for record in trace.records]
    versions = [record.new_server_version for record in trace.records]
    group_ids = [record.group_id for record in trace.records]
    group_versions = [record.group_version for record in trace.records]
    assert aggregated_flags == [False, True, False, True, False, True]
    assert versions == [0, 1, 1, 2, 2, 3]
    assert group_ids == [0, 0, 1, 1, 2, 2]
    assert group_versions == [0, 0, 1, 1, 2, 2]
    assert [group.size for group in trace.completed_groups] == [2, 2, 2]


def test_buffered_async_keeps_partial_group_metadata() -> None:
    sim = AsyncEventSimulator(_clients(), seed=1, buffer_size=4)

    trace = sim.run(max_returns=5)

    assert [record.group_id for record in trace.records] == [0, 0, 0, 0, 1]
    assert [record.group_position for record in trace.records] == [1, 2, 3, 4, 1]
    assert [record.group_closed for record in trace.records] == [False, False, False, True, False]
    assert trace.pending_groups[-1].size == 1
    assert trace.pending_groups[-1].closed is False


def test_cohort_schedule_dispatches_same_version_group() -> None:
    sim = AsyncEventSimulator(_clients(), seed=3, buffer_size=2, schedule_mode="cohort")

    trace = sim.run(max_returns=4)

    assert trace.staleness_values == (0, 0, 0, 0)
    assert [record.group_id for record in trace.records] == [0, 0, 1, 1]
    assert [record.group_closed for record in trace.records] == [False, True, False, True]
    assert [group.server_version_after for group in trace.completed_groups] == [1, 2]


def test_snapshot_store_keeps_versioned_payloads() -> None:
    store: SnapshotStore[str] = SnapshotStore()

    store.put(0, "adapter-v0")
    store.put(1, "adapter-v1")

    assert len(store) == 2
    assert store.get(0).payload == "adapter-v0"
    assert store.latest().version == 1
    assert store.latest().payload == "adapter-v1"
    with pytest.raises(ValueError):
        store.put(1, "duplicate")


