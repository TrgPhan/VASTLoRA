from vastlora.data import iid_partition_indices, label_shard_partition_indices


def test_iid_partition_is_deterministic_and_complete() -> None:
    first = iid_partition_indices(23, 5, seed=11)
    second = iid_partition_indices(23, 5, seed=11)

    assert first == second
    assert sorted(index for part in first for index in part) == list(range(23))
    assert max(len(part) for part in first) - min(len(part) for part in first) <= 1


def test_label_shard_partition_is_deterministic_and_complete() -> None:
    labels = ["a"] * 10 + ["b"] * 10 + ["c"] * 10

    first = label_shard_partition_indices(labels, 3, shards_per_client=2, seed=19)
    second = label_shard_partition_indices(labels, 3, shards_per_client=2, seed=19)

    assert first == second
    assert sorted(index for part in first for index in part) == list(range(30))
    assert all(part for part in first)


def test_label_shards_create_non_iid_clients() -> None:
    labels = [0] * 100 + [1] * 100

    partitions = label_shard_partition_indices(
        labels,
        10,
        shards_per_client=1,
        seed=17,
    )

    positive_rates = [
        sum(labels[index] for index in indices) / len(indices)
        for indices in partitions
    ]
    assert min(positive_rates) == 0.0
    assert max(positive_rates) == 1.0
