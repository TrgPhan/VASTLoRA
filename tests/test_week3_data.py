from riftlora.data import audit_text_split, build_iid_partition_manifest, summarize_partition_manifest


def test_text_split_audit_flags_conflicting_duplicates() -> None:
    records = [
        {"sentence": "A good movie", "label": 1, "idx": 0},
        {"sentence": " a  good movie ", "label": 0, "idx": 1},
    ]

    audit = audit_text_split(
        records,
        split="train",
        text_column="sentence",
        label_column="label",
        index_column="idx",
    )

    assert audit["rows"] == 2
    assert audit["duplicate_normalized_texts"] == 1
    assert audit["conflicting_duplicate_texts"] == 1
    assert audit["issues"][0]["level"] == "warning"


def test_text_split_audit_treats_unlabeled_test_as_allowed() -> None:
    records = [{"sentence": "hidden test label", "label": -1, "idx": 0}]

    audit = audit_text_split(
        records,
        split="test",
        text_column="sentence",
        label_column="label",
        index_column="idx",
        allow_unlabeled=True,
    )

    assert audit["unlabeled_rows"] == 1
    assert audit["issues"] == []


def test_iid_partition_manifest_is_complete_and_balanced() -> None:
    manifest = build_iid_partition_manifest(
        num_items=23,
        labels=[0, 1] * 12,
        num_clients=5,
        seeds=[7, 11],
        rank_schedule=[4, 8, 16, 4, 8],
    )

    summary = summarize_partition_manifest(manifest)

    assert all(row["complete"] for row in summary)
    assert all(row["client_sample_delta"] <= 1 for row in summary)
    assert manifest["partitions"][0]["clients"][2]["rank"] == 16

