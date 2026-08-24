from vastlora.scale.tradeoff import reserved_train_eval_indices, select_tradeoff


def test_reserved_development_holdout_is_deterministic_and_disjoint() -> None:
    train_a, eval_a = reserved_train_eval_indices(
        100, eval_offset=10, eval_examples=20, shuffle_seed=17
    )
    train_b, eval_b = reserved_train_eval_indices(
        100, eval_offset=10, eval_examples=20, shuffle_seed=17
    )

    assert (train_a, eval_a) == (train_b, eval_b)
    assert len(eval_a) == 20
    assert set(train_a).isdisjoint(eval_a)
    assert set(train_a) | set(eval_a) == set(range(100))


def test_tradeoff_selection_rejects_accuracy_only_candidate() -> None:
    rows = [
        {
            "candidate": "freshness",
            "method": "freshness",
            "seed": 2025,
            "final_accuracy": 0.70,
            "final_balanced_accuracy": 0.70,
            "final_nll": 0.50,
            "final_binary_nll": 0.20,
            "final_brier": 0.10,
        },
        {
            "candidate": "hybrid_beta005",
            "method": "mtip_hybrid",
            "seed": 2025,
            "final_accuracy": 0.72,
            "final_balanced_accuracy": 0.72,
            "final_nll": 0.60,
            "final_binary_nll": 0.25,
            "final_brier": 0.14,
        },
        {
            "candidate": "hybrid_beta010",
            "method": "mtip_hybrid",
            "seed": 2025,
            "final_accuracy": 0.71,
            "final_balanced_accuracy": 0.71,
            "final_nll": 0.49,
            "final_binary_nll": 0.19,
            "final_brier": 0.095,
        },
    ]

    selection = select_tradeoff(rows)

    assert selection["status"] == "DEV_GATE_PASS"
    assert selection["selected"]["name"] == "hybrid_beta010"
