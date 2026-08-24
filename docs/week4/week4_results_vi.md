# Week 4 kill-test results

Decision: **NO-GO**

Reason: rho does not add robust predictive and transport value beyond staleness.

## Preregistered gate metrics

| Regime | N | Partial Spearman | 95% CI | CV R2 gain | AUROC gain | VAST - freshness utility | Harmful freshness -> VAST | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| iid_heterogeneous | 180 | -0.060 | [-0.200, 0.098] | 0.012 | -0.033 | -0.000088 | 0.467 -> 0.478 | no |
| iid_homogeneous | 180 | -0.043 | [-0.182, 0.100] | 0.004 | 0.106 | -0.001723 | 0.456 -> 0.472 | no |
| noniid_high_staleness | 180 | 0.009 | [-0.144, 0.159] | 0.002 | -0.001 | 0.005468 | 0.439 -> 0.450 | no |

## Matched-tau analysis

| regime                | tau_band   |   n |   spearman_rho_utility |   high_minus_low_utility |   high_rho_harmful_rate |   low_rho_harmful_rate |
|:----------------------|:-----------|----:|-----------------------:|-------------------------:|------------------------:|-----------------------:|
| iid_heterogeneous     | 0-2        |   9 |                 0.4667 |                   0.0252 |                  0.0000 |                 0.2000 |
| iid_heterogeneous     | 3-7        | 102 |                 0.0158 |                   0.0630 |                  0.4510 |                 0.5294 |
| iid_heterogeneous     | 8+         |  69 |                -0.1797 |                  -0.1152 |                  0.6176 |                 0.4571 |
| iid_homogeneous       | 0-2        |   9 |                -0.1667 |                   0.0091 |                  0.2500 |                 0.4000 |
| iid_homogeneous       | 3-7        | 102 |                -0.0500 |                   0.0213 |                  0.4902 |                 0.4314 |
| iid_homogeneous       | 8+         |  69 |                -0.0518 |                   0.0435 |                  0.5588 |                 0.5429 |
| noniid_high_staleness | 0-2        |  42 |                -0.0926 |                   0.0089 |                  0.4762 |                 0.4286 |
| noniid_high_staleness | 3-7        |  81 |                 0.1029 |                   0.0472 |                  0.5750 |                 0.5854 |
| noniid_high_staleness | 8+         |  57 |                -0.0280 |                  -0.0569 |                  0.4286 |                 0.2759 |

## Interpretation guardrails

- The gate uses two-sided rho and thresholds frozen before the full run.
- Validation loss is used only for post-hoc scientific evaluation, not by the transport rule.
- A single SST-2/BERT-tiny kill-test can reject the current hypothesis, but cannot establish the final thesis claim without a second task and larger backbone.
- Full numeric outputs and plots are in `outputs/week4/analysis/`.
