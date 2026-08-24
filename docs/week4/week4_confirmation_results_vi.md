# Week 4 kill-test results

Decision: **NO-GO**

Reason: rho does not add robust predictive and transport value beyond staleness.

## Preregistered gate metrics

| Regime | N | Partial Spearman | 95% CI | CV R2 gain | AUROC gain | VAST - freshness utility | Harmful freshness -> VAST | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| iid_heterogeneous | 180 | 0.064 | [-0.086, 0.219] | -0.006 | 0.021 | 0.000014 | 0.511 -> 0.489 | no |
| iid_homogeneous | 180 | 0.055 | [-0.092, 0.207] | -0.006 | -0.009 | 0.000026 | 0.511 -> 0.489 | no |
| noniid_high_staleness | 180 | -0.012 | [-0.167, 0.143] | -0.015 | 0.031 | 0.000034 | 0.433 -> 0.444 | no |

## Matched-tau analysis

| regime                | tau_band   |   n |   spearman_rho_utility |   high_minus_low_utility |   high_rho_harmful_rate |   low_rho_harmful_rate |
|:----------------------|:-----------|----:|-----------------------:|-------------------------:|------------------------:|-----------------------:|
| iid_heterogeneous     | 0-2        |   9 |                 0.6000 |                   0.0016 |                  0.5000 |                 0.6000 |
| iid_heterogeneous     | 3-7        | 102 |                 0.0287 |                   0.0001 |                  0.4902 |                 0.4902 |
| iid_heterogeneous     | 8+         |  69 |                 0.0328 |                  -0.0001 |                  0.5882 |                 0.6286 |
| iid_homogeneous       | 0-2        |   9 |                 0.6833 |                   0.0037 |                  0.2500 |                 1.0000 |
| iid_homogeneous       | 3-7        | 102 |                 0.0292 |                   0.0003 |                  0.5294 |                 0.5490 |
| iid_homogeneous       | 8+         |  69 |                -0.0300 |                  -0.0002 |                  0.4706 |                 0.4857 |
| noniid_high_staleness | 0-2        |  42 |                 0.0756 |                   0.0002 |                  0.4286 |                 0.4286 |
| noniid_high_staleness | 3-7        |  81 |                -0.1668 |                  -0.0013 |                  0.5000 |                 0.4390 |
| noniid_high_staleness | 8+         |  57 |                 0.1187 |                   0.0024 |                  0.4286 |                 0.4828 |

## Interpretation guardrails

- The gate uses two-sided rho and thresholds frozen before the full run.
- Validation loss is used only for post-hoc scientific evaluation, not by the transport rule.
- A single SST-2/BERT-tiny kill-test can reject the current hypothesis, but cannot establish the final thesis claim without a second task and larger backbone.
- Full numeric outputs and plots are in `outputs/week4/analysis/`.
