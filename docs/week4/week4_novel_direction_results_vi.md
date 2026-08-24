# Week 4 - MTIP-LoRA direction search

Ngay chay: 2026-08-24

## Verdict

**CONDITIONAL GO** cho MTIP-LoRA tren regime non-IID/high-staleness cua SST-2.

**NO-GO** cho claim task-general o thoi diem hien tai, vi QNLI confirmation khong lap lai ket qua mot cach nhat quan.

MTIP-LoRA la ten lam viec cho **Minimal Two-sided Temporal Intersection Projection**. Day la pivot tu VAST goc, khong phai doi ten VAST.

## Method duoc chon

Voi exact intrinsic innovation `D` va temporal bases rank 2 `Q_L`, `Q_R`:

```text
T_MTIP(D) = Q_L Q_L^T D Q_R Q_R^T
```

Method:

- factor-dispatch dung `B0,A0` tu server;
- aggregate exact innovation `B1 A1 - B0 A0`;
- gauge-invariant compact SVD;
- history 8 accepted updates;
- reference rank 2;
- khong giu stale residual;
- khong can calibration data hay server labels;
- ho tro client rank 4/8/16.

## Cac huong da thu va loai

| Direction | Ket qua |
|---|---|
| VAST residual `mu R` | Fail held-out; residual stale gay drift |
| Residual power `mu^2`, `mu^4` | Dev loss dep hon, unseen seed fail |
| Residual trust budget | Kem projection-only |
| Projection norm compensation | Kem projection-only |
| Left-only projection | Xau ro ret |
| Right-only projection | Loss nhinh hon nhe, accuracy khong hon; novelty threat cao |
| Union/lattice four-block transport | Single-side blocks van mang client drift |
| Cross-timescale persistent core | Khong hon fixed rank 2 |
| Gradient-risk switch | Posthoc tot, unseen QNLI fail |
| Reference rank 8 | Collapse; giu qua nhieu stale modes |
| Reference rank 4 | On dinh, nhung kem rank 2 |
| Reference rank 2 | Tot nhat va lap lai tren 6 SST-2 seeds |

## SST-2 confirmation

Non-IID label shards, mean staleness khoang 7.6, heterogeneous client ranks.

| Seed | Freshness acc | MTIP acc | Gain | Loss gain |
|---:|---:|---:|---:|---:|
| 59 | 73.394% | 74.312% | +0.917 pp | +0.014007 |
| 71 | 71.216% | 73.165% | +1.950 pp | +0.025978 |
| 89 | 70.642% | 74.427% | +3.784 pp | +0.022369 |
| 101 | 68.807% | 73.050% | +4.243 pp | +0.042645 |
| 113 | 63.876% | 72.362% | +8.486 pp | +0.079443 |
| 127 | 73.280% | 74.083% | +0.803 pp | +0.000356 |

Tong hop:

- accuracy wins: **6/6 seeds**;
- mean accuracy gain: **+3.364 pp**;
- minimum accuracy gain: **+0.803 pp**;
- mean loss gain: **+0.030800**;
- loss wins: **6/6 seeds**.

## Regime scope

Tren seed 101/113/127:

| Regime | Mean accuracy gain vs freshness | Mean loss gain | Ket luan |
|---|---:|---:|---|
| IID homogeneous | +0.038 pp | -0.000435 | Neutral/mixed |
| IID heterogeneous rank | +0.115 pp | -0.000012 | Neutral/mixed |
| Non-IID high staleness | **+4.511 pp** | **+0.040815** | GO regime |

Claim phai gioi han vao interaction giua data heterogeneity va high staleness. MTIP khong duoc claim la universally better aggregator.

## QNLI robustness gate

Checkpoint QNLI sach warm-start 5.000 examples dat 70.68% full-validation accuracy. Fixed rank-2 MTIP co mot development batch mixed: mean +1.11 pp, nhung chi thang accuracy 1/3 seed. Tren unseen seeds 179/191/211:

- freshness mean accuracy: **73.372%**;
- MTIP mean accuracy: **71.549%**;
- risk-switch mean accuracy: **71.810%**.

Do do task-general claim bi bac bo. QNLI phai duoc report la negative result.

## Novelty audit

Targeted search den 2026-08-24 khong tim thay mot method trung khop day du voi combination:

```text
asynchronous stale LoRA
+ exact gauge-invariant innovation
+ online two-sided temporal intersection
+ minimal fixed consensus core
+ data-free server transport
+ heterogeneous client ranks
```

Nhung khong the khang dinh tuyet doi "chua ai lam". Cac threat gan nhat:

- FedSteer: dynamic gradient subspace va corrective projection/caching;
- GLoRA: gauge-aware consensus subspace va heterogeneous-rank readout;
- FedAS-LoRA: input/output-side asymmetric factor sharing;
- Dysco: dynamic client-specific LoRA subspace boosting;
- FedRot-LoRA/FLoRG: factor/subspace alignment.

Novelty candidate cua MTIP nam o **two-sided temporal intersection cua exact stale LoRA matrix innovations**, khong nam o projection noi chung.

## Gate tiep theo

1. So sanh faithful voi FedSteer-inspired vector projection va GLoRA-like + freshness.
2. Lap lai tren backbone/task thu ba; QNLI da cho thay task dependence.
3. Thay fixed rank 2 bang rank rule co theoretical basis ma khong lam giam SST-2 result.
4. Neu khong thang strong baselines, giu ket qua nhu mot regime-specific thesis contribution, khong claim general method paper.

Raw artifacts:

- `outputs/week4_rank2_confirmation/`
- `outputs/week4_novel_matrix/`
- `outputs/week4_qnli/`
- `outputs/week4_qnli_risk_confirmation/`
