# Week 8 - RIFT stale-selective gain-mass confirmation

Status: protocol da khoa; chua doc ket qua seeds 3301-3306.

## Ly do thay doi

Development traces cho thay gate nho co hai failure mode:

- gate moi update gay false rejection cho update fresh;
- mot so stale update co first-order gain rat manh bi calibration gate phu quyet.

Ban RIFT duoc chon chi gate update co `tau >= late_tau`, nen singular components
stale de bao phu 90% tong positive predicted gain, va bypass gate khi predicted
gain chuan hoa dat it nhat 1% calibration loss. UCB gate `z=0.25` van chan cac
stale update co signal yeu va khong chac chan.

## Development evidence

So voi Spectral filter tren seeds 3101-3103, eval offset 0:

| Task | RIFT acc | Spectral acc | RIFT class NLL | Spectral class NLL | RIFT late harmful | Spectral late harmful |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QNLI | 81.250% | 81.250% | 0.431991 | 0.432347 | 0.000% | 8.333% |
| MNLI-m | 70.833% | 70.833% | 0.668039 | 0.668243 | 0.000% | 16.667% |
| MNLI-mm | 80.556% | 80.208% | 0.605288 | 0.603357 | 0.000% | 16.667% |

MNLI-mm NLL cua RIFT kem 0.001931, van nam trong bien non-inferiority 0.005.
Day la development selection evidence, khong phai held-out conclusion.

## Confirmation lock

- Manifest: `configs/local_1_5b_rift_gainmass_confirmation_matrix.json`.
- Seeds: 3301-3306, tat ca seed co trong so bang nhau.
- Eval offset: 256, 96 examples moi task.
- Methods: FedRot, Spectral filter, AlignFed calibration va RIFT.
- Regime: non-IID label shard, heterogeneous rank, high staleness.
- Khong bo seed sau khi xem ket qua.

Offset 256 tranh ca development range 0-95 va confirmation cu 64-159. Vi hai
range cu bi overlap 32 examples, confirmation cu phai duoc xem la diagnostic,
khong con la final independent evidence.

## GO gate

RIFT chi duoc GO neu du 6 paired seeds va:

- client coverage bang 1, moi run co it nhat 4 late events;
- acceptance rate toi thieu 30%;
- accuracy gap so voi doi thu manh nhat khong thap hon -0.5 percentage point;
- class-NLL gain khong thap hon -0.005;
- late harmful rate va cumulative late harm deu cai thien duong;
- khong co integrity/provenance failure.

Accuracy/NLL vuot moi doi thu la ket qua mong muon, khong phai dieu kien duoc
phep tao ra bang seed selection. Claim chinh la late-update safety voi utility
non-inferiority.
