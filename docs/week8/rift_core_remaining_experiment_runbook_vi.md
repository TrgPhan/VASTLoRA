# RIFT-Core v2: scaled development va held-out runbook

## Trang thai

Cac artifact v1 voi 96 eval examples van la exploratory evidence. Khong tron
chung vao cohort v2 vi schema, code, matrix va eval sample count khac nhau.
V2 sua FedRot factor convention, tach worst all/late-step harm, dung
`harm_epsilon=1e-6`, pin model/dataset revision va chi resume result tu dung
Git commit. Monitor loss cua cung server version duoc cache de bo forward lap.

## Hai matrix chinh

- `configs/rift_core_development_matrix.json`: 4 tasks, 2 regimes, 8 methods,
  3 seeds = 192 jobs. SST-2 va moi NLI slice dung 512 eval examples.
- `configs/rift_core_heldout_confirmation_matrix.json`: 4 tasks, hard regime,
  8 methods, 6 seeds = 192 jobs. Moi task dung 1024 eval examples.

Methods: raw, freshness, FedRot, Spectral Filter, whole-update calibration,
RIFT gate-only, RIFT-Diag va RIFT-Core. RIFT-Diag co cung repair data, steps
va gate, nen la control truc tiep cho gia tri cua off-diagonal full core.

SST-2 confirmation dung 1024 train examples theo shuffle seed 271828 va
offset4096. Runner loai chung khoi train truoc khi chon calibration, monitor
va client data. Day la internal held-out, khong phai official GLUE validation.
QNLI va MNLI m/mm dung 1024 validation examples tu shuffled offset2048, seed
314159. Development va confirmation khong chong cua so theo tung split.

## Preflight

Chay tu clean committed worktree. Sau khi sua implementation, result tu commit
cu se khong duoc skip.

```powershell
python -m pytest tests/test_rift_core_experiment_matrices.py tests/test_week8_matrix_runner.py tests/test_kaggle_3b_tasks.py tests/test_peft_bridge.py tests/test_kaggle_rift_competitor_analysis.py -q
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_development_matrix.json --output-root outputs/rift_core_scaled_dev_v2_1_5b --dry-run
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_heldout_confirmation_matrix.json --output-root outputs/rift_core_confirmation_v2_1_5b --dry-run
```

Neu override backbone 3B, dung output root rieng. Co the pin exact revision:

```powershell
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_development_matrix.json --model-name Qwen/Qwen2.5-3B-Instruct --model-revision <COMMIT_SHA> --output-root outputs/rift_core_scaled_dev_v2_3b --dry-run
```

Neu khong truyen `--model-revision`, runner bo revision 1.5B ke thua de tranh
dung nham commit hash cho model 3B.

## Chay development

Bat dau voi hard slice. Chay lai cung command de resume; khong dung `--force`.

```powershell
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_development_matrix.json --task sst2 --regime noniid_high_staleness --output-root outputs/rift_core_scaled_dev_v2_1_5b
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_development_matrix.json --task qnli --regime noniid_high_staleness --output-root outputs/rift_core_scaled_dev_v2_1_5b
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_development_matrix.json --task mnli_m --regime noniid_high_staleness --output-root outputs/rift_core_scaled_dev_v2_1_5b
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_development_matrix.json --task mnli_mm --regime noniid_high_staleness --output-root outputs/rift_core_scaled_dev_v2_1_5b
```

Sau hard slice, chay `--regime iid_homogeneous`. Day la control de biet gain
den tu high-staleness/non-IID hay chi la server calibration noi chung.

Phan tich chi khi cohort da chon co du method/seed:

```powershell
python scripts/analyze_kaggle_3b_rift_competitors.py --input-dir outputs/rift_core_scaled_dev_v2_1_5b --output-dir outputs/rift_core_scaled_dev_v2_1_5b_analysis --matrix configs/rift_core_development_matrix.json --target rift_core
```

## Chay held-out

Chi chay sau khi config/Core da freeze va commit. Khong tune theo bat ky seed
6101-6106 nao, khong bo seed sau khi xem ket qua.

```powershell
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_heldout_confirmation_matrix.json --task sst2 --output-root outputs/rift_core_confirmation_v2_1_5b
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_heldout_confirmation_matrix.json --task qnli --output-root outputs/rift_core_confirmation_v2_1_5b
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_heldout_confirmation_matrix.json --task mnli_m --output-root outputs/rift_core_confirmation_v2_1_5b
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_heldout_confirmation_matrix.json --task mnli_mm --output-root outputs/rift_core_confirmation_v2_1_5b
```

```powershell
python scripts/analyze_kaggle_3b_rift_competitors.py --input-dir outputs/rift_core_confirmation_v2_1_5b --output-dir outputs/rift_core_confirmation_v2_1_5b_analysis --matrix configs/rift_core_heldout_confirmation_matrix.json --target rift_core
```

## Cach doc ket qua

Primary quality la final Accuracy va class NLL, voi paired delta/CI95 tren
tat ca seed. Best seed chi mo ta. Safety gom harmful rate, late harmful rate,
cumulative late harm, normalized late harm, worst all-step va worst late-step
loss increase. Luon ghi numerator/denominator late events, acceptance, return
coverage tung client, runtime va memory.

V2 yeu cau `ci95_any`: RIFT-Core phai co CI95 improvement ve Accuracy hoac
class NLL, dong thoi dat non-inferiority va safety gates da khoa. GO safety
cho phep hoa khi ca RIFT va opponent deu co late harm/cumulative harm bang 0;
neu opponent co harm thi van phai co reduction duong voi CI lower >= 0. GO
safety khong tu dong la accuracy breakthrough. De claim full-core mixing co ich,
Core phai vuot RIFT-Diag equal-budget control; neu khong, claim dung o muc
server-calibrated diagonal repair.

`spectral_filter` va `alignfed_calibration` trong simulator la matched local
controls, khong phai full official Spectral Surgery/AlignFed implementations.
Bao cao dung fidelity nay trong thesis.
