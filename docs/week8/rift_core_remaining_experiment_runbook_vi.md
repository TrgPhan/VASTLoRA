# RIFT-Core: runbook cho cac experiment con lai

## Trang thai va pham vi

Development da hoan tat cho QNLI va MNLI-m tren seeds 5101-5103. Khong chay
lai hai cohort nay. Matrix development moi chi bo sung hai task con thieu:

- SST-2, development offset 512, 96 eval examples, max length 192.
- MNLI-mm, development offset 1024, 96 eval examples, max length 128.

Ca hai task chay Spectral Filter, AlignFed calibration, RIFT gate-only,
RIFT-Diag va RIFT-Core tren cung non-IID + high-staleness schedule. Day la
development evidence, khong phai held-out verdict.

Sau khi development du bon task, matrix confirmation rieng se chay SST-2,
QNLI, MNLI-m va MNLI-mm voi sau seed moi. Primary target la RIFT-Core. Controls
gom FedRot, Spectral Filter, AlignFed calibration va RIFT gate-only.
RIFT-Diag la ablation development, khong tham gia primary GO gate.

## File chay

- `configs/rift_core_remaining_tasks_local_4gb_matrix.json`: hai development
  task con thieu, 30 jobs.
- `configs/rift_core_heldout_confirmation_matrix.json`: bon held-out task,
  120 jobs.
- `scripts/run_week8_classification_matrix.py`: tao config tung job, validate,
  skip artifact clean da hoan tat va chay runner.
- `scripts/analyze_kaggle_3b_rift_competitors.py`: tong hop mean, best seed,
  paired CI95, completeness va GO/NO-GO.

Khong sua `src/riftlora`, core hyperparameters, seed hoac held-out offset sau
khi bat dau confirmation.

## Preflight khong chay model

Tu repo root:

```powershell
python -m pytest tests/test_rift_core_experiment_matrices.py tests/test_week8_matrix_runner.py tests/test_kaggle_3b_tasks.py -q
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_remaining_tasks_local_4gb_matrix.json --output-root outputs/rift_core_remaining_dev_1_5b_4gb --dry-run
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_heldout_confirmation_matrix.json --output-root outputs/rift_core_confirmation_1_5b_4gb --dry-run
```

Dry-run phai liet ke 30 development jobs va 120 confirmation jobs, hoac
`skip completed` neu artifact trung fingerprint va den tu clean worktree.

## Chay development con lai

Chay tung task de de resume:

```powershell
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_remaining_tasks_local_4gb_matrix.json --task sst2 --regime noniid_high_staleness --output-root outputs/rift_core_remaining_dev_1_5b_4gb
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_remaining_tasks_local_4gb_matrix.json --task mnli_mm --regime noniid_high_staleness --output-root outputs/rift_core_remaining_dev_1_5b_4gb
```

Chay lai cung lenh de resume; runner tu skip result clean dung schema, matrix
fingerprint va config fingerprint. Khong dung `--force` neu khong co ly do da
ghi lai. Neu can chay mot job de kiem tra GPU truoc:

```powershell
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_remaining_tasks_local_4gb_matrix.json --task sst2 --regime noniid_high_staleness --method rift_core --seed 5101 --output-root outputs/rift_core_remaining_dev_1_5b_4gb
```

Job don nay van la mot phan cua cohort chinh; sau do chay lenh full task de bo
sung cac method/seed con lai.

## Phan tich development

Analyzer se bao incomplete neu moi chay mot phan matrix. Chi dung verdict sau
khi du 30 jobs:

```powershell
python scripts/analyze_kaggle_3b_rift_competitors.py --input-dir outputs/rift_core_remaining_dev_1_5b_4gb --output-dir outputs/rift_core_remaining_dev_1_5b_4gb_analysis --matrix configs/rift_core_remaining_tasks_local_4gb_matrix.json --target rift_core
```

Ghep bao cao nay voi hai artifact da co cho QNLI va MNLI-m. Khong ghep raw
result vao cung input analyzer vi chung den tu matrix fingerprint khac nhau.

## Chay held-out confirmation

Chi bat dau sau khi commit dang chay la clean va khong con tune Core. Thu tu
task chi de quan ly tai nguyen, khong mang y nghia chon task dep:

```powershell
git status --short
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_heldout_confirmation_matrix.json --task sst2 --output-root outputs/rift_core_confirmation_1_5b_4gb
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_heldout_confirmation_matrix.json --task qnli --output-root outputs/rift_core_confirmation_1_5b_4gb
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_heldout_confirmation_matrix.json --task mnli_m --output-root outputs/rift_core_confirmation_1_5b_4gb
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_heldout_confirmation_matrix.json --task mnli_mm --output-root outputs/rift_core_confirmation_1_5b_4gb
```

SST-2 dung 168 mau con lai tu shuffled offset 704. QNLI va hai MNLI split
dung 512 mau tu offset 1536. Tat ca dung fixed `eval_shuffle_seed=314159` tu
base config. Seeds 6101-6106 co trong so bang nhau; khong loai seed dua tren
ket qua.

Phan tich sau khi du 120 jobs:

```powershell
python scripts/analyze_kaggle_3b_rift_competitors.py --input-dir outputs/rift_core_confirmation_1_5b_4gb --output-dir outputs/rift_core_confirmation_1_5b_4gb_analysis --matrix configs/rift_core_heldout_confirmation_matrix.json --target rift_core
```

## Metric va verdict

Theo doi rieng tung task va paired seed:

- Primary quality: final Accuracy va class NLL; bao cao mean, CI95, best va
  worst seed. Best seed chi mo ta, khong dung lam verdict.
- Safety: harmful, late harmful, cumulative late harm, normalized cumulative
  late harm va worst-step loss increase.
- Khong duoc reject de tao ket qua dep: acceptance phai >= 50%, client return
  coverage = 100%, moi seed can it nhat 8 late events.
- Cost: runtime, peak GPU memory, calibration gradient passes va accepted
  update rank.

GO cua matrix confirmation yeu cau du job/provenance, sau paired seeds, quality
khong thua qua margin da khoa, co it nhat mot point improvement ve Accuracy
hoac class NLL, va safety tot hon external controls. So voi RIFT gate-only,
GO gate tap trung vao quality vi day la internal ablation. Neu Accuracy hoac
NLL fail non-inferiority tren hard slice thi ket luan la NO-GO; neu thieu job,
dirty artifact hay khong du late event thi INCONCLUSIVE.

## Prompt handoff de chay o luot sau

```text
Hay doc docs/week8/rift_core_remaining_experiment_runbook_vi.md. Kiem tra Git
worktree va GPU, chay preflight, sau do resume matrix development con thieu.
Khong chay lai QNLI/MNLI-m development da hoan tat, khong sua logic RIFT-Core,
khong dung --force va khong chon seed. Khi du 30 jobs, analyze voi target
rift_core va tong hop SST-2/MNLI-mm cung ket qua QNLI/MNLI-m cu. Chi neu bon
task development hop le moi chay held-out confirmation da freeze; bao cao ba
bang Accuracy, class NLL, Harmful/Late harmful kem paired CI95 va provenance.
```
