# Week 8 - Main classification matrix

Status: infrastructure ready, GPU matrix pending

## Pham vi chot

- Task chinh: SST-2, QNLI, MNLI-m va MNLI-mm.
- Regime chinh: IID homogeneous, IID heterogeneous, non-IID high-staleness.
- Seeds: 6 seed de co paired statistical report.
- Metrics chinh: Accuracy, label NLL, Brier, Harmful, Late harmful, cumulative
  harm, Acceptance va measured-client coverage.
- Sequence NLL va EOS NLL duoc ghi rieng de chan doan; evaluator classification
  chi dung label NLL, khong dung EOS de quyet dinh nhan.
- Local training, RIFT component-scoring gradient va calibration gate cung dung
  label-only objective; khong con truong hop filter chon component theo EOS trong
  khi gate danh gia theo label NLL.

## Da review

- Week 1: novelty contract da thu hep vao late-update safety; Spectral Surgery overlap da duoc ghi ro.
- Week 2: exact innovation, compact SVD, rank-wise gain va paired gate da co tests.
- Week 3: SST-2/QNLI split audit, label-shard diagnostics va async trace da co artifact.
- Week 4: small-model RIFT co research GO voi claim hep; chua phai final thesis GO.
- Week 5: RIFT kernel da duoc tach API, co error handling/logging va full tests.
- Week 6: competitor wrappers co fidelity labels; khong claim full-faithful cho method thieu protocol.
- Week 7: buffered trace/group metadata da co, nhung model runner 3B hien tai van immediate async `buffer_size=1`.
- MNLI-m/mm da duoc implement voi evaluator 3 lop: entailment, neutral, contradiction.
- MNLI dung `validation_matched` va `validation_mismatched` rieng, voi `run_name` rieng de analyzer khong tron hai split.
- `binary_nll` chi co y nghia voi SST-2/QNLI; MNLI dung `label_nll`, Accuracy va multiclass Brier.
  `sequence_nll` va `eos_nll` chi la metric chan doan.
- Cac config `week4_*` la schema/backbone BERT nho, khong duoc dung lam base cho runner Qwen 3B.
- Week 8 dung `kaggle_3b_rift_competitors.json` lam base va overlay task/regime bang script matrix.

## Deliverables

- SST-2, QNLI, MNLI-m va MNLI-mm matrix config.
- Script reproducible de chay tung task/regime/method/seed, co skip run da hoan tat.
- 4 task views x 3 regimes x 8 methods x 6 seeds = 576 runs khi chay full matrix.
- `raw` duoc bo khoi matrix vi trung operator voi `fedex`; vi tri do duoc dung
  cho `spectral_filter` de kiem tra filter-only mechanism cua RIFT.
- `iid_heterogeneous` giu compute time dong nhat va chi doi client rank, tranh
  tron rank heterogeneity voi latency heterogeneity.
- Analyzer tao paired CI95 va hard-slice verdict `GO`/`NO_GO`/`INCONCLUSIVE`.
- Acceptance gate: RIFT phai giu acceptance rate toi thieu 30% tren measured updates.
- Analyzer doc gate truc tiep tu matrix manifest, reject schema/config/matrix
  fingerprint cu, duplicate run va mixed Git commit.
- Hard-slice schedule thu 100 returns (8 warmup + 92 measured) de moi logical
  client co co hoi xuat hien; phan bo return van co the lech theo compute time.
- Runner log them measured return/accept count tung client, `tau >= 16` extreme
  tail, cumulative/normalized harm, worst-step loss, utility per returned update,
  route rate va retained fraction.
- Best accuracy va best label NLL duoc report kem seed nhu descriptive metrics;
  chung khong thay mean/CI va khong duoc dung mot minh de quyet dinh GO.
- Paired statistical report voi CI95.
- Hard-slice verdict cho non-IID + high staleness.
- Seed alignment guard.

## Files

- [configs/week8_rift_classification_matrix.json](../../configs/week8_rift_classification_matrix.json)
- [scripts/analyze_kaggle_3b_rift_competitors.py](../../scripts/analyze_kaggle_3b_rift_competitors.py)
- [scripts/run_kaggle_3b.py](../../scripts/run_kaggle_3b.py)
- [scripts/run_week8_classification_matrix.py](../../scripts/run_week8_classification_matrix.py)

## Cach chay

Dry-run mot slice:

```text
python scripts/run_week8_classification_matrix.py --task qnli --regime noniid_high_staleness --method rift --seed 4101 --dry-run
```

Chay full matrix:

```text
python scripts/run_week8_classification_matrix.py
```

Sau khi co output, phan tich toan bo matrix:

```text
python scripts/analyze_kaggle_3b_rift_competitors.py --input-dir outputs/week8_classification_matrix --output-dir outputs/week8_analysis
```

Analyzer tu dong quet de quy cac `result.json`, sau do kiem tra paired seed trong tung task/regime.

## Tuning guardrail

- Tune RIFT chi tren dev seeds/task da khai bao truoc.
- Chon mot Pareto configuration dua tren late harm, label NLL, accuracy va
  acceptance; khong chon rieng seed co best score.
- Dong bang config truoc khi chay cac seed 4101-4106.
- Cac metric co the lam RIFT mat loi the bieu kien (client coverage, cumulative
  harm, filter-only va extreme tail) van bat buoc report, vi chung phan biet
  method thuc su tot voi reject/cherry-pick.
