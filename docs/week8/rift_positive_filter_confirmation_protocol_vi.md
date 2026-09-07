# Week 8 - RIFT positive-filter held-out protocol

Ngay khoa protocol: 2026-09-07

## Trang thai

Protocol nay duoc khoa truoc khi doc bat ky ket qua nao tai offset `768` hoac
seeds `4301-4306`. Development evidence tai offset `512` khong duoc gop vao
held-out verdict.

Manifest:
`configs/local_1_5b_rift_positive_filter_confirmation_matrix.json`.

## Gia thuyet duoc kiem dinh

Trong asynchronous non-IID Federated LoRA co heterogeneous rank va high
staleness, RIFT component filter + bounded extreme-delay rescue phai:

1. giam late harmful rate va cumulative late harm so voi cac control co update
   utilization cao, trong khi accuracy va class NLL khong kem dang ke;
2. dat quality khong kem va acceptance cao hon whole-update calibration gate,
   trong khi van nam trong mot absolute late-harm budget;
3. khong collapse client coverage hoac acceptance de tao ra safety gia.

Day la claim ve utility-safety-utilization, khong phai claim RIFT co raw harmful
rate thap hon mot method duoc phep reject tuy y nhieu update.

## Setup da khoa

- Backbone: `Qwen/Qwen2.5-1.5B-Instruct`, QLoRA 4-bit.
- Tasks: QNLI, MNLI matched va MNLI mismatched.
- Eval: 96 examples/task, offset `768`.
- Seeds: `4301-4306`, tat ca co trong so bang nhau.
- FL: 4 clients, label-shard non-IID, ranks `[2,4,8,4]`, compute times
  `[1,2,5,10]`, immediate async (`buffer_size=1`).
- 4 warmup returns, 16 measured returns; moi run phai co 4 late returns va
  client coverage 100%.
- Controls: FedRot, Spectral filter matched ablation, AlignFed calibration
  control. Fidelity labels trong analyzer van bat buoc.

## GO gates

Moi comparison tren ca ba task phai co 6 paired seeds, RIFT acceptance >= 90%,
client coverage = 1 va cung pass:

### FedRot va Spectral filter

- lower CI95 cua accuracy delta >= `-0.5 pp`;
- lower CI95 cua class-NLL improvement >= `-0.005`;
- mean late-harm reduction > 0 va lower CI95 >= 0;
- mean cumulative late-harm reduction > 0 va lower CI95 >= 0.

### AlignFed calibration

AlignFed la reject-heavy whole-update gate, nen raw harmful rate khong duoc
dung rieng le de ket luan. RIFT phai:

- pass cung hai CI95 non-inferiority gates ve accuracy va class NLL;
- co accuracy gain hoac class-NLL improvement duong theo point estimate;
- co mean acceptance advantage >= `10 pp`;
- co mean normalized cumulative late harm <= `0.0025`.

Nguong `0.0025` nghia la cumulative harmful increase trung binh khong qua
`0.0025` absolute monitor class-NLL tren moi late return. Nguong va
comparator-specific rule duoc dat sau development analysis, sau do khoa truoc
held-out moi.

## Integrity rules

- Khong bo seed, doi seed, doi offset hoac sua threshold sau khi xem ket qua.
- Best accuracy/best NLL chi la descriptive; verdict dung mean paired va CI95.
- Khong tron artifact tu matrix/commit khac. Analyzer kiem tra matrix hash,
  config fingerprint, clean worktree va measured-return count.
- Neu thieu run/provenance/coverage thi `INCONCLUSIVE`; neu fail utility,
  utilization hay absolute budget thi `NO_GO`; neu safety CI chua tach khoi 0
  thi khong duoc tuyen bo GO.
- Ket qua nay khong chung minh buffered async; `buffer_size > 1` can mot matrix
  rieng sau Week 8 immediate-async confirmation.

## Lenh chay

```powershell
$env:PYTHONPATH = "src"
python scripts/run_week8_classification_matrix.py `
  --matrix configs/local_1_5b_rift_positive_filter_confirmation_matrix.json `
  --output-root outputs/local_1_5b_rift_positive_filter_confirmation

python scripts/analyze_kaggle_3b_rift_competitors.py `
  --input-dir outputs/local_1_5b_rift_positive_filter_confirmation `
  --output-dir outputs/local_1_5b_rift_positive_filter_confirmation_analysis `
  --matrix configs/local_1_5b_rift_positive_filter_confirmation_matrix.json
```
