# Qwen2.5-1.5B held-out confirmation protocol

Ngay dong bang: 2026-09-05, truoc khi chay bat ky held-out cell nao.

## Pham vi

- Tasks: QNLI, MNLI matched va MNLI mismatched.
- Methods: FedRot, Spectral filter, AlignFed calibration va RIFT.
- Seeds: `3201-3206` cho moi task/method.
- Eval offset: `64`, tach khoi development offset `0`.
- Moi run: 4 warmup + 16 measured returns va 96 eval examples.
- Regime: label-shard non-IID, client rank `[2, 4, 8, 4]`, compute time
  `[1, 2, 5, 10]`.
- Config: `configs/local_1_5b_rift_remaining_tasks_confirmation_matrix.json`.

## Seed policy

Khong duoc chon hoac loai seed dua tren accuracy, NLL, harmful rate hay ket qua
RIFT/doi thu. Moi seed co trong manifest co trong so bang nhau.

Mot run chi duoc rerun khi crash, OOM, output NaN/khong day du, sai schema,
fingerprint hoac Git provenance. Tieu chi nay ap dung giong nhau cho moi method;
rerun phai giu nguyen task, method va seed.

## Bao cao

- Primary: paired mean difference va CI95 tren du 6 seed.
- Robust secondary: median, paired win/tie/loss count va tung seed result.
- Descriptive: best accuracy va best class NLL, khong dung de ra verdict.
- Safety: harmful rate, late harmful rate, cumulative late harm, acceptance,
  late-event count va client coverage.

Khong seed don le nao duoc phep "chiem nhieu" hon seed khac. Median va paired
win count giam anh huong cua outlier ma khong pha vo held-out protocol.

## Verdict

`GO` chi khi analyzer pass day du gate accuracy/NLL non-inferiority,
acceptance, client coverage, late-event count va late-harm improvement voi moi
doi thu. Neu RIFT chi thang accuracy hoac harmful trung binh nhung CI/gate
khong pass, verdict phai la `INCONCLUSIVE` hoac `NO_GO` dung theo output.
