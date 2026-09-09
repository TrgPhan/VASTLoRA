# RIFT-Core: prototype va plan thu nghiem

## Vi sao RIFT cu chua thang accuracy

Confirmation offset 768 da hoan tat 72 runs. So voi Spectral Filter, RIFT
khong co strict accuracy win tren 18 task-seed pairs; bay wins deu dung NLL
tie-break. MNLI-m mean accuracy hoa, MNLI-mm thap hon 0.347 pp, QNLI thap hon
0.174 pp. Day la ket qua cu, khong phai ket qua cua RIFT-Core.

Co che cu dung cung positive-component filter voi Spectral, roi chu yeu scale
hoac reject. No khong chu dong tao mot to hop huong moi de sua decision
boundary. NLL giam rat nho co the chi thay doi confidence ma khong doi argmax.
Eval chi 96 examples nen moi du doan doi nhan lam accuracy doi khoang 1.042 pp
tren mot seed. Calibration gradient 8/9 examples cua candidate cu cung de
nhieu; 16 measured returns va 4 late events moi run chi la pilot ngan.

Acceptance khong do tan suat doi huong hay scale: acceptance 96.88% KHONG co
nghia chi 3.12% updates bi thay doi. Can doc route/scale va update magnitude.

## Gia thuyet moi va implementation

Voi innovation D = U diag(s) V^T, Spectral chi chon cac diagonal components.
RIFT-Core toi uu C day du trong D_repaired = U C V^T. Cac phan tu ngoai duong
cheo cho phep ket hop u_i voi v_j (i != j). Matrix rank van khong vuot rank D,
nhung khong gian candidate rong hon diagonal reweighting.

- Khoi tao tu positive filter, tren cung current server state.
- Hoc C bang class NLL tren gradient-calibration split, cac trong so model
  duoc freeze. Microbatch giai phong tung graph; khong tao dense d_out*d_in.
- Chuan hoa C bang ||s||_2; penalty proximal ve positive-filter anchor.
- Moi layer bi rang buoc ||C-C0||_F <= radius / sqrt(1 + tau/delay_scale).
  Day la heuristic conservative theo delay, CHUA phai convergence theorem.
- Hook nhan server_update_weight, de objective fit dung step magnitude.
- Gate thu cac repaired scales va original positive-filter candidate, sau
  khi aggregate VA rank-project ve server_max_rank. Chon min empirical risk.
- Neu khong co candidate pass gate thi reject. Khong dung rescue bypass.
- Diagonal ablation rift_diag dung cung optimizer, data, steps va gate nhung
  chi hoc diagonal. Diagonal co the co he so am; no khac hard positive filter.

File: src/riftlora/scale/core_repair.py. Methods moi: rift_core, rift_diag.
Old rift/spectral_filter/alignfed_calibration giu implementation cu.

Gate voi z=0.25 khong la bao dam safety 95%. Tai su dung gate de chon nhieu
candidate va nhieu returns co the overfit; monitor va final eval phai tach
biet, va can calibration-shift ablation. Fitting truoc rank projection la
surrogate; chi gate moi danh gia dung state sau projection.

## Novelty va literature

Khong claim "chua ai lam". Preliminary search chi ho tro mot research
hypothesis, khong la exhaustive novelty clearance.

- Spectral Surgery: https://arxiv.org/abs/2603.03995
  Gradient-guided singular-value reweighting da co. Full core thay doi ca
  directions trong fixed left/right spans, khong chi singular values.
- LoRA-XS: https://arxiv.org/abs/2405.17604
  Hoc mot r*r core giua hai factors frozen da co. Khong claim full-core
  parameterization la moi. Prototype muon y tuong nay cho incoming stale
  innovation thay vi SVD cua pretrained weight, voi delay trust region va
  selection sau server-rank projection. Gia tri ket hop nay con phai chung minh.
- FedLAW: https://proceedings.mlr.press/v202/li23s.html
  Server proxy-data optimization cua aggregation weights da co.
- LoRA-FAIR: https://arxiv.org/abs/2411.14961
  Aggregation/initialization refinement da co; can so sanh objective va
  permitted parameter space, khong claim server repair la moi.
- FedPA-LoRA: https://arxiv.org/abs/2608.15381
  Product-space consistency va heterogeneous-rank reconstruction da co.

Claim co the kiem chung: delay-constrained full-core repair of returned LoRA
innovations, with selection after server-rank projection. Chi giu claim full
core neu no thang diagonal equal-budget control. Adam tren SVD coordinates
khong tu dong bao dam invariance trong degenerate singular subspaces.

## Chay bang model khac

Can torch, transformers, datasets, peft, accelerate, bitsandbytes. Trong repo:

```bash
python -m pip install -e ".[scale,dev]"
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_smoke_matrix.json --output-root outputs/rift_core_smoke
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_development_matrix.json --task qnli --regime noniid_high_staleness --output-root outputs/rift_core_dev_1_5b
python scripts/run_week8_classification_matrix.py --matrix configs/rift_core_development_matrix.json --model-name Qwen/Qwen2.5-3B-Instruct --task qnli --regime noniid_high_staleness --output-root outputs/rift_core_dev_3b
```

--model-name override tung task va nam trong config fingerprint. Moi backbone
dung output-root rieng. Lap lai cung lenh tu skip clean matching completed
runs; khong dung --force. Chay tu clean commit/worktree de artifact resume duoc.
Khong bat buoc chay full matrix ngay: full matrix co 120 runs (4 eval slices,
2 regimes, 5 methods, 3 seeds). Co the --task mnli_m / mnli_mm / sst2 va
--method rift_core de gioi han. Tat ca controls phai du cung seed truoc so sanh.

SST-2 offset512 la development da tung dung. QNLI/MNLI offset1024 la development
cho prototype nay. Khong goi bat ky ket qua tu matrix nay la held-out verdict.
MNLI-m/mm dung cung train/calibration trace, chi khac eval domain; khong dem
hai safety trajectories nay nhu hai thu nghiem doc lap.

## Thu tu thu nghiem va stop rules

1. Smoke QNLI: kiem tra hooks, finite losses, core movement va gate route.
   Hai measured returns/24 eval examples chi la integration test.
2. QNLI va MNLI-m: 3 development seeds, 5 methods. Primary la mean final
   accuracy, secondary class NLL, harmful, late harm, acceptance va runtime.
   Best accuracy/NLL chi de mo ta, khong dung chon seed/checkpoint tren test.
3. Ablation radius=0 (filter + same gate), full vs diagonal, delay_scale rat
   lon (gan nhu no-delay), gradient sizes 24/96. Chi thay tung yeu to;
   luu matrix/output rieng. Truoc khi tune tiep kiem tra core route duoc chon.
4. Neu co tin hieu, tang local_steps 1 -> 4, measured returns 16 -> 64 cho
   MOI method, eval 96 -> 512 cho QNLI/MNLI. SST-2 can offset/range rieng de
   khong vuot validation size. Bao cao equal-returns va server wall-clock.
   Local training hien toi uu absolute label-token NLL (da loai EOS), trong
   khi server core toi uu class-normalized NLL. Day la hai objective khac
   nhau; them ablation local class-NLL cho tat ca methods de kiem tra lieu
   loi ich chi den tu server bu dap objective mismatch.
5. Freeze winner tren development theo mean accuracy tren tat ca seeds,
   class NLL noninferiority margin 0.005. Neu full khong hon diag/spectral
   thi dung claim "cross-component interaction improves accuracy".
6. Dang ky confirmation range chua tung evaluate va 6 seeds moi truoc chay.
   Muon claim accuracy breakthrough: paired accuracy delta CI95 lower > 0
   truoc ca Spectral va whole-update calibration control; point gain >=1 pp,
   NLL NI, harmful magnitude khong tang qua budget da khoa; bao cao costs.

Neu core chi giam NLL ma khong doi accuracy, ket luan dung la cai thien
probabilities. Neu extra server gradients giai thich het loi ich, can them
server-calibration training control cung chi phi truoc claim FedLoRA novelty.
Khong co co so de hua dominate moi task/backbone hay official AlignFed.

## Kiem thu

Unit tests kiem tra full-core rotation vs diagonal, weighted microbatch
equivalence, trust radius, zero-radius baseline va cleanup khi loss NaN.
Full repository regression tests pass truoc GPU smoke. Ket qua GPU smoke se
duoc ghi rieng; prototype chua co accuracy confirmation.

## GPU smoke da chay (2026-09-09)

Nguon: outputs/rift_core_smoke, clean commit
70c1985bbcbad9190bab69fe01c00e2058d4778c. Qwen 1.5B 4-bit, QNLI development
offset1024, seed5101, 4 warmup + 2 measured returns, 24 eval examples.

| Method | Accuracy % | Class NLL | Harmful % | Runtime s | Peak torch allocated GiB |
|---|---:|---:|---:|---:|---:|
| Spectral Filter | 66.667 | 0.491231 | 50.0 | 20.45 | 2.1895 |
| RIFT-Diag | 66.667 | 0.487835 | 50.0 | 57.26 | 2.1895 |
| RIFT-Core | 66.667 | 0.472861 | 0.0 | 34.42 | 2.1895 |

Ca ba cung baseline accuracy 66.667%, class NLL 0.494557. Khong co late event,
client-return coverage chi 50%; harm 50% chi la 1/2 updates. Runtime la so do
mot lan (khong gom toan bo startup), khong du de ket luan core nhanh hon diag.
Ca hai repaired candidates duoc gate chon scale1. Full core off-diagonal norm
trung binh ~0.400 va ~0.187 tren hai events, nen co che cross-component da
thuc su hoat dong. Day chi la integration pass va tin hieu NLL; accuracy hoa.

Trong artifact smoke commit 70c1985, retained_fraction mo ta positive mask
truoc repair. Ban code sau smoke da sua metric nay thanh so compact columns
cua accepted update / raw innovation, va them core_positive_filter_rank,
core_candidate_rank, core_accepted_update_rank, core_server_state_rank. Day
khong phai bandwidth saving: client da gui update truoc khi server repair.
Doc core_* diagnostics va route de kiem tra repair. Diagonal/full dung extra
gradient passes, Spectral khong co.
