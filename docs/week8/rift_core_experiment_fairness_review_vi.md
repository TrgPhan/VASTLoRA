# Review RIFT-Core: fairness, bugs, coverage va runtime

Ngay: 2026-09-09. Code duoc review: cf98fdd2 (main co tracked deletion
`tmp_pyproject_view.txt` tu truoc). Review nay khong sua logic model/config.

## Trang thai ap dung sau review

Ban lam viec hien tai da ap dung cac sua doi sau; can commit clean truoc khi
chay confirmation:

- FedRot dung cung asymmetric LoRA factor convention tai load va aggregate;
  co regression test no-op voi partial server weight.
- Result schema v5 tach `worst_step_loss_increase` tren moi measured event va
  `worst_late_step_loss_increase`; ca hai clamp tai 0. Harmful flag dung
  `harm_epsilon=1e-6` da khoa trong matrix. Cumulative harm cung bo cac delta
  khong vuot epsilon; verdict cho phep safety tie neu ca hai ben deu bang 0.
- Matrix resume doi chieu exact `git_commit`, schema, matrix/config hash va
  clean-worktree flag.
- Model Qwen 1.5B va GLUE revision duoc pin. Model-name override bo revision
  cu, hoac nhan `--model-revision` moi.
- Monitor loss cache theo server version bo current-state forward lap lai.
- Development v2 dung 512 eval examples/task; confirmation v2 dung 1024.
  Raw/freshness/FedRot va RIFT-Diag da duoc them vao cohort chinh.
- Runner fail neu eval bi truncate am tham hoac split co label ngoai range.
- Confirmation v3 thay positive-only `spectral_filter` bang Spectral Surgery
  operator port: per-example mean-absolute sensitivity, smooth spectral
  reweight va L1 preservation. FedRot v3 luan phien A/B, ho tro soft rotation
  va ep `det(R)>0`. Chi tiet fidelity va lenh confirmation nam tai
  `docs/week8/competitor_fidelity_confirmation_runbook_vi.md`.

Phan findings ben duoi la audit trail tai thoi diem phat hien. Muc P1/P2 da
fix nhu tren khong con la known bug trong code hien tai; caveat ve fidelity
paper, server-compute control va do bao phu khoa hoc van con hieu luc.

## 1. Ket luan va pham vi

Co tin hieu development that: Core/Diag cai thien du doan tren cung eval
examples va cung client schedule. Chua du de claim thang official Spectral
Surgery/AlignFed, chua chung minh loi ich rieng cua asynchronous repair, va
chua la held-out GO. Khong co bang chung tu audit nay rang accuracy bi tinh
sai hoac label cua eval bi dua vao train. Van co bugs va confounds can xu ly.

Da doc runner, matrix builder, core repair, component scoring, PEFT bridge,
simulator, analyzer, config va 45 result.json cua 3 cohort hoan tat:

- `outputs/rift_core_dev_1_5b`: QNLI, 15 runs, dirty provenance.
- `outputs/rift_core_mnli_local_4gb`: MNLI-m, 15 runs, clean provenance.
- `outputs/rift_core_remaining_tasks_local_4gb`: SST-2, 15 runs, clean.
- MNLI-mm trong cohort remaining chua co completed result.json.

## 2. Findings theo muc do

### P1: FedRot port tao thay doi khi client khong hoc

`src/riftlora/scale/peft_bridge.py:92` load client bang B=U*S, A=V^T/scaling.
Nhung aggregation tai :182 va helper :256 dung server factors balanced
B=U*sqrt(S), A=sqrt(S)*V^T/scaling. Orthogonal rotation khong sua duoc
chenh lech scale nay, nen averaging tao artificial update.

Da reproduce voi rank=1, scaling=1, server/client deu bieu dien diag(4,0),
weight=0.5, client B=[4,0]^T va A=[1,0]. Ket qua aggregate la diag(4.5,0),
lech 12.5% du client khong hoc. Day la integration issue cua port voi bridge,
khong phai ket luan FedRot paper sai. Test hien tai dung weight=1 nen khong
bat duoc loi averaging nay (`tests/test_peft_bridge.py:89`).

Can dung factor convention nhat quan tai dispatch va aggregation, test
no-op tai weight=0.25/0.5 va rotated equivalent factors. Chi su dung FedRot
lam evidence sau khi fix/revalidate. Loi nay KHONG duoc thuc thi trong 45
runs Core/Diag/Spectral/RIFT/whole-update control vua audit.

### P1: Thieu doi chung extra server training va full-core novelty

Client chi hoc 1 example/return: local_steps=1, local_batch_size=1,
gradient_accumulation_steps=1 trong base config. Core/Diag them 3 optimizer
steps qua 24 calibration examples moi measured return (:433 runner).
16 returns tuong ung 1,152 example exposures rieng cho repair, so voi 16
client exposures trong measured phase; 24 examples nay duoc lap lai, khong
phai 1,152 examples doc lap. Spectral co 24-example scoring pass moi return
nhung khong co 3 repair steps. Core/Diag con dung gate va candidate selection.

So sanh equal-returns la hop le NEU khai bao server labels/compute; no chua
isolate contribution cua stale repair. Can server-only calibration LoRA,
Spectral + same-budget server fine-tuning, va Core vs Diag cung gate/steps.
Local loss la label-token NLL (:1063), server loss la class NLL (:1378):
can ablation local class NLL cho tat ca methods de tach objective mismatch.

Confirmation matrix :120 bo Diag, du SST-2 Diag dang co accuracy cao hon.
Can giu Diag trong confirmation neu muon claim full core can thiet.
Khong duoc goi 3 luot backward tren 24 examples la 3 batch-size-1 steps.

### P1: Ten baseline va tieu chi GO khong du cho claim accuracy breakthrough

`spectral_filter` (:464 runner, `scale/objective.py:151`) la positive hard
filter, khong phai full Spectral Surgery magnitude-constrained reweighting.
`alignfed_calibration` (:1754 runner) la whole-update scalar gate, khong co
day du grouping/alignment/fairness cua AlignFed. Fidelity nay da duoc ghi
trong `src/riftlora/diagnostics/competitors.py:46`.

Confirmation dung `quality_superiority=point_any` (:138 config); analyzer
:665 chap nhan accuracy HOAC NLL point estimate tot hon. Co the pass quality
du accuracy hoa va NLL chi tot hon rat nho. Cac gate safety/NI khac van phai
pass: day khong phai chi can NLL giam la tu dong GO. Tuy nhien no khong
thuc thi tieu chi accuracy CI95 lower > 0 va gain >=1 pp trong research plan.
Can tach verdict safety va verdict accuracy, dang ky truoc khi xem held-out.

### P2: Skip khong kiem tra code identity

`scripts/run_week8_classification_matrix.py:219` chi check schema, method,
seed, matrix hash, config hash va dirty flag. Da reproduce bang payload
co `git_commit='not-the-current-code-commit'`: function van return True.
Khi sua code ma khong doi config, ket qua cu co the bi skip nham.
Analyzer check cac runs cung commit, nhung khong chung minh do la code dang
duoc yeu cau. Can expected implementation hash/commit hoac audited
compatibility allowlist; khong nhat thiet rerun khi chi sua documentation.

### P2: Worst-step metric sai pham vi va co the am

`scripts/run_kaggle_3b.py:879` tinh `max(late_deltas)` cho ten
`worst_step_loss_increase`. No bo qua early harmful va khong clamp >=0.
SST-2 whole-update control seed5101: reported=0, nhung max tren moi measured
event=0.00038874149322509766. Seed5102: reported=-0.0001570582389831543,
max that=0.00006648898124694824. Can tach worst_all_step va worst_late_step,
giu signed delta rieng neu can. Co the tinh lai tu events.csv, khong train lai.

Harmful flag dung epsilon=1e-12 (:689), nho hon nhieu numerical variation
co the co cua float16/4bit. Khong tu dong coi moi tiny increase la meaningful
harm. Bao cao signed delta, positive magnitude, va sensitivity voi epsilon
duoc khoa truoc; khong chon epsilon sau khi xem method nao thang.

### P2: QNLI provenance va do bao phu

Ca 15 QNLI runs co git_worktree_dirty=true, commit 3eca98db. Cung config hash
khong chung minh cung uncommitted implementation. Giu lam exploratory;
confirmation can clean code. MNLI-m dung e518a203, SST-2 dung cf98fdd2.
Khong gop ba cohort thanh mot strict same-commit verdict.

Hien chi 4 clients, 16 measured returns, 4 late events/run, 96 eval examples,
label-shard/high-staleness/rank heterogeneity cung luc. Schedule co dinh theo
compute_times, khong do thoi gian server repair/network; hop le cho replay
operator comparison, chua chung minh end-to-end FL throughput.

## 3. Nhung kiem tra da pass

- 79 tests pass tren core repair, objective, bridge, task evaluation, matrix,
  competitor analyzer, async simulator va competitor transports.
- 9 task-seed pairs: 5 methods/pair cung config fingerprint, partition and
  calibration diagnostics, return trace, eval text va gold labels.
- Reconstruct split tu cached GLUE Arrow va cung stratification seed:
  khong co exact normalized content overlap giua client/gradient/gate/
  monitor/eval trong 9 pairs. Normalize case va whitespace. Chua audit
  semantic near-duplicates hoac backbone pretraining contamination.
- Prompt builders dung input texts; candidate likelihood causal shift,
  prediction argmin label NLL, class probabilities softmax(-label NLL).
  Gold label duoc dung de tinh metric, khong duoc chen vao prompt.
- Cac label strings hien tai deu 1 token voi tokenizer Qwen 1.5B cached.
  Eval truncation: SST-2 0/96, QNLI 0/96, MNLI-m 2/96.
- Khong thay historical result.json voi explicit shuffle_seed=314159 va
  config range overlap confirmation windows 704:872 SST-2 / 1536:2048 NLI.
  Day la config-range audit gioi han, khong chung nhan tat ca notebook/old
  outputs/missing shuffle metadata hay pretraining deu chua tung thay data.
- Reproduce bugs o muc Python/CPU; khong train lai model trong review nay.

## 4. Accuracy cao co bat thuong khong?

Mean final accuracy (%), 3 development seeds 5101-5103, 96 eval examples:

| Task | Backbone truoc train | Spectral filter | RIFT cu | Whole-update gate | Diag | Core |
|---|---:|---:|---:|---:|---:|---:|
| SST-2 | 88.54 | 88.89 | 88.89 | 88.54 | 92.01 | 91.67 |
| QNLI | 71.88 | 74.65 | 74.65 | 73.96 | 79.86 | 80.90 |
| MNLI-m | 63.54 | 65.97 | 65.63 | 61.81 | 66.67 | 74.31 |

Mot cau = 1.0417 pp/seed. SST-2 Diag so Spectral: seed5101 sua dung 4,
sua sai 0; seed5102 2/0; seed5103 3/0. Core: 4/1, 3/1, 3/0. Day la
du doan that, nhung sample rat nho. Cung 96 eval examples tren 3 seeds
khong bien thanh 288 unique test examples. CI theo seeds chi do variability
training/calibration tren tap eval co dinh, chua do uncertainty ve examples.

Them repair gradient, discriminate class loss va gate selection la nhung
co che hop ly de tang accuracy; chua biet contribution rieng cua tung phan.
Khong co co so noi ket qua la fabricated, cung khong co co so noi da thang
tat ca doi thu. Tuned development duoc phep, cherry-pick seed/test window
roi goi la held-out superiority thi khong hop le.

## 5. Doi chung uu tien

1. Frozen backbone, raw/freshness async; server-only calibration LoRA.
2. Spectral + ordinary server calibration training cung examples, steps,
   gate va comparable server compute. Hien thieu doi chung quan trong nay.
3. Diag/full, radius=0 voi cung gate, no-delay trust radius, gate-only;
   giai thich them parameter count va chi phi, khong chi so steps.
4. Faithful Spectral Surgery port, FedRot sau fix, FedEx neu giu exact residual.
5. Full AlignFed chi khi co verified implementation. Khong doi ten scalar
   gate thanh full AlignFed. FedLAW-inspired trainable scalar aggregation la
   mot control hop ly cho server optimization; neu port phai ghi adaptation.

FedEx trong runner hien la alias raw (`scale/coordinator.py:95`) va van
rank-project state o aggregator. Khong goi no full exact-residual FedEx-LoRA.
Khong can port tat ca paper neu setting/assumption khong khop; chon vai
doi thu faithful va mot bo controls du de isolate contribution.

## 6. Experiment can bo sung

Uu tien hoan tat 4 eval slices, tang power va controls truoc mo rong task.
MNLI-m/mm chung training task/domain evaluation, khong phai hai FL training
problems doc lap. Hien runner train lai tung split: sau nay nen luu compact
checkpoint va evaluate ca hai split tren cung trajectory.

- Freeze config va 6 seeds confirmation; bao cao tat ca seeds, final metric,
  mean/dispersion va paired confidence intervals. Best seed chi mo ta.
- QNLI/MNLI eval 512 la buoc dau; 1,000-2,000 examples chua tung dung co ich
  neu can resolve gain 1 pp, can power/paired-discordance analysis de chon N.
  SST-2 window 168 co resolution 0.595 pp, van rong CI. Khong mo rong vao
  development examples roi goi toan bo la unseen confirmation.
- IID + homogeneous + low delay; non-IID + low delay; IID + high delay;
  non-IID + high delay; rank homogeneous/heterogeneous de tach cac yeu to.
- Hoan vi rank/label distribution/compute speed giua clients, them clients
  va longer horizon 64-128 returns; bao cao accept/utility theo client.
- Local_steps 1 vs 4/8 hoac nhieu sampled examples cho MOI method. Client
  compute matched va server budget matched la hai comparison rieng.
- Calibration size/shift ablation: balanced proxy data la assumption manh
  khi client label-shard. Test proxy lech distribution va monitor shift.
- HANS: evaluate MNLI-trained checkpoint de kiem tra syntactic shortcuts;
  explicit mapping entailment vs non-entailment cho 3-class probabilities.
- ANLI R1: optional challenge NLI moi, dung de stress generalization.
  Kho hon khong dam bao se phan biet methods tot hon; co the gap floor effect.
- Chi them generation task neu thesis claim general generative LLM quality;
  khi do can metric generation, khong dung class-NLL de suy ra chat quality.

Metrics chinh: final Acc/Class NLL, paired deltas/CI, positive harm magnitude,
late harmful voi numerator/denominator, acceptance va client coverage, server
compute/wall-clock va peak memory. GO accuracy va GO safety phai tach ro.

## 7. Tai sao lau va toi uu nao it rui ro?

Mean minutes/run tu runtime_seconds (KHONG bao gom model load/base eval):

| Task | Spectral | Whole-update gate | Diag | Core |
|---|---:|---:|---:|---:|
| SST-2 | 5.43 | 10.79 | 19.16 | 14.84 |
| QNLI | 5.23 | 8.92 | 14.72 | 13.98 |
| MNLI-m | 5.96 | 14.19 | 19.24 | 22.00 |

Core measured return voi active candidates: 24 scoring microbatches backward,
3*24 repair microbatches backward, (current + comparator + 3 scales)*48=240
gate microbatches forward, 2*48=96 monitor microbatches forward, them local
train/compact SVD/state load. Moi classification microbatch lap prompt cho
2 labels (SST/QNLI) hoac 3 labels (MNLI). Full core chi r*r parameters nhung
gradient van backprop qua transformer. Day la nguyen nhan co cau; chua co
stage profiler nen khong gan % thoi gian cho tung stage hay do loi hardware.
SST-2 Diag cham hon Core trong lan do nay khong chung minh intrinsic cost.

Uu tien toi uu sau audit, giu objective/data/candidate set:

1. Cache tokenization/collated CPU tensors cho gate/monitor/eval; gradient
   batches hien da cache. Khong cache logits qua cac server states khac nhau.
2. Immediate async: current monitor state bang accepted state event truoc;
   reuse loss bang exact state identity. Rejected identical state khong can
   evaluate lai. Tiet kiem gan nua monitor forwards, KHONG phai nua total.
3. Single-token label fast path: tinh prompt 1 lan, gather label logits.
   Causal model hien lap cung prompt 2/3 lan. Kiem tra same prompt budget,
   EOS diagnostic va gradient/decision equivalence; multi-token fallback.
4. Reuse MNLI training checkpoint cho m/mm va baseline eval cache keyed by
   model/tokenizer revision, prompt, truncation, quantization, dataset IDs.
5. Tang eval/gradient microbatch khi VRAM cho phep; weighted gradient mean
   giu math objective nhung floating point/order co the doi gate o bien.
   Can paired replay, max loss/gradient difference va same route checks.
6. Profile model loading, train, scoring, repair, gate, monitor, SVD/I/O;
   report both end-to-end va algorithm time. Memory >4GB co the cho batch
   lon hon; khong hua speedup hay ket qua bitwise truoc benchmark.

Khong giam calibration, so repair steps, candidates hay doi precision roi
goi do la optimization khong doi ket qua. Khong dung parallel jobs tren GPU
4GB; device_map hien {"":0}, khong tu dong chia model sang GPU thu hai.

## 8. Nguon primary da doi chieu

- Spectral Surgery: https://arxiv.org/abs/2603.03995
  Gradient-guided singular reweighting, magnitude constraint; khac hard mask.
- AlignFed: https://arxiv.org/abs/2606.08197
  Grouping, semantic alignment, fairness aggregation; scalar gate khong du.
- FedRot official code: https://github.com/haoran-zh/FedRot-LoRA
- FedEx-LoRA: https://arxiv.org/abs/2410.09432
  Exactness bang residual vao frozen weights, khong chi truncate raw delta.
- FedLAW: https://proceedings.mlr.press/v202/li23s/li23s.pdf
  Proxy-data aggregation learning va Server-FT control da co tien le.
- HANS: https://arxiv.org/abs/1902.01007
- ANLI: https://aclanthology.org/2020.acl-main.441/

Review literature nay phuc vu competitor selection, khong la exhaustive
novelty clearance. Ket luan can giu: promising development, can sua bugs va
bo sung matched controls truoc khi khang dinh research/held-out GO.
