# Week 9: generative/NLL matrix

Ngay: 2026-09-12. Protocol v3. Pham vi: buoc tiep theo cua guide RIFT, sau classification Week 8.

Trang thai: CODE_READY, full development/confirmation CHUA CHAY theo yeu cau.
Xem [tong hop Week 9](week9_results_summary_vi.md) de phan biet smoke cu va protocol hien tai.

## Vi sao can buoc nay

Week 8 co tin hieu chat luong tot: RIFT-Core co class NLL trung binh thap nhat
tren bon view SST-2, QNLI, MNLI-m, MNLI-mm. Class NLL la xac suat chuan hoa
giua cac nhan ung vien; no khong do xac suat cua toan bo cau tra loi tu do.
RIFT-Diag/RIFT goc van tot hon Core tren mot so safety metrics. Cac run dirty
va 7 late events/run cua v3 cung khong du de declare strict thesis GO.

Week 9 kiem tra response NLL va chat luong sinh tu do. MNLI-m/mm la hai view
cua cung task, khong phai hai bang chung hoan toan doc lap. Week 9 them mot
loai bai toan moi, chu khong chi them cot accuracy vao bang cu.

## Dataset va splits

Nguon: [Databricks Dolly-15k dataset card](https://huggingface.co/datasets/databricks/databricks-dolly-15k).
Day la instruction dataset do Databricks cong bo, license CC BY-SA 3.0.
Revision khoa: `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`.

Pilot chon `open_qa` va `closed_qa`. Filter chi dua tren category, prompt rong,
trung prompt va token length; khong dua tren ket qua cua method nao.
Qwen tokenizer: prompt toi da 192 tokens, response gom EOS toi da 64 tokens.
Prompt dung chat template cua tokenizer Qwen da pin, chi dua instruction/context
vao user turn, mo assistant turn bang add_generation_prompt. Teacher forcing va
greedy generation dung cung prompt token IDs. Reference ket thuc bang im_end/EOS;
generation dung hop cua EOS tokenizer va danh sach stop tokens trong generation config.
max_new_tokens=128 tach rieng khoi reference cap64; khong ep model viet dung reference.
Day la sua protocol chung cho ca 6 method, khong phai tune rieng de RIFT thang.
Vuot ngan sach thi loai mau truoc chia/sampling; khong cat cau tra loi roi
gia vo dang do NLL toan cau. Ket luan chi ap dung short QA, chua bao phu
instruction dai, reasoning dai hay summarization.

Split theo hash cua context da chuan hoa (hoac instruction neu khong co context),
voi salt 9000 va ti le 70/15/15. Cac cau hoi dung chung context o cung split.
Dedup theo instruction+context. Chua loai duoc moi paraphrase/semantic duplicate,
va khong xac minh duoc dataset nay co trong pretraining cua backbone hay khong.

Audit da chay tren du lieu that:

| Hang muc | So mau |
|---|---:|
| Nguon | 15011 |
| Loai category khac | 9496 |
| Trung prompt | 49 |
| Qua ngan sach token | 2391 |
| Train duoc giu | 2131 |
| Development duoc giu | 456 |
| Held-out duoc giu | 488 |

Correction v3: v2 dem nham len(BatchEncoding)=2 thay vi so prompt tokens.
V3 yeu cau return_dict=False va kiem tra flat integer IDs, sau do collate thu
voi tokenizer that cho ca3 splits ngay trong --prepare-only. Audit v2 cu
2731/594/611 khong hop le; KHONG import audit/result v2 vao cohort v3.
Model, seeds, LoRA method math va cac ngan sach khai bao khong thay doi.

Moi run reserve gradient/gate/monitor tu train truoc, roi lay client pool.
V2 con loai context groups da vao mot role khoi cac role con lai. Nhieu cau hoi
dung chung context khong duoc trai giua client/gradient/gate/monitor/eval.
Result ghi source IDs va group IDs cho ca nam nhom. Eval shuffle seed 314159 giu cung
cau hoi giua cac seed; seed training chi doi sampling/partition/model RNG.
Grouping ngan leakage tu context chinh xac; khong la bang chung khong co
semantic overlap hay contamination tu pretraining.

## Protocol du kien truoc matrix

| Tham so | Development | Confirmation |
|---|---|---|
| Model | Qwen2.5-1.5B-Instruct NF4 | Giong dev |
| LoRA | q_proj/v_proj, rank [2,4,8,4] | Giong dev |
| Client pool | 512 | 512 |
| Calibration/gate/monitor | 24/48/48 | 24/48/48 |
| Warmup/measured returns | 8/64 | 8/64 |
| Local steps/batch/accumulation | 1/1/1 | 1/1/1 |
| LR/server weight | 0.0002/0.5 | Giong dev |
| Eval examples | 128 validation | 256 test |
| Seeds | 9101-9103 | 9201-9206 |
| Regimes | category-shard non-IID va IID; ca hai high staleness | Giong dev |
| So jobs | 36 | 72 |

Partition category-shard la non-IID theo loai instruction, khong phai chia
nhan dap an dung/sai. Scheduler immediate async, compute times [1,2,5,10].
Dry-run cho 15 measured late returns/run. Cac method co cung lich return,
client examples, calibration allocation va eval; chi thay phep xu ly update.

`raw` la exact product-space addition truoc khi rank-cap chung; khong phai
FedEx-LoRA residual aggregation. `alignfed_calibration` la whole-update gate,
khong phai full AlignFed. Raw/freshness khong su dung calibration gradient,
nhung van reserve cung nhom de client pool khop giua methods.

## Loss va metrics

- Train: teacher-forcing cross entropy tren response tokens gom EOS.
- Gradient scoring/core repair/gate/monitor: mean per-example response NLL,
  giup paired gate so sanh tung instruction voi trong so bang nhau.
- Primary eval: `token_nll = sum(response_token_losses) / sum(response_tokens)`.
- `perplexity = exp(token_nll)`; khong lay exp cua mean per-example NLL roi
  goi do la corpus perplexity. Bang tong hop giu ca per-seed PPL.
- ROUGE-L: greedy generate tu prompt KHONG co reference, toi da 128 new tokens,
  scorer tu package rouge-score. Exact match va generation-limit rate di kem.
- ROUGE-L/exact match tren QA tu do la lexical diagnostics, khong thay cho
  factual correctness. Khong tu dong gan confidence GO theo ROUGE-L.
- Harmful/late harmful: tang mean response NLL tren monitor rieng, epsilon 1e-6.
- Runtime va peak CUDA memory duoc ghi. Runtime runner loai baseline eval/model
  loading theo quy uoc cu, nen khong phai toan bo chi phi wall-clock job.

Accuracy/class NLL duoc ghi null cho generation. Khong dien ROUGE vao cot Acc.
Khong chon checkpoint theo held-out; baseline/final deu evaluate theo lich co dinh.

## Gate va muc claim

Gate Week 9: voi moi control raw/freshness/whole-gate va moi regime, tinh
paired difference `NLL(target) - NLL(control)` tren tat ca 6 seed.
Can tren CI95 hai phia (Student t) <= 0.05 nats/token de pass non-inferiority.
Margin 0.05 tuong ung khoang 5.13% ty le PPL, la nguong thiet ke truoc matrix,
khong phai gia tri suy ra tu cac ket qua thang. Bao cao ca interval va tung seed.
Voi 6 seed, CI van phu thuoc gia dinh paired differences va co the rong.

Can them du 72 runs, data/config khop, mot clean implementation commit,
it nhat 8 late events/run va acceptance cua target >= 0.5 moi run.
Thieu data/provenance: INCOMPLETE_OR_UNVERIFIED; chua du CI: INCONCLUSIVE_NLL;
CI95 lower > margin: NO_GO_NLL. PASS_WEEK9_NLL_GATE_ONLY khong thay the
safety/novelty/scale gates cua thesis. Dev va smoke KHONG duoc pass confirmation.
Primary target khai bao truoc la rift_core. RIFT va RIFT-Diag van duoc bao cao,
nhung --target rift/--target rift_diag chi cho EXPLORATORY_TARGET_ONLY, khong
doi primary winner sau khi xem held-out. Generation-limit rate >0 can doc
prediction; no khong tu dong lam sai teacher-forced NLL nhung han che sequence metrics.

Luu y ve exit criterion trong guide: train tren generation task nay do kha nang
hoc generation cua method. No KHONG truc tiep chung minh adapter da train SST-2/QNLI
giu nguyen generation quality. Claim literal ve cross-task forgetting con can
evaluate cac classification checkpoints da freeze tren mot generative probe chung.

## Lenh chay

```powershell
python -m pip install -e ".[scale,dev,generation]"
python scripts/run_week9_generation.py --output-root outputs/week9_v3_dev --prepare-only
python -m pytest tests/test_week9_real_tokenizer.py tests/test_week9_generation.py tests/test_week9_analysis.py tests/test_week9_launcher.py -q
python scripts/run_week9_generation.py --output-root outputs/week9_v3_dev --dry-run
python scripts/run_week9_generation.py --output-root outputs/week9_v3_dev --plan-only
```

Smoke local (6 jobs, seed9001, 1 warmup + 5 measured returns, eval4):

```powershell
python scripts/run_week9_generation.py --smoke --output-root outputs/week9_v3_smoke --gpu 0
python scripts/analyze_week9_generation.py --input-dir outputs/week9_v3_smoke
```

Development du ma tran:

```powershell
python scripts/run_week9_generation.py --output-root outputs/week9_v3_dev --gpu 0 --gpu 1
python scripts/analyze_week9_generation.py --input-dir outputs/week9_v3_dev
```

Sau khi review dev va freeze code/config, chay confirmation tren clean checkout:

```powershell
python scripts/run_week9_generation.py --phase confirmation --output-root outputs/week9_v3_confirmation --prepare-only
python scripts/run_week9_generation.py --phase confirmation --output-root outputs/week9_v3_confirmation --dry-run
python scripts/run_week9_generation.py --phase confirmation --output-root outputs/week9_v3_confirmation --gpu 0 --gpu 1
python scripts/analyze_week9_generation.py --input-dir outputs/week9_v3_confirmation --target rift_core
```

Mot GPU thi chi dung --gpu 0. --gpu 0 --gpu 1 la hai job doc lap, KHONG ghep
VRAM hai T4 thanh mot GPU32GB. Khong doi batch/rank/length khi OOM.

Co the loc `--method`, `--seed`, `--regime` de chia jobs giua may/GPU. Tat ca
selection deu phai nam trong manifest. Output cu chi skip neu config/matrix/
seed/commit va clean provenance khop. Dirty smoke da co result thi dung output
root moi khi chay lai; khong force ghi de ket qua cu. Script dung khi worker
loi/OOM, khong tu doi budget. Tren GPU 4GB chi bat dau bang smoke; chua xac
nhan full development 36 jobs tren GPU nay.

### Resume va xu ly loi

1. Chay lai dung lenh va output root: completed job du JSON + hai eval CSV + events
   duoc validate roi skip. Kiem tra ca token sums, PPL, sequence metrics, measured
   boundary, harmful/late harmful tu monitor va provenance. Chi co result.json la chua du.
   CSV thieu cot/khong parse duoc se duoc ghi thanh issue, khong crash ca analyzer.
   runs.csv luon duoc ghi lai, ke ca bang rong co header khi0 run hop le.
2. --resume-root /kaggle/input/.../week9_v3_confirmation: import completed runs dung
   config/seed/implementation tu input dataset. Khong import dirty/v1/mixed commit.
3. --max-jobs 2 chi gioi han 2 job MOI trong phien; manifest van giu du 36/72.
4. Neu bi ngat giua job: mac dinh dung va chi ro thu muc partial. Them
   --retry-incomplete de archive vao incomplete/ roi train lai seed do tu dau.
   Chua co mid-job optimizer/client/server checkpoint resume. Khong goi day la
   tiep tuc tu dung training step bi ngat.
   JSON da khop provenance nhung CSV hong/thieu cung co the archive/restart.
   JSON hong den muc khong xac minh duoc identity, hoac sai commit/config thi
   tu choi; giu nguyen artifacts va dung root moi, khong force ghi de.
5. Root co file lock de khong chay hai launcher de len cung ket qua. OOM/exit
   khac0: dung toan bo workers cua launcher, giu logs va partial artifacts,
   khong tu retry hay sua config. Ctrl+C terminate/wait cac worker con lai.
6. Sau khi sua config/code, dung output root va commit moi, khong tron cohort.

Output: matrix.json, job_plan.json (lenh va trang thai tung job), job_configs/,
logs/, cac result.json/events.csv/baseline_eval_details.csv/final_eval_details.csv,
launcher.json (wall time ca job), analysis_rift_core/{verdict.json,runs.csv,results.md}.
--prepare-only tai/tokenize dataset va test du lieu cho tung seed/regime, KHONG train.
--plan-only ghi config va command, KHONG nap model. --dry-run chi kiem tra config/trace.

### Kaggle

Dung `notebooks/kaggle_qwen_1_5b_rift_week9_generation.ipynb`, Internet on, T4 x2.
Notebook clone GitHub, checkout commit co dinh, cai dependencies va test tiny PEFT
truoc khi train. RUN_TRAINING=False mac dinh: Run All chi preflight va xuat report.
Notebook cache tokenizer that truoc tests, dat REQUIRE_WEEK9_TOKENIZER=1 de
khong skip integration test khi thieu cache. Test nay chi dung tokenizer that
va tiny random PEFT model tren CPU, KHONG tai pretrained model weights.
Khi san sang: MODE='smoke', RUN_TRAINING=True; review stop rate/predictions/NLL.
Notebook khoa Qwen2.5-1.5B-Instruct NF4; doi model can cohort/config version moi.

| MODE | So job duoc chon | Muc dich |
|---|---:|---|
| preflight | 36 plans, khong train | Audit du lieu/config full dev; RUN_TRAINING phai False |
| smoke | 6 | Seed9001, 1+5 returns, eval4; chi test pipeline/GPU |
| pilot | 6 | Seed9101, non-IID, full dev budget 8+64 returns, eval128 |
| development | 36 | 3 seeds x 2 regimes x 6 methods, eval128 |
| confirmation | 72 | 6 seeds x 2 regimes x 6 methods, eval256 held-out |

Sau smoke, dung MODE='pilot' de thu mot seed voi budget that. Pilot va development
cung root week9_v3_development va manifest 36 jobs: job pilot hoan tat hop le se
duoc skip khi chay development. Report pilot van thieu30 jobs la dung, khong phai
loi va khong duoc xem la confirmation. Smoke co root rieng, khong tron vao dev.
MAX_JOBS chi gioi han so job moi trong phien, khong loai seed khoi analysis.
Moi lan doi MODE, chay lai settings va cac cell phia sau. Muon tiep tuc o phien
Kaggle moi, attach ZIP da giai nen vao input va them root vao RESUME_ROOTS.

Chi doi MODE='confirmation' sau khi review dev va
dat CONFIRM_PROTOCOL_FROZEN=True; freeze ca config, implementation, primary target.
Model weights chi precache khi RUN_TRAINING=True. Notebook zip artifacts cuoi cung.

## Trang thai

- [x] Doc Week 9 va doi chieu limitations Week 8.
- [x] Dataset pin, group split, dedup, length filter va audit du lieu that.
- [x] Collator response-only, gradient/NLL/perplexity va greedy sequence metrics.
- [x] Tai su dung 6 method trong shared async runner.
- [x] Matrix development/confirmation va analyzer co completeness/provenance gates.
- [x] Tiny-Qwen + PEFT integration cho ca 6 method.
- [x] Chat protocol, EOS/multi-stop, context-disjoint reservations va regression tests.
- [x] Job plans, one-worker/GPU, locks, stop-on-error, validated resume/import.
- [x] Notebook va summary/runbook; mac dinh khong train.
- [ ] Smoke Qwen1.5B cua protocol v3 (smoke v1 khong duoc tai su dung).
- [ ] Full development matrix va review failure modes.
- [ ] Freeze clean implementation sau dev; full held-out confirmation.
- [ ] Ket luan empirical Week 9 NLL gate.

GPU smoke va cac bang ket qua thuc te duoc cap nhat rieng; cac o chua chay
khong duoc suy dien thanh PASS.
