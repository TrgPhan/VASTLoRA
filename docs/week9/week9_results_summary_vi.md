# Week 9: tong hop tien do va ket qua

Cap nhat 2026-09-12. Protocol v3; trang thai: FIXED_AND_PREFLIGHT_TESTED / EMPIRICAL_PENDING.
Theo yeu cau, dot hoan thien nay KHONG chay them job train Qwen1.5B dai.
Chi chay unit/integration tests bang tiny random Qwen+PEFT, dry-run va data preflight.

## Doi chieu dau viec guide

| Dau viec | Trang thai | Bang chung / phan con lai |
|---|---|---|
| Mot generative/instruction task | Code + audit xong | Dolly short QA, open_qa/closed_qa |
| Token NLL va perplexity | Code + tiny integration xong | Response-only, gom EOS; tong loss / tong tokens |
| Sequence metric | Code + tests xong | Greedy ROUGE-L, exact match, token-limit rate |
| Exact/freshness/whole-gate controls | Wiring + integration xong | raw/freshness/alignfed_calibration; khong gan nhan full FedEx/AlignFed |
| RIFT / Diag / Core | Tai su dung method, integration xong | Khong doi transport/filter/gate/core math |
| Split va matched setup | Audit du lieu that xong | Source/group IDs, context-disjoint 5 roles, same eval giua seed |
| Jobs co the chay/resume | Code + tests xong | 36 dev, 72 confirmation; manifest/config/CSV validation |
| Kaggle T4 x2 | Notebook da chuan bi | Hai independent workers; chua test train dai tren Kaggle |
| Ket qua NLL matrix day du | CHUA CHAY | Can smoke v3, development, freeze, confirmation |
| Exit criterion / thesis GO | CHUA KET LUAN | Tiny tests va smoke khong thay cho bang chung empirical |

## Ket qua smoke da co, protocol v1

Nguon: `outputs/week9_generation_smoke_local/`, tung
`dolly_short_qa/noniid_high_staleness/<method>/<method>_seed9001/result.json`.
Qwen2.5-1.5B-Instruct NF4, seed9001, 1 warmup + 5 measured returns, 4 eval examples,
client pool32, gradient/gate/monitor2/2/2. Checkout dirty: chi la diagnostic.

| Method | Token NLL | Perplexity | ROUGE-L | Harmful | Late harmful |
|---|---:|---:|---:|---:|---|
| raw | 4.028182 | 56.1587 | 0.177773 | 0% | N/A |
| freshness | 4.030312 | 56.2785 | 0.177773 | 0% | N/A |
| alignfed_calibration | 4.028743 | 56.1903 | 0.176786 | 0% | N/A |
| RIFT | 4.029942 | 56.2576 | 0.176786 | 0% | N/A |
| RIFT-Diag | 4.025986 | 56.0355 | 0.177773 | 0% | N/A |
| RIFT-Core | 4.012312 | 55.2745 | 0.177773 | 0% | N/A |

Backbone token NLL chung: 4.037046. Ca 6 job xong, khong OOM tren GPU local4GB.
Nhung day KHONG phai ket qua victory: n=1 seed, chi4 cau/51 reference tokens,
0 late events, acceptance100%, exact match0, generation-limit100% cho ca6 method.
N/A late harmful vi mau so bang0, khong phai bang chung late harmful=0%.
Runtime runner khoang49-121s/job; peak allocated1.94-2.10GiB, khong bao gom moi
chi phi CUDA/desktop/driver va khong du bao full matrix.

## Protocol va sua loi v3

- Prompt dung pinned Qwen chat template; khong dua reference vao generation prompt.
- Reference cap64 gom EOS, generation cap128 rieng; dung ca stop tokens cua model.
- Tiep tuc log generation-limit rate: tang cap KHONG dam bao het bi cat cau.
- Reserve context groups rieng giua client/gradient/gate/monitor; group split
  train/dev/test va eval seed co dinh duoc giu.
- Analyzer tinh lai NLL/PPL/harmful tu CSV, kiem tra warmup/late coverage,
  sample manifests, same schedule, same starting backbone va fixed eval set.
- Skip chi cho completed artifacts khop clean commit/config. Partial run archive
  va restart tu dau khi explicitly --retry-incomplete; OOM dung, khong doi budget.
- Primary target rift_core khai bao truoc; secondary targets khong tu duoc GO.

V1 smoke KHONG duoc gop vao v3, vi prompt va sample eligibility/reservation da doi.
V2 co loi API tokenizer, chua co training result hop le. V3 chua train;
cac con so o bang tren KHONG duoc gan cho v2/v3.

Audit v3: 15011 source, excluded category9496, duplicate49, overlength2391;
giu train2131 / dev456 / held-out488. Data preflight kiem tra du512 client
examples +24/48/48 calibration cho tung seed/regime. Full trace du kien15 late
events trong64 measured returns, so voi0 late events o smoke cu.

## Correction sau review

1. BatchEncoding bi xu ly nhu list token: da sua return_dict=False, validate flat
   integer IDs, test Qwen tokenizer that va tiny PEFT loss/backward. Chi loi
   TokenBudgetError moi duoc tinh la overlength; loi tokenizer khac phai fail.
2. CSV thieu cot gay AttributeError: da kiem tra schema cua ca2 eval CSV va
   events.csv, chuyen loi thanh issue de analyzer tiep tuc bao cao.
3. runs.csv cu con sot khi0 valid runs: da luon xuat lai bang, bang rong van
   co header va khong chua ket qua cu. Test tren ca3 target reports.

Audit v2 cu2731/594/611 bi rut lai, khong dung lam bang chung. Ma tran doi ten
sang v3 va output root moi; ngan sach, seed va thuat toan method khong doi.

## Lich su kiem tra v2

- Truoc review co100 tests pass: Week9 generation/analyzer/launcher/notebook, task classification,
  shared objective, core repair, matrix runner, paper-baseline integration va Spectral v3.
- Them27 tests Week9 generation/analyzer/launcher pass trong clean detached
  checkout `188349cd2e828c4f9d723dc134bac4599f0a17b7`.
- Cac test tren dung tokenizer gia nen KHONG du de xac nhan chay that. Danh gia
  CODE_READY cua v2 da duoc rut lai. V3 them test tokenizer that bat buoc tren Kaggle.

## Kiem tra v3

- Regression ngay2026-09-12: 140 tests pass, gom ca real-tokenizer tests (khong skip).
- Clean checkout `3f9c5b226298996419308d67d493bcf35f506d8b`: them48 tests
  real-tokenizer/analyzer/launcher pass voi REQUIRE_WEEK9_TOKENIZER=1; tao du72
  confirmation job configs qua --plan-only. Notebook pin commit nay.
- Tokenizer Qwen that: flat IDs, prompt masking, response/EOS, overlength filter;
  tiny PEFT teacher-forcing NLL va backward. Khong tai pretrained model weights.
- Schema tests bo tung29 cot bat buoc giua events va2 eval CSV; them CSV rong/
  malformed, kiem tra analyzer ghi issues va launcher nhan IncompleteRunError.
- Export tests chay lai ca3 target khi run hop le tro thanh invalid: runs.csv rong
  co header, khong con ket qua cu.
- Notebook pass nbformat validation; 5 code cells compile, execution_count=null,
  khong co output training gia; RUN_TRAINING=False.
- Dry-run du36 development +72 confirmation, tat ca15 measured late events.
- Data-only preflight cho ca3 dev seeds va6 confirmation seeds, moi seed ca2 regimes;
  du512 client examples,24/48/48 calibration va128/256 eval examples.
- Analyzer moi doc lai du6 smoke v1: SMOKE_ONLY,0 missing,6 dirty-provenance warnings.
  Khong doi chung thanh clean confirmation hay che generation-limit100%.

Lenh regression (khong tai/train Qwen1.5B; integration dung tiny random models):

```powershell
python -m pytest tests/test_week9_real_tokenizer.py tests/test_week9_generation.py tests/test_week9_analysis.py tests/test_week9_launcher.py tests/test_week9_notebook.py tests/test_kaggle_3b_tasks.py tests/test_scale_objective.py tests/test_core_repair.py tests/test_week8_matrix_runner.py tests/test_paper_baseline_integration.py tests/test_spectral_v3_confirmation.py -q
```

Artifacts audit v3: `outputs/week9_v3_preflight_development/data_audit.json`
va `outputs/week9_v3_preflight_confirmation/data_audit.json`.
Notebook pin runtime commit da sua loi, khong checkout main dong. Ket qua chay tren
mot commit khac khong tu dong duoc skip/import vao cohort cua notebook.

## Chay tiep va doc ket qua

Xem [protocol va runbook](week9_generation_protocol_vi.md) va
`notebooks/kaggle_qwen_1_5b_rift_week9_generation.ipynb`.
Notebook mac dinh RUN_TRAINING=False. Thu tu: smoke v3, dev36, review/freeze,
confirmation72. Khong chon seed dep, khong tune theo held-out.

Primary: paired CI95 cua NLL(RIFT-Core)-NLL(control), voi moi control va regime.
Upper <=0.05 nats/token, du6 paired seeds, >=8 late events/run va acceptance>=0.5.
Can xem them harmful/late harmful, ROUGE-L, generation-limit, runtime va memory.
PASS_WEEK9_NLL_GATE_ONLY khong co nghia RIFT thang Acc hay toan bo thesis GO.

Gioi han quan trong: matrix nay TRAIN tren generation task moi. No chua do
catastrophic forgetting cua adapter da TRAIN classification o Week8. Neu muon
claim literal "classification gain khong lam hong generation", can mot frozen
checkpoint cross-task probe rieng. Chua co bang chung nay thi khong danh dau
exit criterion toan dien la da xong.
