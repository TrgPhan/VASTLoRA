# Week 9: tong hop tien do va ket qua

Cap nhat 2026-09-11. Trang thai: CODE_READY / EMPIRICAL_PENDING.
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
| Ket qua NLL matrix day du | CHUA CHAY | Can smoke v2, development, freeze, confirmation |
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

## V2 sua nhung gi

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

V1 smoke KHONG duoc gop vao v2, vi prompt va sample eligibility/reservation da doi.
V2 chua co training result; cac con so o bang tren KHONG duoc gan cho v2.

Audit v2: 15011 source, excluded category9496, duplicate49, overlength1530;
giu train2731 / dev594 / held-out611. Data preflight kiem tra du512 client
examples +24/48/48 calibration cho tung seed/regime. Full trace du kien15 late
events trong64 measured returns, so voi0 late events o smoke cu.

## Chay tiep va doc ket qua

Xem [protocol va runbook](week9_generation_protocol_vi.md) va
`notebooks/kaggle_qwen_1_5b_rift_week9_generation.ipynb`.
Notebook mac dinh RUN_TRAINING=False. Thu tu: smoke v2, dev36, review/freeze,
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
