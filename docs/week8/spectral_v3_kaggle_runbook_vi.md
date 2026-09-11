# Spectral post-hoc tren configuration confirmation v3

Notebook: `notebooks/kaggle_qwen_1_5b_spectral_v3_confirmation.ipynb`.
Runner: `scripts/run_spectral_v3_confirmation.py`.

## Protocol

Runner doc truc tiep `configs/rift_core_heldout_confirmation_matrix.json`,
giu tasks, regimes, seeds va training/eval settings. Chi doi method sang
`spectral_surgery_posthoc`, khai bao edit `q_proj/v_proj` va train base bang
`freshness`. Metadata cua cohort duoc doi ten de truy vet baseline bo sung.

- Qwen2.5-1.5B-Instruct NF4, model/dataset revision giong v3.
- SST-2, QNLI, MNLI-m, MNLI-mm; seeds 6101-6106: 24 jobs.
- Rank [2,4,8,4], 4 clients, label shards, compute times [1,2,5,10].
- 8 warmup + 32 measured returns; server weight 0.5; learning rate 0.0002.
- Calibration gradient/gate/monitor = 24/48/48; eval batch size 1.
- 1024 eval examples/task; SST-2 train reserved offset4096, QNLI/MNLI validation offset2048.
- Max length 192 cho SST-2/QNLI, 128 cho MNLI; training LoRA q_proj/v_proj.
- Edit mot lan sau FL voi smooth_abs va L1 preservation. Gradient edit dung
  answer-token NLL gom EOS; final quality van do bang accuracy va class NLL.

Ban nay la Spectral Surgery post-hoc duoc dieu chinh cho benchmark v3,
khong phai reproduction nguyen benchmark paper. Kiem tra moi commit giup
tranh tron ket qua Spectral cu (edit tung return) voi ket qua post-hoc moi.
Seed/eval nay da duoc xem trong v3; day la baseline bo sung, khong phai
confirmation doc lap moi hoac ly do de tune tren held-out.

## Kaggle

Bat Internet, chon T4 x2, upload notebook va Run All. Notebook clone GitHub
va checkout dung commit, cai dependencies, pre-cache model/GLUE, chay tests
va dry-run truoc khi chay 24 jobs. Moi GPU chay mot job doc lap.

`MAX_JOBS = None` chay moi job con thieu; co the gioi han so job moi moi phien.
Neu het phien, luu output thanh Kaggle Dataset, attach vao phien tiep theo,
dien `RESUME_ROOTS` bang thu muc chua cac task sst2/qnli/mnli_m/mnli_mm.
Import chi chap nhan result cung commit, matrix/config hash, seed, schema
va day du CSV. Ket qua v3 cu hay development seed7201 khong thay the cac job nay.

Resume o cap job: job da xong duoc skip; job dang do se train lai tu dau.
OOM/worker failure se dung nhom process va hien log; khong tu doi rank,
batch size, calibration hay tham so method. File lock ngan hai launcher
ghi cung output root. Chua xac nhan VRAM/runtime tren T4 that cho cohort nay.

## Ket qua

Output rieng: `/kaggle/working/spectral_posthoc_v3_supplement`.
Notebook hien 3 bang Accuracy, Class NLL, Harmful; moi task co so seed da xong.
`runs.csv` giu tung seed, Acc/NLL truoc-sau edit va delta; `summary.csv`,
`results.md`, `completion.json` va manifests/logs duoc dong goi ZIP.

Harmful/late harmful cua Spectral post-hoc mo ta TRAINING truoc edit.
Khong dung chung de claim post-hoc chan harmful returns. Tac dung truc tiep
cua edit nam o `edit_accuracy_delta_pp` va `edit_class_nll_delta`.
Pre-edit evaluation chi la diagnostic, khong chon co giu edit hay khong.
Khong ghi ket qua gia vao notebook hoac ghi de bang RIFT v3 hien tai.

Chay preflight local khong tai model:

```powershell
python scripts/run_spectral_v3_confirmation.py --output-root outputs/spectral_posthoc_v3_supplement --dry-run
```
