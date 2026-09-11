# Spectral Surgery / AlignFed: audit va ban tai trien khai

Ngay: 2026-09-11. Day la thay doi implementation, chua phai ket qua accuracy moi.

## Ket luan

Cac baseline trong confirmation v3 chua phai full reproduction theo paper.
Khong doi nhan cac ket qua do sau khi them code moi.

| Implementation | Tinh trang sau audit |
|---|---|
| `spectral_surgery` | Du 4 policy spectral; van edit innovation moi return async |
| `spectral_surgery_posthoc` | Moi: edit adapter sau FL, answer-token calibration, chon module edit rieng |
| `alignfed_calibration` | Giu nguyen whole-update calibration control; khong phai AlignFed |
| `alignfed_reference` | Moi: du cac giai doan thuat toan, rank-linear T va pooling duoc khai bao la gia dinh tai trien khai |
| `factor_fedbuff` | Doi chung factor averaging cung scheduler/rank/data voi AlignFed reference; khong phai official FedBuff reproduction |

Nguon doi chieu:

- [Spectral Surgery, arXiv:2603.03995v1](https://arxiv.org/html/2603.03995v1), Sec. 3.3-3.4, 4.3 va Appendix A.
- [AlignFed, arXiv:2606.08197v1](https://arxiv.org/html/2606.08197v1), Eq. (3)-(8), Algorithm 1.
- Tim kiem chua xac minh duoc repository chinh thuc cua hai paper.
  [Repo AlignFed cua mansik-11](https://github.com/mansik-11/AlignFed-Federated-LLM)
  tu mo ta la from-scratch implementation; khong su dung no nhu bang chung official fidelity.

## Spectral Surgery

Da them `random_index`, signed `grad_direction`, `smooth_align_mid`, clipping va
tham so buoc signed. Signed gradient duoc lay trung binh co dau; magnitude lay
trung binh absolute tung example. Hai dai luong khong thay the cho nhau.

Sensitivity dung perturbation don vi, tranh chia gradient cho singular value rat
nho. Test kiem tra voi sigma = 0, 1e-20, 1 va 1e5. Random control dung RNG rieng,
khong lam doi lich lay batch cua client. U/V va cac module khong duoc chon giu
nguyen; mac dinh bao toan L1 spectral mass.

Duong post-hoc moi train bang `freshness` (co the chon `raw`), roi chi edit mot
lan o adapter cuoi. Calibration teacher-forcing chi tinh answer tokens, gom EOS.
Mac dinh edit `o_proj/down_proj`; config audit train ca bay projection modules va
dung 128 calibration examples. Khong the edit cac module chua co LoRA trong
checkpoint cu `q_proj/v_proj`; runner bao loi neu target bi thieu.

`result.json` ghi `spectral_posthoc.pre_edit_metrics` va final metrics. Khong dung
eval de chon policy, checkpoint hoac quyet dinh co giu edit hay khong. Harmful/late
harmful cua run nay do cac return TRUOC post-hoc; khong duoc dien giai la kha nang
chan harmful return cua phep edit cuoi.

Signed step sizes (mac dinh 0.1), positive power (1), random seed (0) la cac lua
chon duoc cong khai trong code, khong tu nhan la hyperparameter official. Van
khac paper o backbone, QLoRA, task, lich FL va training budget. Train 40 returns
khong chung minh adapter da hoi tu. Vi vay day la protocol gan hon paper, khong
phai tai lap nguyen benchmark 8B cua tac gia.

## AlignFed

`scripts/run_alignfed_reference.py` dung runner rieng, vi runner immediate async
cu khong du kha nang bieu dien framework nay.

| Giai doan | Code va quy uoc |
|---|---|
| Client local representation penalty | Task loss + squared L2 discrepancy voi dispatch snapshot, teacher detached |
| Increment tren A/B | Snapshot va load dung factor; khong tai phan tich SVD giua cac version |
| Buffer | Flush theo count hoac simulated timeout, ke ca khong co arrival tai deadline |
| Version grouping / centering | Gom theo base version va tru trung binh factor increments trong nhom |
| Semantic alignment | Hoc hai rank-linear maps rieng cho A/B tren calibration mean features; fresh group identity |
| Fairness | Exponential freshness, inverse centered-update norm, inverse sqrt upload frequency; normalize |
| Server update | Cong weighted aligned factor increments; version tang khi flush |

Gia dinh can cong khai: T gom hai ma tran r x r rieng; feature la mean prompt
tokens sau decoder block ap chot; Adam, 3 steps, LR 0.01; giu iterate co calibration
feature discrepancy thap nhat, gom identity. Day la cach cu the hoa Eq. (6), chua
duoc xac minh voi implementation cua tac gia. Gradient di qua functional PEFT
factors, khong copy detached weights roi gia vo dang optimize T.

Rank dong nhat va toa do A/B co dinh la rang buoc cua runner moi. Khong zero-pad
hay rotate heterogeneous rank roi goi do la exact AlignFed. Calibration duoc
reserve tu train, tach client/monitor/eval; 128 examples trong audit nay la ngan
sach thich nghi, khong tu nhan giong ti le public calibration cua paper.

### Van de centering khong the tu dong sua ma van goi dung paper

Tu Eq. (5), voi mot nhom chi co delta:

```text
centered_delta = delta - mean([delta]) = 0
T(0) = 0 vi T la linear
```

Do do buffer_size=1 khong tao server update trong implementation literal nay.
Voi hai update cung version, centered deltas la d va -d. Neu freshness, upload
count va norm cho cung weight, tong cung bang 0. Day la suy luan dai so, da co
test phan vi du; khong phai ket luan thuc nghiem cua tac gia.

Khong am tham bo centering, cong lai group mean, thay T bang additive repair,
hay bo singleton de lam accuracy dep. Nhung thay doi do la method variant.
Runner ghi `singleton_groups`, `zero_centered_returns`, `version_group_sizes`
va weights de thay hien tuong. Buffer lon hon khong tu dong loai bo no.

## Cong bang va performance

Khong co cam ket "them du paper ma accuracy khong giam". Calibration loss thap
hon khong dam bao held-out accuracy cao hon. De tranh lam doi thu yeu do config,
can tune tren development voi ngan sach ngang nhau, giu ca ban cu va ban moi,
khoa cau hinh truoc khi chay mot confirmation moi. Khong chon theo seed thang.

`configs/paper_baseline_audit_matrix.json` la DEVELOPMENT: seeds 7201-7203,
SST-2 eval 512; QNLI/MNLI-m/MNLI-mm eval 1024. Cac config thay doi module, rank,
calibration va server weight, nen phai chay lai cac doi chung can so sanh.
Khong gop ket qua voi `rift_core_final_results_board_vi.md` v3.

Spectral/RIFT trong matrix van dung immediate async. AlignFed reference va
factor_fedbuff dung buffered runner. So sanh trong tung protocol truoc; chua
tuyen bo AlignFed reference thang/thua RIFT cong bang ve scheduler. De lam
head-to-head tiep theo, can port RIFT vao cung persistent-factor buffered
protocol hoac thiet ke matched return/time-budget study ro rang.

Harmful cua buffered runner tinh tren FLUSH; late la buffer chua it nhat mot
return late. Ten metric la `harmful_buffer_rate`, `late_harmful_buffer_rate`,
khong gop voi per-return harmful. Khong co late buffer thi rate la null.
Chi so can xem: accuracy, class NLL, so flush, so late buffer, zero-centered
returns, calibration compute va runtime. Training budget hien tai la pilot,
chua phai convergence study. Wall-clock alignment cost duoc do trong runtime
nhung chua cong vao simulated client schedule.

## Lenh chay

Dry-run khong tai model/data:

```powershell
python scripts/run_week8_classification_matrix.py --matrix configs/paper_baseline_audit_matrix.json --task sst2 --method spectral_surgery_posthoc --seed 7201 --dry-run
python scripts/run_alignfed_reference.py --task sst2 --method alignfed_reference --seed 7201 --dry-run
```

Pilot tren GPU du VRAM, chay tung job:

```powershell
python scripts/run_week8_classification_matrix.py --matrix configs/paper_baseline_audit_matrix.json --task sst2 --method freshness --method spectral_surgery --method spectral_surgery_posthoc --seed 7201 --output-root outputs/paper_baseline_development_spectral
python scripts/run_alignfed_reference.py --task sst2 --method factor_fedbuff --seed 7201
python scripts/run_alignfed_reference.py --task sst2 --method alignfed_reference --seed 7201
```

Thay `sst2` bang `qnli`, `mnli_m`, `mnli_mm`. Runner AlignFed bao loi neu
`result.json` da ton tai de tranh ghi de. Dung output-dir moi khi doi config.
Chua chay pilot 1.5B/3B hay xac nhan VRAM tren T4 trong audit nay. Seven-module
LoRA va semantic alignment ton tai nguyen hon config cu.

## Verification

Ket qua: 79 tests pass trong lenh duoi. Dry-run config cua ca 4 task hop le.
Suffix-only answer loss va gradient khop teacher-forcing day du tren tiny Qwen;
duong post-hoc dung cach nay de giam bo nho vocabulary logits.

Test offline dung Qwen2 khoi tao ngau nhien rat nho voi PEFT that, khong download:
functional gradients, local training, timer flush, stale alignment, post-hoc
edit, result JSON. Unit tests kiem tra cong thuc signed/random, mass, singleton
centering, fairness weights, teacher detach, disconnected-gradient rejection.
Day la kiem tra code; khong phai bang chung accuracy cua Qwen 1.5B/3B.

```powershell
python -m pytest -q tests/test_paper_baseline_integration.py tests/test_alignfed_reference.py tests/test_spectral_surgery.py tests/test_scale_objective.py tests/test_kaggle_3b_tasks.py tests/test_week8_matrix_runner.py tests/test_competitor_transports.py
```
