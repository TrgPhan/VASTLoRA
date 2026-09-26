# Week 8: FlexLoRA, FLoRIST va FLoRG

Ngay: 2026-09-26. Pham vi: implementation va kiem tra ky thuat; chua chay
confirmation 1.5B. Khong thay so lieu cu trong final results board.

## Notebook va job

Notebook: [kaggle_qwen_1_5b_week8_heldout_classification.ipynb](../../notebooks/kaggle_qwen_1_5b_week8_heldout_classification.ipynb).

- `RUN_MODE = 'confirmation'`: 4 task SST-2/QNLI/MNLI-m/MNLI-mm,
  6 seed 6101-6106, chi `flexlora`, `florist`, `florg`: 72 job.
- `RUN_MODE = 'smoke'`: SST-2, seed6101, ca ba method: 3 job diagnostic.
- `RUN_TRAINING = False`: chi preflight/tests; doi thanh `True` de train.
- Mot process/GPU, hai GPU T4 chay hai job doc lap; khong gop VRAM thanh 32GB.
- `SHARD_COUNT`/`SHARD_INDEX` chia danh sach job, khong thay matrix/seed.
- Output moi: `week8_qwen15b_spectral_v1_confirmation`; smoke co root rieng.
- `RESUME_ROOTS` chi nhap ket qua cua ba method tren. Runner chi skip khi
  method/seed/config fingerprint/matrix SHA/commit/clean worktree khop.
- Queue dung khi co job loi. Sau khi sua, dung output root/release phu hop.
- Notebook tai GitHub va detach commit; `release.json` ghi commit da dung.
  Khi resume, giu `REPO_REF` bang SHA nay. Pin release ngan lay nham code cu.
- Cell cuoi hien metric thuc va xuat CSV/ZIP, nhom thieu seed la preliminary.

`scripts/week8_spectral_suite.py` tao matrix tu confirmation v3 hien tai.
Task, held-out slice 1024 mau, data/model revisions, seed, local training,
client ranks `[2,4,8,4]`, arrival trace va calibration reservation duoc giu.
Ba method khong dung calibration labels de toi uu server.

## Nguon da doi chieu

| Method | Paper | Code/nguon tham khao |
|---|---|---|
| FlexLoRA | [NeurIPS 2024, Sec. 3.2/Algorithm](https://arxiv.org/html/2402.11505v2) | [Official aggregation](https://github.com/alibaba/FederatedScope/blob/1cb1ab76bf4c9394d617c4c5a800cb0df145796d/fed_utils/model_aggregation.py), branch FlexLoRA |
| FLoRIST | [MLSys 2026, Eq. 1-4/Algorithm 1](https://arxiv.org/html/2506.09199v2) | [Official aggregation](https://github.com/DASS-Lab-Group/FLoRIST/blob/670a2247a32b4fc8ac165d0f9691416e2cc41c15/fed_utils/model_aggregation.py) |
| FLoRG | [ICLR 2026, Eq. 1/4/7/11](https://arxiv.org/html/2602.17095v1) | Reimplementation theo equations; chua xac minh duoc official executable repository |

Code moi duoc viet theo equations va kiem tra bang dense reference.
Khong copy nguyen training stack cua paper. Trong code FLoRIST upstream da doc,
nhanh stacking co cau truc `for ... else` va weighting de gay nham; implementation
o day dat sample weights dung mot lan va test bat buoc khop tong product.

## FlexLoRA

Operator round-level: `W = sum_i w_i * scaling_i * B_i @ A_i`.
Phan ra SVD va lay cac thanh phan dau theo rank cua client khi dispatch.
Trien khai QR + SVD core nho tinh cung product/SVD ma khong materialize W dense.
Test doi chieu voi SVD dense, heterogeneous ranks va scaling khac 1.
Do backend toi uu, khong dung runtime nay de claim toc do cua dense reference.

Trong runner async: `W_next = truncate((1-eta)*W_server + eta*W_client_after)`.
Day la lua chon async protocol, khong phai synchronous paper round.
Khong nham `W_client_after` voi innovation `W_client_after-W_client_before`.
Rank cap 8 la ngan sach simulator. Truoc khi dispatch client rank r, lay top-r.

## FLoRIST

Weighted stacking -> SVD rieng B/A -> SVD intermediate product -> chon rank
nho nhat giu it nhat `tau=0.9` tong `sigma^2`. Khong dung tong sigma va khong
soft-threshold bang phep tru. Gia tri singular duoc giu nguyen sau chon rank.
Implementation su dung hai factor SVD va intermediate SVD nhu paper.

Wrapper async dung cung full-state convex mix nhu FlexLoRA. Server cap 8 co
the lam retained energy thap hon 0.9; events ghi ro:

- `spectral_rank_before_cap`, `spectral_rank_after_cap`.
- `spectral_mean_retained_energy`, `spectral_cap_binding_fraction`.

Client initialization o base version 0 giu zero B/random A. Sau do dung
zero-padding/truncation theo Algorithm 1; khong tu them random free directions
moi vao nhung version da train. Tai tau=1 va cung cap, FLoRIST va FlexLoRA
phai khop product trong sai so so hoc; test bao phu tinh chat nay qua dense oracle.

## FLoRG: cac loi da sua

1. Adapter/buffers dat cung device voi target weight, giu trainable FP32 ca khi
   backbone NF4. Forward tinh qua `L A^T` va `A R`, khong dung Gram dense.
2. Freeze backbone ca tren CPU va CUDA. Hai fixed bases L/R khong train.
3. RNG cua A su dung seed da truyen, khong tai su dung seed chi theo shape.
4. Client rank nho thuc su nhan rank-r Gram approximation va zero inactive
   rows; truoc do chi mask gradient nen forward van dung tat ca rows.
5. Aggregate Gram qua SVD stacked factors, tuong duong eigendecomposition.
   Lam rectangular Procrustes tren full numerical-rank decomposition theo
   Eq. 11; bo cach truncate truoc Procrustes trong implementation cu.
6. Coverage tinh tu actual client returns; extreme harmful chia cho extreme
   events; cac field khong ap dung ghi null. Bo sung late cumulative harm.

Gioi han toan hoc can ghi ro: voi Gram rank r' > row budget r, khong ton tai
S kich thuoc r x r' thoa `S.T @ S = I_(r')`. Code ap dung rectangular
`U @ Vh` nhu Eq. 11, nhung khong claim bao toan Gram khi r' > r. Test bao phu
truong hop rank-fit va rank-overflow. Heterogeneous rank dispatch la extension
cua simulator, vi paper setup co rank chung.

FLoRG khoi tao A Gaussian std 0.01 (cau hinh implementation hien tai), nen
adapter ban dau khong bang zero nhu LoRA B=0; baseline metric ghi tai model
thuc te nay. Khong coi baseline initialization giua hai parameterization la
hoan toan dong nhat.

## Muc do fidelity va fair comparison

Day la paper-equation operators trong matched async experiment. Khong goi la
full-paper reproduction: model, optimizer, rank budget, local steps va protocol
khac benchmark goc. FlexLoRA/FLoRIST co 8 freshness warmup return theo Week 8;
FLoRG dung Gram aggregation tu dau, warmup chi loai khoi measured safety window.
Su khac biet nay duoc ghi trong matrix notes de reviewer nhin thay.

Khong ket luan "RIFT thang paper X" tu cac adaptation nay. Can bao cao fidelity,
ngan sach server calibration, so seed va cap-binding truoc khi giai thich ket qua.
LR cua confirmation giu cung protocol; day khong phai best-tuned-paper baseline.

## Kiem tra va chay

```powershell
python -m pytest tests/test_spectral_fedlora.py tests/test_fed_lora_baselines.py tests/test_week8_spectral_integration.py -q
python scripts/week8_spectral_suite.py --output outputs/week8_spectral_preflight/matrix.json
python scripts/run_week8_classification_matrix.py --matrix outputs/week8_spectral_preflight/matrix.json --output-root outputs/week8_spectral_confirmation --dry-run
```

Test gom weighted-product dense oracle, squared-energy boundary, rank cap,
zero matrix, rectangular Procrustes, CUDA forward/gradient, NF4 hook backward,
va 12 tiny-Qwen integration runs (4 task x 3 method). Tiny model/data duoc tao
local, khong tai weights hay chay full 1.5B confirmation trong buoc kiem tra.
