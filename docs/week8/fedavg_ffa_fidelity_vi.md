# FedAvg-LoRA va FFA-LoRA: doi chieu va sua implementation

Ngay: 2026-09-26. Implementation tag: `persistent_factors_v2`.

## Nguon doi chieu

- FedIT paper: https://arxiv.org/abs/2305.05644
- Repo chinh thuc FedIT: https://github.com/JayZhang42/FederatedGPT-Shepherd
- Ham FedAvg da doc, pin theo commit:
  https://github.com/JayZhang42/FederatedGPT-Shepherd/blob/90d8aa612852f637dc1ef300f549c8c2dc67c700/fed_utils/model_aggregation.py
- FFA-LoRA (ICLR 2024): https://arxiv.org/html/2403.12313v1
  Doi chieu Section 4 va Appendix A.8: A khoi tao mot lan, giu co dinh;
  chi train va aggregate B. Khong xac minh duoc repo tac gia FFA trong lan
  tra cuu nay, nen day la reimplementation theo paper, khong phai import
  code chinh thuc FFA. Khong su dung repo ben thu ba de tuyen bo fidelity.

## Phan da sua

`src/riftlora/baselines/factor_averaging.py`:

- FedAvg giu A/B that theo tung server version. Dung trung binh tham so:
  `A_next = sum(p_i * A_i)`, `B_next = sum(p_i * B_i)`.
- `average_factor_states` ho tro nhieu client va trong so so luong mau,
  duoc chuan hoa theo tong trong so, nhu ham aggregation cua FedIT.
- FFA giu mot A0 chung, chi tinh `B_next = sum(p_i * B_i)`.
  Neu A0 bi thay doi, aggregation/load bao loi thay vi im lang chap nhan.
- Khong SVD/reinitialize A/B khi dispatch, train hay evaluate.
- CompactSVD chi la ban xuat cho diagnostics/checkpoint danh gia;
  khong phai nguon trang thai cho factor training.

Runner `scripts/run_kaggle_3b.py` dung snapshot factor cho ca stale version,
current version, monitor va final evaluation. Warmup cua hai method nay
cung dung chinh method do; chi khong tinh warmup vao measured returns.
Khong dung freshness warmup roi SVD-load vao FFA.

Result co `factor_implementation` va `factor_protocol`. Week 8 resume va
Week 9 artifact validation khong chap nhan ket qua hai method thieu tag moi.

## Gioi han fidelity can ghi ro

Day KHONG phai reproduction day du moi thiet lap trong paper:

| Thanh phan | Paper operator | Runner confirmation hien tai |
|---|---|---|
| FedAvg | Trung binh A/B client theo trong so | `(1-w)*server + w*returned_client` tren A/B |
| FFA | A0 bat bien, trung binh B | A0 bat bien, noi suy B voi server |
| Lich | Synchronous FL | Immediate asynchronous, stale snapshots |
| Rank | Homogeneous operator | Prefix client ranks, zero padding |
| Warmup | Khong dung quy tac freshness cua RIFT | Native method, unmeasured returns |
| Privacy | Paper FFA co nghien cuu DP va non-DP | Non-DP, khong tuyen bo bao dam DP |

Chinh sach rank mo rong: FedAvg zero-pad A va B ngoai active rank;
FFA giu toan bo A0, chi zero-pad B. Vi the FFA khong bi co ca A va B
tren phan rank ma client khong co. Day la extension can bao cao, khong
gan chinh sach heterogeneous rank nay cho paper FFA.

Trong so server `w` cua async KHONG phai trong so so mau cua synchronous
FedIT. Helper nhieu client co operator synchronous, nhung runner hien tai
khong duoc goi la synchronous paper reproduction. Optimizer, backbone,
quantization va ngan sach van theo config experiment cua du an.

Khong tuyen bo tiet kiem mang do duoc tu simulator: snapshot noi bo van
luu A de audit, khong co truyen mang that. Checkpoint milestone la
evaluation-only, khong resume training tu CompactSVD.

## Kiem thu va ket qua cu

```powershell
python -m pytest -q tests/test_persistent_factor_baselines.py tests/test_fed_lora_baselines.py tests/test_week8_spectral_integration.py tests/test_week9_generation.py
```

Kiem tra: sample-weighted mean, factor gauge counterexample, stale history,
FFA B-only voi heterogeneous rank, A0 bat bien qua AdamW/dispatch/evaluation,
resume tu choi legacy results, real tiny-Qwen train/backprop cho ca bon task.

Ket qua verification: 78 tests passed (CPU, tiny models); chua chay lai
confirmation 1.5B/3B va chua do accuracy moi hay test quantization tren T4.

Ket qua cu `fedavg_lora`/`ffa_lora` khong duoc dung de ket luan thang baseline
paper. Giu ket qua cu de audit, rerun vao output root moi, khong ghi de ket
qua RIFT hay cac method khac. Khong co dam bao sua dung se tang accuracy.
## Kaggle Week 8

Notebook: `notebooks/kaggle_qwen_1_5b_week8_heldout_classification.ipynb`.

```python
METHOD_SUITE = 'factor_only'
RUN_MODE = 'confirmation'
RUN_TRAINING = True
```

Chi chay `fedavg_lora` va `ffa_lora`, 4 tasks x 6 seeds (6101-6106) = 48 jobs.
Qwen2.5-1.5B, mot process tren moi GPU T4, toi da 2 jobs dong thoi.
`RUN_MODE='smoke'` chi chay SST-2 seed6101, 2 jobs ngan de kiem tra ha tang;
khong tron ket qua smoke vao confirmation. Mac dinh RUN_TRAINING=False.

`METHOD_SUITE='spectral_only'` van chi chay FlexLoRA/FLoRIST/FLoRG (72 jobs).
Hai suite va hai run modes co output root, matrix va ZIP rieng.
Ket qua factor confirmation luu trong
`/kaggle/working/week8_qwen15b_week8-factor-v2_confirmation`.
Runner chi skip ket qua cung config, matrix, commit sach va implementation tag.
Notebook pin code release tren GitHub; khong dung commit cu truoc ban sua.

### Khoi phuc notebook7b4f16575b

Batch nay da hoan thanh 48/48 job; loi nam o cell report (`No module named
'riftlora'`) chu khong phai train. Notebook da them `REPO_DIR/src` vao
`sys.path` cua kernel, giu nguyen training commit db4ca69 de skip ket qua hop le.
Xem `docs/week8/week8_factor_kaggle_recovery_results_vi.md`.

Gan output notebook cu vao Kaggle Add Input. `RESUME_ROOTS=['/kaggle/input']`,
`REQUIRE_RESUME=True`, `RUN_TRAINING=False`: chi khoi phuc bang va ZIP.
Neu con job thieu that su, dat RUN_TRAINING=True; queue loc job hoan thanh
truoc khi cache model, va kiem tra lai truoc khi launch. Chi result khong
day du moi duoc force rerun, khong ghi de job da kiem tra hop le.
