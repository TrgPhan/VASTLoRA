# Week 4 kill-test plan (frozen before full run)

Ngay dong bang: 2026-08-24

## Cau hoi nghien cuu

Kiem tra lieu compatibility hai phia `rho_two_sided` co giai thich true post-hoc utility cua stale LoRA innovation sau khi da control version staleness `tau` hay khong.

Week 4 khong toi uu tham so theo ket qua test. Cau hinh model, local training, reference history va freshness lambda duoc dong bang trong `configs/week4_killtest.json` truoc khi chay 3 seed chinh.

## Matrix chinh

| Regime | Partition | Rank | Muc dich |
|---|---|---|---|
| `iid_homogeneous` | IID | tat ca rank 8 | primary isolated kill-test |
| `iid_heterogeneous` | IID | 4, 8, 16 | kiem tra rank robustness |
| `noniid_high_staleness` | label shard | 4, 8, 16 | kiem tra conditional regime |

Moi regime chay seed `17, 31, 43`, 8 warm-up returns va 60 diagnostic returns. Tong muc tieu la 540 replayable updates, trong do phan lon la stale.

## Phan tich bat buoc

1. Spearman utility voi `tau` va `rho_two_sided`.
2. Partial Spearman cua utility va rho sau khi control tau, kem bootstrap 95% CI.
3. Held-seed-out regression R2: `tau` so voi `tau + rho`.
4. Held-seed-out harmful-update AUROC: `tau`, `rho`, va `tau + rho`.
5. Matched-tau analysis trong ba band `0-2`, `3-7`, `8+`.
6. So sanh paired raw, freshness-only va VAST transported utility/harmful rate.
7. Bao cao tung seed va tung regime; khong chi bao cao pooled result.

## Gate da dong bang

Mot regime pass khi tat ca dieu kien sau dung:

- partial Spearman >= 0.10 va bootstrap CI lower bound > 0;
- it nhat 2/3 seed co partial correlation duong;
- CV R2 gain hoac harmful AUROC gain >= 0.02 khi them rho vao tau;
- mean VAST utility lon hon freshness-only;
- VAST harmful-update rate khong cao hon freshness-only.

Quyet dinh:

- `GO`: IID homogeneous va IID heterogeneous cung pass.
- `CONDITIONAL GO`: chi primary pass, hoac chi strong non-IID/high-staleness pass; claim phai thu hep theo regime duoc ho tro.
- `NO-GO`: khong regime nao pass; dung trien khai full VAST/NVFlare va pivot.

## Guardrails

- Khong chon seed, bo outlier, doi rho, reference rank, history hoac lambda sau khi xem full result.
- SST-2 validation chi do utility offline; transport rule khong duoc truy cap calibration data.
- Dataset conflict warning da ghi o Week 3 duoc giu nguyen. Chi chay clean-label ablation neu ket qua sat nguong va phai bao cao ca hai.
- Ket qua duong tren mot tiny model/mot dataset chi cho phep di tiep, chua du de khang dinh final thesis claim. Sau GO van can task thu hai va backbone lon hon.

## Exploratory tuning va confirmation freeze

Preregistered run dau tien cho ket qua NO-GO. De kiem tra lieu ket qua nay den tu baseline update qua manh hoac reference budget chua phu hop, seed `17` duoc dung lam development seed cho sensitivity co nhan ro la exploratory.

Cau hinh ung vien duoc khoa truoc confirmation:

```text
server_update_weight = 0.1
reference_rank = 4
history_size = 8
reference_decay = 0.1
freshness_lambda = ln(2) / 4
```

Ly do chon: trajectory validation loss on dinh, point partial correlation vuot 0.10, mean VAST utility cao hon freshness va harmful rate thap hon tren development run. CI cua mot seed van cat 0, nen development result khong duoc tinh la GO evidence.

Confirmation dung ba seed moi `59, 71, 89`, khong doi tham so sau khi xem result. Gate dinh luong giu nguyen nhu phan tren. Neu primary IID khong pass, ket luan cuoi cung la NO-GO; khong tiep tuc grid search.
