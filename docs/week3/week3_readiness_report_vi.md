# Week 3 readiness report

Ngày chạy audit: 2026-08-24

## Dataset đã chọn

Chọn `nyu-mll/glue`, subset `sst2`.

Đây là dataset diagnostic chính cho Week 3 vì rẻ, metric accuracy rõ, có validation label thật, và xuất hiện trong nhiều baseline GLUE/NLU đã liệt kê ở Week 1.

Không dùng `test` split để tính `raw_update_utility` vì label của split này là `-1`.

## Artifact local

Script:

```powershell
python scripts/prepare_week3_dataset.py
```

Tạo:

```text
outputs/week3/sst2_dataset_audit.json
outputs/week3/sst2_iid_partitions.json
```

Lưu ý: `outputs/*` đang được ignore bởi Git, nên cần chạy lại script nếu chuyển máy/cache.

## Kết quả kiểm tra dữ liệu

| Split | Rows | Label counts | Unlabeled | Empty/null text | Duplicate idx | Duplicate normalized texts | Conflicting duplicates |
|---|---:|---|---:|---:|---:|---:|---:|
| train | 67,349 | 0: 29,780; 1: 37,569 | 0 | 0 | 0 | 371 | 5 |
| validation | 872 | 0: 428; 1: 444 | 0 | 0 | 0 | 0 | 0 |
| test | 1,821 | hidden labels | 1,821 | 0 | 0 | 0 | 0 |

Blocking issue: không có.

Warning duy nhất: train có 5 normalized texts xuất hiện với label mâu thuẫn:

```text
idx 1433 / 54844: laughably
idx 8493 / 66327: the only reason
idx 10928 / 30663: excruciatingly
idx 18521 / 38655: sleeper
idx 38764 / 42877: sillier
```

Khuyến nghị: giữ nguyên SST-2 chuẩn cho baseline compatibility, nhưng ghi warning này trong limitations/audit. Nếu Week 4 phân tích quá nhiễu, chạy thêm ablation nhỏ loại 10 dòng conflict để kiểm tra độ nhạy.

## Partition Week 3

Cấu hình hiện tại:

- clients: 10;
- partition: IID;
- seeds: `17, 31, 43`;
- rank schedule: `[4, 8, 16, 4, 8, 16, 4, 8, 16, 8]`;
- target staleness: `[0, 1, 2, 4, 8]`.

Kết quả partition:

| Seed | Complete | Min samples/client | Max samples/client | Positive-rate range |
|---:|---|---:|---:|---|
| 17 | yes | 6,734 | 6,735 | 0.5455 - 0.5647 |
| 31 | yes | 6,734 | 6,735 | 0.5501 - 0.5692 |
| 43 | yes | 6,734 | 6,735 | 0.5519 - 0.5635 |

IID split đủ tốt cho Week 3 diagnostic.

## Week 3 còn cần bổ sung trước Week 4

Dataset side đã sẵn sàng. Để hoàn thành đúng Week 3, phần còn thiếu là minimal FedLoRA diagnostic run:

1. Minimal LoRA trainer trên model nhỏ.
2. Snapshot adapter theo `base_version`.
3. Khi client return, lưu được initial/final LoRA factors để dựng exact innovation.
4. Diagnostic dataframe phải có đủ cột:

```text
client_id
base_version
current_version
tau
rank
num_samples
virtual_latency
||D||
effective_rank
rho_left
rho_right
rho_two_sided
raw_update_utility
```

5. Mỗi stale update cần thêm metadata replay:

```text
dataset_name
dataset_fingerprint_sha256
partition_seed
partition_artifact
client_indices_artifact
base_snapshot_id
current_snapshot_id
update_artifact_id
validation_split
metric
```

6. Trước Week 4, dataframe cần pass các sanity checks:

- tối thiểu vài trăm stale updates;
- `tau` phủ được `{1, 2, 4, 8}` hoặc gần nhất có thể;
- không thiếu `rho_*`, `||D||`, `effective_rank`, `raw_update_utility`;
- replay một sample ngẫu nhiên của stale updates khớp lại metric/utility đã log;
- cùng seed chạy lại cho cùng return ordering, staleness histogram, và dataframe schema.
