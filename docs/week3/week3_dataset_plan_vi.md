# Week 3 dataset plan

Dataset diagnostic chính: `nyu-mll/glue`, subset `sst2`.

Lý do chọn:

- phù hợp yêu cầu Week 3: một dataset nhỏ, metric rẻ, dễ lặp nhiều seed;
- nằm trong nhóm GLUE diagnostic mà tài liệu đề xuất;
- có trong nhiều baseline liên quan: GLoRA, FedRot-LoRA, FedEx-LoRA, SDFLoRA, FSLoRA;
- split validation có label thật, dùng được để tính `raw_update_utility`;
- split test của GLUE/SST-2 có label ẩn `-1`, nên không dùng cho Week 3 utility.

Artifacts cần tạo trước Week 4:

- `outputs/week3/sst2_dataset_audit.json`;
- `outputs/week3/sst2_iid_partitions.json`.

Lệnh chạy:

```powershell
python scripts/prepare_week3_dataset.py
```

Week 3 nên dùng cấu hình trong `configs/week3_dataset.json`:

- clients: 10;
- ranks: `{4, 8, 16}` theo schedule cố định;
- partition: IID;
- seeds: `17, 31, 43`;
- target staleness: `{0, 1, 2, 4, 8}`;
- metric: accuracy trên validation split.

Checklist trước Week 4:

- train/validation không có text rỗng/null;
- train/validation không có label thiếu;
- test split được ghi nhận là unlabeled và không dùng để tính utility;
- partition IID đầy đủ, không mất/trùng index;
- mỗi seed có manifest riêng để replay client local data;
- audit report có fingerprint SHA-256 cho từng split;
- diagnostic dataframe phải lưu dataset name, split, seed, partition id, client indices artifact, base/current server state id, và update artifact id để replay độc lập.
