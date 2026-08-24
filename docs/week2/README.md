# Week 2 - Low-rank algebra foundation và async simulator

Mục tiêu Week 2 là tạo nền kỹ thuật đủ chắc để sang Week 3-4 thu thập stale innovations và chạy kill-test.

## Trạng thái hiện tại

Đã triển khai:

- low-rank matrix representation;
- exact LoRA innovation factorization;
- compact QR/SVD không materialize dense matrix trong đường chính;
- weighted low-rank sum;
- recompression theo rank budget;
- temporal reference subspace builder;
- projection và compatibility score;
- deterministic IID partition;
- deterministic label-shard non-IID partition;
- in-memory versioned snapshot store;
- deterministic asynchronous event simulator;
- script demo simulator.

## Lệnh kiểm tra

```powershell
python -m pytest
python scripts/run_week2_simulator.py
```

## File chính

- `src/vastlora/lowrank/core.py`
- `src/vastlora/asyncfl/simulator.py`
- `src/vastlora/asyncfl/snapshots.py`
- `src/vastlora/data/partitioning.py`
- `configs/week2_simulator.json`
- `scripts/run_week2_simulator.py`
- `tests/lowrank/test_core.py`
- `tests/test_async_simulator.py`
- `tests/test_partitioning.py`

