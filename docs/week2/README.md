# Week 2 - Low-rank algebra foundation vÃ  async simulator

Má»¥c tiÃªu Week 2 lÃ  táº¡o ná»n ká»¹ thuáº­t Ä‘á»§ cháº¯c Ä‘á»ƒ sang Week 3-4 thu tháº­p stale innovations vÃ  cháº¡y kill-test.

## Tráº¡ng thÃ¡i hiá»‡n táº¡i

ÄÃ£ triá»ƒn khai:

- low-rank matrix representation;
- exact LoRA innovation factorization;
- compact QR/SVD khÃ´ng materialize dense matrix trong Ä‘Æ°á»ng chÃ­nh;
- weighted low-rank sum;
- recompression theo rank budget;
- temporal reference subspace builder;
- projection vÃ  compatibility score;
- deterministic IID partition;
- deterministic label-shard non-IID partition;
- in-memory versioned snapshot store;
- deterministic asynchronous event simulator;
- script demo simulator.

## Lá»‡nh kiá»ƒm tra

```powershell
python -m pytest
python scripts/run_week2_simulator.py
```

## File chÃ­nh

- `src/riftlora/lowrank/core.py`
- `src/riftlora/asyncfl/simulator.py`
- `src/riftlora/asyncfl/snapshots.py`
- `src/riftlora/data/partitioning.py`
- `configs/week2_simulator.json`
- `scripts/run_week2_simulator.py`
- `tests/lowrank/test_core.py`
- `tests/test_async_simulator.py`
- `tests/test_partitioning.py`


