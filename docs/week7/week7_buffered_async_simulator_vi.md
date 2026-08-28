# Week 7 - Buffered async simulator

Status: done

## Da lam

- `buffer_size > 1` trong `AsyncEventSimulator`.
- Version-aware update groups via `group_id`, `group_version`, `group_position`.
- Delayed group aggregation voi `completed_groups` va `pending_groups`.
- Compatibility voi RIFT wrapper thong qua trace metadata giu nguyen API cu.
- `schedule_mode="cohort"` de chay synchronous cohort-style protocol cho baseline gan GLoRA/FedRot.
- `scripts/run_week2_simulator.py` in them group summary.

## Exit criterion

- So sanh `buffer_size=1` vs `K` khong doi local training.
- Cohort mode cho phep tao trace dong bo hon cho protocol faithful hon.
