# Week 8: notebook7b4f16575b recovery

Verified: 48/48. Pending/invalid: 0.

Source: `outputs/kaggle_notebook7b4f16575b/week8_qwen15b_week8-factor-v2_confirmation`.
Training release: `db4ca69024fdd0697f7ed667efbf10f025c558a3`.

All values below are means over completed seeds; accuracy/harmful are percentages.
Only groups with 6 seeds are complete confirmation groups.

| Task | Method | Seeds | Accuracy (%) | Class NLL | Harmful (%) | Late harmful (%) | Runtime (min/job) |
|---|---|---:|---:|---:|---:|---:|---:|
| sst2 | fedavg_lora | 6 | 78.97 | 0.435950 | 33.33 | 71.43 | 4.38 |
| sst2 | ffa_lora | 6 | 76.19 | 0.493026 | 39.06 | 71.43 | 4.37 |
| qnli | fedavg_lora | 6 | 71.68 | 0.539073 | 57.29 | 38.10 | 5.01 |
| qnli | ffa_lora | 6 | 72.15 | 0.535883 | 59.38 | 35.71 | 5.00 |
| mnli_m | fedavg_lora | 6 | 73.70 | 0.646803 | 52.08 | 66.67 | 5.68 |
| mnli_m | ffa_lora | 6 | 73.58 | 0.655751 | 50.52 | 64.29 | 5.60 |
| mnli_mm | fedavg_lora | 6 | 72.85 | 0.665449 | 52.08 | 66.67 | 5.61 |
| mnli_mm | ffa_lora | 6 | 72.62 | 0.678624 | 50.52 | 64.29 | 5.64 |

## Nguyen nhan loi

Kernel log: Finished shard: 48, sau do ModuleNotFoundError: No module named 'riftlora' o cell report.
pip install -e chay trong subprocess; kernel notebook dang mo chua nap editable-package path moi.
Day la loi import/bao cao, khong phai bang chung train fail, OOM hay loi thuat toan.

## Xu ly

Da them REPO_DIR/src vao sys.path cua kernel va kiem tra package dung checkout.
Giu nguyen training release de tai su dung ket qua cu, khong sua logic method.
Gan output notebook cu vao Add Input, RUN_TRAINING=False de tao lai bang/ZIP.
Queue chi giu job chua hoan thanh sau khi kiem tra config, commit, implementation va du artifacts.
REQUIRE_RESUME=True chan train nham khi chua gan output cu.

Audit doi chieu metrics voi 1024 eval rows, 40 returns/32 measured, harmful flags va paired data/trace.
Khong suy ra RIFT thang/thua tu bang nay: can doi chieu cohort RIFT cung protocol, seed va ngan sach.
