# Kaggle notebooks

`kaggle_qwen_3b_mtip_scale.ipynb` runs the memory-bounded Qwen2.5-3B benchmark on
Kaggle T4x2. Enable Internet and select the T4x2 accelerator before running all
cells. The default `pilot` mode smoke-tests leakage-controlled residual tuning
and one confirmation seed. Set `RUN_MODE = "full"` to tune on a holdout removed
from client training, freeze one target variant, and run the five-seed Pareto
gate on the full SST-2 validation set.

The notebook clones and checks out runner commit `01e5d63`, records the resolved
commit in every result, and writes CSV, JSON, Markdown, plots, and a downloadable
ZIP under `/kaggle/working/vastlora-3b-results`. Updated runs also write
per-example NLL debug CSVs and worst-regression tables under
`/kaggle/working/vastlora-3b-results/summary/nll_debug`.

The gate reports sequence NLL, label-only NLL, EOS NLL, binary candidate NLL,
Brier score, balanced accuracy, and paired confidence intervals. Tuning
artifacts and the frozen target are stored under
`/kaggle/working/vastlora-3b-results/tuning`.

`kaggle_qwen_3b_vast_slice_matrix.ipynb` runs the follow-up Qwen2.5-3B
slice-matrix benchmark for the revised VAST question. It checks three regimes:
IID homogeneous rank, IID heterogeneous rank, and non-IID high-staleness
heterogeneous rank. The notebook compares `freshness`, `vast`, and `mtip`,
then writes paired NLL/calibration/accuracy summaries and a VAST hard-slice
verdict under `/kaggle/working/vastlora-3b-slice-results/slice_summary`.
