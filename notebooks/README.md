# Kaggle notebooks

`kaggle_qwen_3b_mtip_scale.ipynb` runs the memory-bounded Qwen2.5-3B benchmark on
Kaggle T4x2. Enable Internet and select the T4x2 accelerator before running all
cells. The default `pilot` mode smoke-tests leakage-controlled residual tuning
and one confirmation seed. Set `RUN_MODE = "full"` to tune on a holdout removed
from client training, freeze one target variant, and run the five-seed Pareto
gate on the full SST-2 validation set.

The notebook clones and checks out runner commit `01e5d63`, records the resolved
commit in every result, and writes CSV, JSON, Markdown, plots, and a downloadable
ZIP under `/kaggle/working/riftlora-3b-results`. Updated runs also write
per-example NLL debug CSVs and worst-regression tables under
`/kaggle/working/riftlora-3b-results/summary/nll_debug`.

The gate reports sequence NLL, label-only NLL, EOS NLL, binary candidate NLL,
Brier score, balanced accuracy, and paired confidence intervals. Tuning
artifacts and the frozen target are stored under
`/kaggle/working/riftlora-3b-results/tuning`.

`kaggle_qwen_3b_vast_slice_matrix.ipynb` runs the follow-up Qwen2.5-3B
slice-matrix benchmark for the revised VAST question. It checks three regimes:
IID homogeneous rank, IID heterogeneous rank, and non-IID high-staleness
heterogeneous rank. The notebook compares `raw`, `freshness`, `vast`, and
`mtip`, then writes paired NLL/calibration/accuracy summaries and a VAST
hard-slice verdict under `/kaggle/working/riftlora-3b-slice-results/slice_summary`.

`kaggle_qwen_3b_week1_competitor_board.ipynb` runs the same 3B reproduced
baseline family (`raw`, `fedex`, `freshness`, `fedrot`, `vast`, `mtip`) and
builds a Week 1 competitor board next to literature targets such as GLoRA,
FedRot-LoRA, FLoRG, FedEx-LoRA, SDFLoRA, FSLoRA, and AlignFed. `fedex` and
`fedrot` are matched-simulator ports of the public-code external baselines;
remaining paper rows stay reference-only unless they have been ported into the
matched simulator, so the notebook can judge whether the current VAST result is
strong enough without overclaiming against incompatible paper numbers.

`kaggle_qwen_3b_rift_week8_classification_matrix.ipynb` is the reproducible
Week 8 entry point for the RIFT thesis. It clones the GitHub repository,
installs the current code, runs the tests, and launches the SST-2/QNLI/MNLI
classification matrix with two one-GPU workers when T4x2 is available. Use
`smoke` for a quick check, `focused` for the five-control hard-slice comparison,
and `full` for the complete 576-run acceptance matrix. Large runs can be split
deterministically with `SHARD_COUNT` and `SHARD_INDEX`. Results, best-observed
descriptive scores, provenance checks, and the analyzer verdict are packaged as
a ZIP under `/kaggle/working`.

