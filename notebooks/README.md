# Kaggle notebooks

`kaggle_qwen_3b_mtip_scale.ipynb` runs the memory-bounded Qwen2.5-3B benchmark on
Kaggle T4x2. Enable Internet and select the T4x2 accelerator before running all
cells. The default `pilot` mode compares Freshness, VAST, fixed MTIP, and
adaptive MTIP on one seed. Set `RUN_MODE = "full"` for the three-seed GO gate.

The notebook clones and checks out runner commit `5c44028`, records the resolved
commit in every result, and writes CSV, JSON, Markdown, plots, and a downloadable
ZIP under `/kaggle/working/vastlora-3b-results`.
