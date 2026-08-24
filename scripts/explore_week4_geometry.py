from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vastlora.diagnostics import analyze_innovation_geometry, partial_spearman
from vastlora.lowrank import CompactSVD, LowRankMatrix, compact_svd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exploratory reference-subspace grid on a development seed")
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/week4/week3_diagnostics.csv")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--regime", action="append", default=["iid_homogeneous"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--reference-ranks", type=int, nargs="+", default=(4, 8, 16, 32, 64))
    parser.add_argument("--history-sizes", type=int, nargs="+", default=(4, 8, 16))
    parser.add_argument("--decays", type=float, nargs="+", default=(0.1,))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    frame = frame[frame["seed"] == args.seed].copy()
    frame = frame[frame["regime"].isin(args.regime)]
    if frame.empty:
        raise ValueError(f"seed {args.seed} is not present")
    device = torch.device(args.device)
    cached = _load_updates(frame, args.input.parent, device)
    burn_in = max(args.history_sizes)
    results = []
    for reference_rank, history_size, decay in product(
        args.reference_ranks,
        args.history_sizes,
        args.decays,
    ):
        for regime, updates in cached.items():
            histories: dict[str, list[CompactSVD]] = {
                name: [] for name in updates[0][1]
            }
            rows = []
            for offset, (metadata, innovations) in enumerate(updates):
                compact = {name: compact_svd(value) for name, value in innovations.items()}
                if offset >= burn_in:
                    geometry = analyze_innovation_geometry(
                        innovations,
                        histories,
                        reference_rank=reference_rank,
                        history_size=history_size,
                        reference_decay=decay,
                    )
                    rows.append(
                        {
                            "tau": metadata.tau,
                            "raw_update_utility": metadata.raw_update_utility,
                            "rho_two_sided": geometry.rho_two_sided,
                            "rho_left": geometry.rho_left,
                            "rho_right": geometry.rho_right,
                        }
                    )
                for name, value in compact.items():
                    histories[name].append(value)
            diagnostic = pd.DataFrame(rows)
            results.append(
                {
                    "seed": args.seed,
                    "regime": regime,
                    "reference_rank": reference_rank,
                    "history_size": history_size,
                    "reference_decay": decay,
                    "n": len(diagnostic),
                    "partial_two_sided": partial_spearman(
                        diagnostic,
                        x="rho_two_sided",
                        y="raw_update_utility",
                        controls=("tau",),
                    ),
                    "partial_left": partial_spearman(
                        diagnostic,
                        x="rho_left",
                        y="raw_update_utility",
                        controls=("tau",),
                    ),
                    "partial_right": partial_spearman(
                        diagnostic,
                        x="rho_right",
                        y="raw_update_utility",
                        controls=("tau",),
                    ),
                    "rho_two_sided_mean": diagnostic["rho_two_sided"].mean(),
                }
            )
        print(f"finished rank={reference_rank} history={history_size} decay={decay}")

    output = pd.DataFrame(results)
    output_path = args.input.parent / "analysis" / f"geometry_grid_seed{args.seed}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    primary = output[output["regime"] == "iid_homogeneous"].sort_values(
        "partial_two_sided", ascending=False
    )
    print(primary.head(10).to_string(index=False))
    print(f"output: {output_path}")


def _load_updates(
    frame: pd.DataFrame,
    root: Path,
    device: torch.device,
) -> dict[str, list[tuple[object, dict[str, LowRankMatrix]]]]:
    cached = {}
    for run_id, part in frame.groupby("run_id"):
        part = part.sort_values("update_id")
        artifact_path = root / str(part.iloc[0]["update_artifact_id"]).split("#", 1)[0]
        bundle = torch.load(artifact_path, map_location="cpu", weights_only=False)
        updates = []
        for row in part.itertuples(index=False):
            artifact = bundle["updates"][row.update_id]
            innovations = {
                name: LowRankMatrix(value["left"].to(device), value["right"].to(device))
                for name, value in artifact["innovation"].items()
            }
            updates.append((row, innovations))
        cached[str(part.iloc[0]["regime"])] = updates
    return cached


if __name__ == "__main__":
    main()
