from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vastlora.asyncfl import AsyncEventSimulator, ClientProfile  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Week-2 VAST-LoRA async simulator.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "week2_simulator.json",
        help="Path to a JSON simulator config.",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    clients = [ClientProfile(**item) for item in config["clients"]]
    simulator = AsyncEventSimulator(
        clients,
        seed=config["seed"],
        buffer_size=config["buffer_size"],
    )
    trace = simulator.run(max_returns=config["max_returns"])

    output = {
        "return_order": trace.return_order,
        "dispatch_versions": trace.dispatch_versions,
        "arrival_versions": trace.arrival_versions,
        "staleness_values": trace.staleness_values,
        "staleness_histogram": trace.staleness_histogram(),
        "records": [record.__dict__ for record in trace.records],
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

