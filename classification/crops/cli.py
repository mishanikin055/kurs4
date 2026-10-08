from __future__ import annotations

import argparse
import signal
from pathlib import Path

from detection.common import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение классификаторов на GT и detector crops")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--split", choices=["dev", "test"], default="test")
    prepare.add_argument(
        "--detector-run",
        type=Path,
        default=ROOT / "reports/detection/test-v2/rf_detr_small_repeat1",
    )
    prepare.add_argument(
        "--output-dir", type=Path, default=ROOT / "reports/classification/crop-data-v1"
    )
    bench = sub.add_parser("benchmark")
    bench.add_argument("--manifest", type=Path, required=True)
    bench.add_argument("--output-dir", type=Path, required=True)
    bench.add_argument("--resume", action="store_true")
    bench.add_argument("--limit", type=int)
    for command in ["_worker", "_evaluate"]:
        internal = sub.add_parser(command)
        internal.add_argument("--run-dir", type=Path, required=True)
        internal.add_argument("--resume", action="store_true")
    for command in ["bootstrap", "report"]:
        p = sub.add_parser(command)
        p.add_argument("--runs-dir", type=Path, required=True)
        if command == "bootstrap":
            p.add_argument("--samples", type=int, default=1000)
            p.add_argument("--seed", type=int, default=42)
        else:
            p.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    def terminate(signum: int, frame: object) -> None:
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    if args.command == "prepare":
        from classification.crops.datasets import prepare as prepare_data

        prepare_data(args.split, args.detector_run, args.output_dir)
    elif args.command == "benchmark":
        from classification.crops.runner import benchmark

        benchmark(args.manifest, args.output_dir, args.resume, args.limit)
    elif args.command == "_worker":
        from classification.crops.runner import worker

        worker(args.run_dir, args.resume)
    elif args.command == "_evaluate":
        from classification.crops.metrics import evaluate

        evaluate(args.run_dir)
    else:
        from classification.crops.report import report

        if args.command == "bootstrap":
            from classification.crops.metrics import bootstrap

            bootstrap(args.runs_dir, args.samples, args.seed)
        report(
            args.runs_dir,
            args.output_dir if args.command == "report" else args.runs_dir / "comparison",
        )


if __name__ == "__main__":
    main()
