"""Annotation comparison entry point."""

import argparse
from pathlib import Path

from annotation.common import ROOT, specs


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download")
    d.add_argument("--model", choices=list(specs()))
    sub.add_parser("prepare")
    for name in ["smoke", "dev", "benchmark"]:
        b = sub.add_parser(name)
        b.add_argument("--model", choices=list(specs()))
        b.add_argument("--output", type=Path, required=True)
        if name != "benchmark":
            b.add_argument("--limit", type=int, default=1 if name == "smoke" else 100)
    w = sub.add_parser("_worker")
    w.add_argument("run", type=Path)
    r = sub.add_parser("report")
    r.add_argument("--experiment", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "download":
        from annotation.download import download

        download([args.model] if args.model else list(specs()))
    elif args.command == "prepare":
        from annotation.datasets import prepare

        prepare()
    elif args.command == "_worker":
        from annotation.runner import worker

        worker(args.run)
    elif args.command == "report":
        from annotation.report import report

        report(args.experiment, args.output)
    else:
        from annotation.runner import benchmark

        mode = "test" if args.command == "benchmark" else args.command
        dataset = ROOT / (
            "annotation/configs/test500.json"
            if mode == "test"
            else "annotation/configs/dev100.json"
        )
        benchmark(dataset, args.output, args.model, getattr(args, "limit", None), mode)


if __name__ == "__main__":
    main()
