"""Command line interface for explicit downloads and offline experiments."""

from __future__ import annotations

import argparse
import signal
from pathlib import Path

from detection.common import ROOT, read_json, registry, sha256, verify_model, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение четырёх готовых COCO-детекторов")
    commands = parser.add_subparsers(dest="command", required=True)
    d = commands.add_parser("download", help="Скачать и проверить веса")
    d.add_argument("--model", choices=[m["id"] for m in registry()["models"]])
    d.add_argument("--dry-run", action="store_true")
    commands.add_parser("verify", help="Проверить SHA-256 всех скачанных файлов")
    p = commands.add_parser("prepare", help="Зафиксировать COCO 500 dev / 4500 test")
    p.add_argument("--download", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    for name in ["benchmark", "smoke"]:
        b = commands.add_parser(name)
        b.add_argument("--config", type=Path, default=ROOT / "detection/configs/benchmark.json")
        b.add_argument("--output-dir", type=Path, required=True)
        b.add_argument("--model", choices=[m["id"] for m in registry()["models"]])
        b.add_argument("--resume", action="store_true")
        if name == "benchmark":
            b.add_argument("--manifest", type=Path, default=ROOT / "data/coco/test.json")
            b.add_argument("--limit", type=int)
        else:
            b.add_argument("--image", type=Path, default=ROOT / "data/coco/smoke/000000397133.jpg")
    r = commands.add_parser("report")
    r.add_argument("--runs-dir", type=Path, required=True)
    r.add_argument("--output-dir", type=Path, required=True)
    ci = commands.add_parser("bootstrap")
    ci.add_argument("--runs-dir", type=Path, required=True)
    ci.add_argument("--samples", type=int, default=1000)
    ci.add_argument("--seed", type=int, default=42)
    for name in ["_worker", "_evaluate"]:
        internal = commands.add_parser(name)
        internal.add_argument("--run-dir", type=Path, required=True)
        internal.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    def terminate(signum: int, frame: object) -> None:
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    if args.command == "download":
        from detection.download import download_models

        download_models(args.model, args.dry_run)
    elif args.command == "verify":
        for model in registry()["models"]:
            verify_model(model["id"])
            print(model["id"], "SHA-256 OK")
    elif args.command == "prepare":
        from detection.datasets import prepare_dataset

        prepare_dataset(args.download, args.seed)
    elif args.command in {"benchmark", "smoke"}:
        from detection.runner import run_benchmark

        if args.command == "smoke":
            from PIL import Image

            with Image.open(args.image) as image:
                width, height = image.size
            manifest = {
                "schema_version": "1.0",
                "split": "smoke",
                "images": [
                    {
                        "image_id": 397133,
                        "path": str(args.image.resolve().relative_to(ROOT)),
                        "width": width,
                        "height": height,
                        "sha256": sha256(args.image),
                    }
                ],
            }
            args.output_dir.mkdir(parents=True, exist_ok=True)
            path = args.output_dir / "smoke_manifest.json"
            write_json(path, manifest)
            run_benchmark(
                args.config, path, args.output_dir, None, args.resume, args.model, smoke=True
            )
        else:
            run_benchmark(
                args.config, args.manifest, args.output_dir, args.limit, args.resume, args.model
            )
    elif args.command == "_worker":
        from detection.runner import worker

        worker(args.run_dir, args.resume)
    elif args.command == "_evaluate":
        from detection.metrics import evaluate_run

        evaluate_run(args.run_dir)
    elif args.command == "bootstrap":
        if args.samples < 1:
            parser.error("--samples must be positive")
        from detection.metrics import bootstrap_runs
        from detection.report import build_report

        runs = [
            p.parent
            for p in args.runs_dir.glob("*/run.json")
            if read_json(p)["status"] == "complete"
        ]
        bootstrap_runs(runs, args.samples, args.seed, args.runs_dir / "bootstrap.json")
        build_report(args.runs_dir, args.runs_dir / "comparison")
    else:
        from detection.report import build_report

        build_report(args.runs_dir, args.output_dir)


if __name__ == "__main__":
    main()
