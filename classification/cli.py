from __future__ import annotations

import argparse
import signal
from pathlib import Path

from classification.common import CONFIGS, registry, verify_model
from detection.common import ROOT, sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сравнение четырёх готовых ImageNet классификаторов"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--model", choices=[m["id"] for m in registry()["models"]])
    download.add_argument("--dry-run", action="store_true")
    commands.add_parser("verify")
    fetch = commands.add_parser("fetch-data")
    fetch.add_argument("--dry-run", action="store_true")
    fetch.add_argument("--token-file", type=Path, default=Path("/run/project-secrets/hf_token"))
    commands.add_parser("import-data")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--images-dir", type=Path, default=ROOT / "data/imagenet/val")
    prepare.add_argument("--seed", type=int, default=42)
    for name in ["smoke", "benchmark"]:
        bench = commands.add_parser(name)
        bench.add_argument("--output-dir", type=Path, required=True)
        bench.add_argument("--config", type=Path, default=CONFIGS / "benchmark.json")
        bench.add_argument("--resume", action="store_true")
        if name == "benchmark":
            bench.add_argument(
                "--model",
                choices=[m["id"] for m in registry()["models"]],
                help="Выполнить только одну модель общего протокола",
            )
            bench.add_argument(
                "--manifest", type=Path, default=ROOT / "data/imagenet/evaluation5000.json"
            )
            bench.add_argument("--limit", type=int)
        else:
            bench.add_argument(
                "--image", type=Path, default=ROOT / "data/coco/smoke/000000397133.jpg"
            )
    for name in ["_worker", "_evaluate"]:
        internal = commands.add_parser(name)
        internal.add_argument("--run-dir", type=Path, required=True)
        internal.add_argument("--resume", action="store_true")
    bootstrap = commands.add_parser("bootstrap")
    bootstrap.add_argument("--runs-dir", type=Path, required=True)
    bootstrap.add_argument("--samples", type=int, default=1000)
    bootstrap.add_argument("--seed", type=int, default=42)
    report = commands.add_parser("report")
    report.add_argument("--runs-dir", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, required=True)
    report.add_argument("--repeats", type=int, help="Выбрать первые N повторов только для отчёта")
    args = parser.parse_args()

    def terminate(signum: int, frame: object) -> None:
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    if args.command == "download":
        from classification.download import download_models

        download_models(args.model, args.dry_run)
    elif args.command == "verify":
        for spec in registry()["models"]:
            verify_model(spec["id"])
            print(spec["id"], "SHA-256 OK")
    elif args.command in {"fetch-data", "import-data", "prepare"}:
        from classification.datasets import fetch_validation, import_huggingface, prepare_imagenet

        if args.command == "fetch-data":
            fetch_validation(args.token_file, args.dry_run)
        elif args.command == "import-data":
            import_huggingface()
        else:
            prepare_imagenet(args.images_dir, args.seed)
    elif args.command in {"benchmark", "smoke"}:
        from classification.runner import run_benchmark

        if args.command == "smoke":
            from PIL import Image

            with Image.open(args.image) as image:
                width, height = image.size
            args.output_dir.mkdir(parents=True, exist_ok=True)
            path = args.output_dir / "smoke_manifest.json"
            write_json(
                path,
                {
                    "schema_version": "1.0",
                    "split": "smoke",
                    "mode": "whole_image",
                    "images": [
                        {
                            "image_id": args.image.stem,
                            "path": str(args.image.resolve().relative_to(ROOT)),
                            "sha256": sha256(args.image),
                            "width": width,
                            "height": height,
                        }
                    ],
                },
            )
            run_benchmark(args.config, path, args.output_dir, None, args.resume, True)
        else:
            run_benchmark(
                args.config,
                args.manifest,
                args.output_dir,
                args.limit,
                args.resume,
                only_model=args.model,
            )
    elif args.command == "_worker":
        from classification.runner import worker

        worker(args.run_dir, args.resume)
    elif args.command == "_evaluate":
        from classification.metrics import evaluate_run

        evaluate_run(args.run_dir)
    elif args.command == "bootstrap":
        from classification.metrics import bootstrap_runs
        from classification.report import build_report

        bootstrap_runs(args.runs_dir, args.samples, args.seed)
        build_report(args.runs_dir, args.runs_dir / "comparison")
    else:
        from classification.report import build_report

        build_report(args.runs_dir, args.output_dir, repeats=args.repeats)


if __name__ == "__main__":
    main()
