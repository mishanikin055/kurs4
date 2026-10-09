"""Rebuild numeric reports and blind human-review forms from immutable captions."""

import csv
import json
import random
from pathlib import Path
from typing import Any

from annotation.common import specs
from annotation.metrics import quality
from detection.common import ROOT, object_hash, read_json, sha256, write_json


def read_samples(run: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    meta, dataset = read_json(run / "run.json"), read_json(run / "dataset.json")
    samples = [json.loads(s) for s in (run / "samples.jsonl").read_text().splitlines()]
    expected = [r["image_id"] for r in dataset["images"]]
    if [r["image_id"] for r in samples] != expected or len(expected) != len(set(expected)):
        raise ValueError("Missing, duplicate or reordered samples")
    if meta["status"] != "complete" or meta["completed_images"] != len(samples):
        raise ValueError("Participant incomplete")
    if object_hash(dataset) != meta["identity"]["dataset_hash"]:
        raise ValueError("Dataset identity changed")
    for sample in samples:
        if not isinstance(sample["raw_caption"], str) or not sample["raw_caption"].strip():
            raise ValueError("Empty/invalid caption")
        if sample["output_tokens"] != len(sample["output_token_ids"]):
            raise ValueError("Token contract mismatch")
    return meta, dataset, samples


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(experiment: Path, output: Path) -> None:
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    summary = []
    hashes = {}
    all_samples = {}
    image_rows = None
    expected_ids = None
    for model in specs():
        run = experiment / model
        try:
            meta, dataset, samples = read_samples(run)
        except (ValueError, FileNotFoundError) as error:
            summary.append({"model": model, "status": "incomplete", "error": str(error)})
            continue
        ids = [r["image_id"] for r in samples]
        if expected_ids is not None and ids != expected_ids:
            raise ValueError("Participants used different datasets")
        expected_ids = ids
        image_rows = dataset["images"]
        if (
            sha256(ROOT / "data/coco/annotations/instances_val2017.json")
            != dataset["instances_sha256"]
        ):
            raise ValueError("CHAIR ground-truth instances changed")
        scores, details, per_cider = quality(samples, image_rows)
        write_json(output / f"{model}_chair.json", details)
        write_csv(
            output / f"{model}_per_image.csv",
            [
                {
                    "image_id": s["image_id"],
                    "CIDEr": per_cider[i],
                    "output_tokens": s["output_tokens"],
                    "total_ms": s["timing"]["total_ms"],
                }
                for i, s in enumerate(samples)
            ],
        )
        row = {"model": model, "status": "complete", "images": len(samples), **scores}
        for field in [
            "preprocess_ms",
            "generate_ms",
            "postprocess_ms",
            "total_ms",
            "total_with_read_ms",
            "tokens_per_second",
        ]:
            values = [s["timing"][field] for s in samples]
            row[field + "_mean"] = float(np.mean(values))
            row[field + "_p50"] = float(np.percentile(values, 50))
            row[field + "_p95"] = float(np.percentile(values, 95))
        row.update(
            output_tokens_mean=float(np.mean([s["output_tokens"] for s in samples])),
            words_mean=float(np.mean([len(s["raw_caption"].split()) for s in samples])),
            token_budget_reached=sum(s["output_tokens"] >= 96 for s in samples),
            cold_load_ms=meta["cold_load_ms"],
            **meta["resources"],
        )
        summary.append(row)
        all_samples[model] = {s["image_id"]: s for s in samples}
        hashes[model] = {p.name: sha256(p) for p in run.iterdir() if p.is_file()}
    fields = list(dict.fromkeys(k for r in summary for k in r))
    write_csv(output / "annotation.csv", [{k: r.get(k) for k in fields} for r in summary])
    write_json(
        output / "report_manifest.json",
        {
            "experiment": str(experiment),
            "inputs": hashes,
            "confidence_intervals": "not computed by user instruction",
            "human_scores": "pending",
            "external_dataset": "not provided",
            "SPICE": "not evaluated; evaluator not validated",
            "rows": summary,
        },
    )
    markdown = [
        "# Аннотаторы: COCO test500, один проход",
        "",
        "| Модель | Статус | n | CIDEr | BLEU-4 | ROUGE-L | CHAIRs | CHAIRi | p50, мс | p95, мс | VRAM allocated, MiB | RSS, MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        if r["status"] != "complete":
            markdown.append(
                f"| {r['model']} | incomplete | — | — | — | — | — | — | — | — | — | — |"
            )
        else:
            markdown.append(
                f"| {r['model']} | complete | {r['images']} | {r['CIDEr']:.4f} | {r['BLEU_4']:.4f} | {r['ROUGE_L']:.4f} | {100 * r['CHAIRs']:.2f}% | {100 * r['CHAIRi']:.2f}% | {r['total_ms_p50']:.1f} | {r['total_ms_p95']:.1f} | {r['vram_allocated_peak_bytes'] / 2**20:.0f} | {r['rss_peak_bytes'] / 2**20:.0f} |"
            )
    markdown += [
        "",
        "CIDEr — исходная шкала evaluator (для шкалы 0–100 умножить на 100). CHAIR — GT captions+instances и словарь авторов; метрика учитывает только объекты COCO и не оценивает действия/свойства.",
        "",
        "FP16, batch 1, greedy decoding, максимум 96 новых токенов, один прогрев на dev-изображении. p50/p95 относятся к одному проходу; межпрогонная устойчивость не измерена. Доверительные интервалы не рассчитывались.",
        "",
        "Экспертные оценки и внешний разрешённый набор отсутствуют; заготовка слепой оценки подготовлена, оценки не заполнены. SPICE не вычислялся.",
    ]
    (output / "annotation.md").write_text("\n".join(markdown) + "\n")
    if len(all_samples) == 4:
        blind_review(output, all_samples, image_rows)


def blind_review(
    output: Path, all_samples: dict[str, dict[int, dict[str, Any]]], images: list[dict[str, Any]]
) -> None:
    rng = random.Random(42)
    selected = rng.sample(images, 100)
    rows = []
    for image in selected:
        for model, samples in all_samples.items():
            rows.append(
                {
                    "model": model,
                    "image_id": image["image_id"],
                    "image_path": image["path"],
                    "caption": samples[image["image_id"]]["raw_caption"],
                }
            )
    rng.shuffle(rows)
    key, form = [], []
    for i, row in enumerate(rows):
        identifier = f"review-{i + 1:04d}"
        key.append({"review_id": identifier, "model": row["model"], "image_id": row["image_id"]})
        form.append(
            {
                "review_id": identifier,
                "image_path": row["image_path"],
                "raw_caption": row["caption"],
                "factual_correctness_1_5": "",
                "completeness_1_5": "",
                "coherence_1_5": "",
                "search_usefulness_1_5": "",
                "invented_objects": "",
                "invented_actions": "",
                "invented_properties": "",
                "notes": "",
            }
        )
    write_csv(output / "expert_blind_form.csv", form)
    write_csv(output / "expert_private_key.csv", key)
