"""Print saved comparison CSV tables to the console; standard library only."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

PRIMARY_COLUMNS = (
    "model",
    "model_id",
    "status",
    "processed",
    "expected_images",
    "shortened",
    "mAP",
    "AP50",
    "precision",
    "recall",
    "top1",
    "top5",
    "macro_f1",
    "CIDEr",
    "BLEU_4",
    "ROUGE_L",
    "CHAIRs",
    "CHAIRi",
    "p50_ms",
    "p95_ms",
    "total_ms_p50",
    "total_ms_p95",
)


def display(value: str) -> str:
    if not value or value == "None":
        return "—"
    try:
        number = float(value)
    except ValueError:
        return value.replace("\n", " ")
    if not math.isfinite(number):
        raise ValueError("Non-finite values in comparison CSV")
    return f"{number:.6g}"


def print_table(path: Path, full: bool = False) -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    if "status" not in fields or not any(k in fields for k in ("model", "model_id")):
        raise ValueError(f"Expected model and status columns: {path}")
    if len(rows) != 4:
        raise ValueError(f"Expected four comparison participants: {path}")
    model_key = "model_id" if "model_id" in fields else "model"
    if len({row[model_key] for row in rows}) != 4:
        raise ValueError(f"Duplicate model in comparison: {path}")
    columns = fields if full else [key for key in PRIMARY_COLUMNS if key in fields]
    cells = [[display(row[key]) for key in columns] for row in rows]
    widths = [max(len(key), *(len(row[i]) for row in cells)) for i, key in enumerate(columns)]
    print(f"\n{path}")
    print(" | ".join(key.ljust(width) for key, width in zip(columns, widths, strict=True)))
    print("-+-".join("-" * width for width in widths))
    for row in cells:
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)))
    print(
        "Качество — исходные доли/шкала CSV; время — мс. Сокращённый запуск не является полным сравнением."
    )
    for row in rows:
        if row.get("error"):
            print(f"{row[model_key]}: {row['error']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Вывести сохранённые сравнительные таблицы в консоль"
    )
    parser.add_argument(
        "--table", type=Path, action="append", required=True, help="CSV; параметр можно повторить"
    )
    parser.add_argument(
        "--full", action="store_true", help="Показать все столбцы, включая измерения ресурсов"
    )
    args = parser.parse_args()
    for path in args.table:
        print_table(path, args.full)


if __name__ == "__main__":
    main()
