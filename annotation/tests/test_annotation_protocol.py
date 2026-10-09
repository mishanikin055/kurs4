"""CPU protocol and evaluator checks; separate from real model smoke."""

import json
from pathlib import Path

import pytest

from annotation.common import CONFIG, validate_config
from annotation.evaluation.chair_words import CHAIR
from annotation.report import read_samples
from detection.common import file_lock, object_hash, read_json, write_json


def test_single_pass_and_no_ci():
    config = read_json(CONFIG)
    validate_config(config)
    for changes in [
        {"repeats": 2},
        {"batch_size": 2},
        {"bootstrap_samples": 1},
        {"confidence_intervals": True},
        {"do_sample": True},
    ]:
        with pytest.raises(ValueError):
            validate_config({**config, **changes})


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        ("Two men ride bicycles.", ["person", "bicycle"]),
        ("A baby bird near a wine glass.", ["bird", "wine glass"]),
        ("A passenger jet above a traffic light.", ["airplane", "traffic light"]),
        ("A toilet seat.", ["toilet"]),
        ("A hot dog and two dogs.", ["hot dog", "dog"]),
    ],
)
def test_author_chair_lexical_rules(caption, expected):
    assert CHAIR([1], None).caption_to_words(caption)[1] == expected


def make_run(tmp_path: Path):
    dataset = {"images": [{"image_id": 1}, {"image_id": 2}]}
    write_json(tmp_path / "dataset.json", dataset)
    write_json(
        tmp_path / "run.json",
        {
            "status": "complete",
            "completed_images": 2,
            "identity": {"dataset_hash": object_hash(dataset)},
        },
    )
    rows = [
        {
            "image_id": i,
            "raw_caption": "original caption",
            "output_tokens": 1,
            "output_token_ids": [4],
        }
        for i in [1, 2]
    ]
    (tmp_path / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def test_read_samples_preserves_raw_caption(tmp_path):
    make_run(tmp_path)
    assert read_samples(tmp_path)[2][0]["raw_caption"] == "original caption"


def test_reject_partial_and_duplicate_samples(tmp_path):
    rows = make_run(tmp_path)
    for invalid in [rows[:1], [rows[0], rows[0]], list(reversed(rows))]:
        (tmp_path / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in invalid))
        with pytest.raises(ValueError):
            read_samples(tmp_path)


def test_lock_prevents_second_model_owner(tmp_path):
    with file_lock(tmp_path / "inference.lock"):
        with pytest.raises(RuntimeError):
            with file_lock(tmp_path / "inference.lock"):
                pass
