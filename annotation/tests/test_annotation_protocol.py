"""CPU protocol/evaluator checks; real GPU smoke remains separate."""

import csv
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from annotation.evaluation.chair_words import CHAIR
from scripts.annotation import common


def test_single_pass_and_no_ci():
    config = common.read_json(common.CONFIG)
    common.validate_config(config)
    for changes in [
        {"repeats": 2},
        {"batch_size": 2},
        {"bootstrap_samples": 1},
        {"confidence_intervals": True},
        {"do_sample": True},
    ]:
        with pytest.raises(ValueError):
            common.validate_config({**config, **changes})


def test_historical_manifest_hash_is_compatible():
    dataset = common.read_json(common.ROOT / "annotation/configs/test500.json")
    assert common.object_hash({k: v for k, v in dataset.items() if k != "hash"}) == dataset["hash"]
    identity = common.comparison_identity(2)
    assert len(identity["image_ids"]) == identity["expected_images"] == 2
    assert identity["shortened"] is True


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


def make_run(path: Path, status="complete"):
    path.mkdir(exist_ok=True)
    common.write_json(
        path / "run.json",
        {
            "status": status,
            "processed": 2,
            "expected_images": 2,
        },
    )
    rows = [
        {
            "image_id": i,
            "raw_caption": " original caption  ",
            "output_tokens": 1,
            "output_token_ids": [4],
        }
        for i in [1, 2]
    ]
    (path / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def test_read_samples_preserves_raw_caption(tmp_path):
    make_run(tmp_path)
    assert common.read_samples(tmp_path, [1, 2])[1][0]["raw_caption"] == " original caption  "


def test_reject_partial_duplicate_reordered_and_wrong_token_count(tmp_path):
    rows = make_run(tmp_path)
    for invalid in [
        rows[:1],
        [rows[0], rows[0]],
        list(reversed(rows)),
        [{**rows[0], "output_tokens": 9}, rows[1]],
    ]:
        (tmp_path / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in invalid))
        with pytest.raises(ValueError):
            common.read_samples(tmp_path, [1, 2])


def test_report_keeps_four_rows_and_error_states(tmp_path, capsys):
    identity = {
        "expected_images": 2,
        "full_dataset_images": 500,
        "shortened": True,
        "image_ids": [1, 2],
    }
    common.write_json(tmp_path / "comparison.json", identity)
    for model, status in zip(common.MODEL_IDS[1:], ["oom", "failed", "running"]):
        (tmp_path / model).mkdir()
        common.write_json(
            tmp_path / model / "run.json",
            {
                "identity": identity,
                "status": status,
                "processed": 1,
            },
        )
    common.report(tmp_path)
    rows = list(csv.DictReader((tmp_path / "comparison.csv").open()))
    assert [row["status"] for row in rows] == ["not_run", "oom", "failed", "incomplete"]
    assert len(rows) == 4
    assert "Короткая проверка" in (tmp_path / "comparison.md").read_text()
    console = capsys.readouterr().out
    assert "CIDEr" in console
    assert all(model in console for model in common.MODEL_IDS)
    assert all(status in console for status in ("not_run", "oom", "failed", "incomplete"))


def test_report_rejects_mixed_identity(tmp_path):
    common.write_json(tmp_path / "comparison.json", {"expected_images": 2, "shortened": True})
    (tmp_path / "blip_base").mkdir()
    common.write_json(tmp_path / "blip_base/run.json", {"identity": {"expected_images": 3}})
    with pytest.raises(ValueError, match="different"):
        common.report(tmp_path)


def test_complete_report_computes_real_lexical_metrics(tmp_path, monkeypatch):
    identity = common.comparison_identity(2)
    common.write_json(tmp_path / "comparison.json", identity)
    images = common.dataset_rows(identity)
    run = tmp_path / "blip_base"
    run.mkdir()
    samples = [
        {
            "image_id": image["image_id"],
            "raw_caption": image["references"][0],
            "output_tokens": 1,
            "output_token_ids": [4],
            "timing": {
                name: 10
                for name in [
                    "preprocess_ms",
                    "generate_ms",
                    "postprocess_ms",
                    "total_ms",
                    "total_with_read_ms",
                    "tokens_per_second",
                ]
            },
        }
        for image in images
    ]
    (run / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in samples))
    common.write_json(
        run / "run.json",
        {
            "identity": identity,
            "status": "complete",
            "processed": 2,
            "expected_images": 2,
            "cold_load_ms": 5,
            "resources": {
                "rss_peak_bytes": 100,
                "vram_allocated_peak_bytes": 10,
                "vram_reserved_peak_bytes": 20,
            },
        },
    )
    common.report(tmp_path)
    row = list(csv.DictReader((tmp_path / "comparison.csv").open()))[0]
    assert row["status"] == "complete"
    assert float(row["CHAIRs"]) == 0
    assert float(row["CHAIRi"]) == 0
    assert float(row["BLEU_4"]) > 0.99
    assert float(row["CIDEr"]) > 0

    original = common.quality

    def broken_evaluator(*_args):
        raise RuntimeError("evaluator unavailable")

    monkeypatch.setattr(common, "quality", broken_evaluator)
    common.report(tmp_path)
    row = list(csv.DictReader((tmp_path / "comparison.csv").open()))[0]
    assert row["status"] == "evaluation_failed"
    assert "CIDEr" not in row
    assert common.read_json(run / "run.json")["status"] == "evaluation_failed"
    monkeypatch.setattr(common, "quality", original)
    common.report(tmp_path)
    assert common.read_json(run / "run.json")["status"] == "complete"


def test_lock_prevents_second_owner_and_survives_parent_close(tmp_path):
    child = None
    try:
        with common.file_lock(tmp_path / "inference.lock") as fd:
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(10)"], pass_fds=(fd,)
            )
            with pytest.raises(RuntimeError):
                with common.file_lock(tmp_path / "inference.lock"):
                    pass
        with pytest.raises(RuntimeError):
            with common.file_lock(tmp_path / "inference.lock"):
                pass
    finally:
        if child is not None:
            child.terminate()
            child.wait()
    with common.file_lock(tmp_path / "inference.lock"):
        pass


def test_sigterm_stops_model_process_group(tmp_path):
    module = Path(common.__file__).resolve()
    child_script = tmp_path / "model.py"
    child_script.write_text(
        "import subprocess, sys, time\n"
        f"from pathlib import Path\np = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"Path({str(tmp_path / 'grandchild.pid')!r}).write_text(str(p.pid))\n"
        "time.sleep(60)\n"
    )
    launcher = tmp_path / "parent.py"
    launcher.write_text(
        "import importlib.util\nfrom pathlib import Path\n"
        f"s=importlib.util.spec_from_file_location('common', {str(module)!r})\n"
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
        f"with m.file_lock(Path({str(tmp_path / 'lock')!r})) as fd:\n"
        f" m.run_child(Path({str(child_script)!r}), Path({str(tmp_path)!r}), fd)\n"
    )
    parent = subprocess.Popen([sys.executable, str(launcher)])
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "grandchild.pid").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (tmp_path / "grandchild.pid").exists()
        pid = int((tmp_path / "grandchild.pid").read_text())
        parent.send_signal(signal.SIGTERM)
        parent.wait(timeout=15)
        deadline = time.monotonic() + 3
        while Path(f"/proc/{pid}/stat").exists() and time.monotonic() < deadline:
            if Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z":
                break
            time.sleep(0.02)
        if Path(f"/proc/{pid}/stat").exists():
            assert Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()


@pytest.mark.parametrize("model", common.MODEL_IDS)
def test_model_scripts_import_without_ml_runtime(model):
    path = common.ROOT / "scripts/annotation" / f"{model}.py"
    subprocess.run([sys.executable, str(path), "--help"], check=True, capture_output=True)


def test_image_paths_cannot_escape_data(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    assert common.image_path("data/image.jpg") == tmp_path / "data/image.jpg"
    for invalid in ["../image.jpg", "/etc/passwd", "data/../secret.jpg"]:
        with pytest.raises(ValueError, match="escapes"):
            common.image_path(invalid)
    (tmp_path / "data/link").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="escapes"):
        common.image_path("data/link/image.jpg")


def test_cgroup_oom_requires_signal_and_real_counter():
    assert common.child_status(-9, {"oom_kill": 3}, {"oom_kill": 4}) == "oom"
    assert common.child_status(-9, {"oom_kill": 3}, {"oom_kill": 3}) == "incomplete"
    assert common.child_status(-15, {"oom_kill": 3}, {"oom_kill": 4}) == "incomplete"
