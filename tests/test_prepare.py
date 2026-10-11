"""Checks on frozen sources and data import without network or ML weights."""

import json

import pytest

from scripts import prepare


def test_frozen_manifest_is_not_rewritten_or_replaced(tmp_path):
    path = tmp_path / "manifest.json"
    prepare.freeze_manifest(path, {"seed": 42})
    stat = path.stat()
    prepare.freeze_manifest(path, {"seed": 42})
    assert path.stat().st_mtime_ns == stat.st_mtime_ns
    with pytest.raises(ValueError):
        prepare.freeze_manifest(path, {"seed": 43})
    assert json.loads(path.read_text()) == {"seed": 42}


def test_balanced_selection_is_seeded_and_independent_of_input_order():
    rows = [{"image_id": i, "target_index": i // 10} for i in range(30)]
    selection = prepare.balanced_selection(rows, 42, 2)
    assert selection == prepare.balanced_selection(list(reversed(rows)), 42, 2)
    assert len(selection) == 6
    assert len({r["image_id"] for r in selection}) == 6


def test_source_filename_embedded_synset_is_verified():
    assert (
        prepare.source_image_name("ILSVRC2012_val_00000001_n01440764.JPEG", "n01440764")
        == "ILSVRC2012_val_00000001.JPEG"
    )
    with pytest.raises(ValueError):
        prepare.source_image_name("ILSVRC2012_val_00000001_n01440764.JPEG", "n01443537")


def test_reference_label_order_is_verified():
    classes = prepare.labels()
    names = [r["name"] for r in classes]
    prepare.validate_label_names(names, classes)
    names[0], names[1] = names[1], names[0]
    with pytest.raises(ValueError):
        prepare.validate_label_names(names, classes)


def test_pinned_download_rejects_corrupted_existing_file(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"bad")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        prepare._download_file_attempt(
            "https://unused.invalid", path, {"size": 3, "sha256": "0" * 64}
        )
    assert path.read_bytes() == b"bad"


def test_verify_weights_checks_pinned_hash_even_if_manifest_matches_file(tmp_path, monkeypatch):
    base = tmp_path / "detection"
    (base / "models/m").mkdir(parents=True)
    (base / "configs").mkdir()
    path = base / "models/m/weights.bin"
    path.write_bytes(b"bad")
    spec = {"id": "m", "files": [{"path": "weights.bin", "sha256": "0" * 64}]}
    prepare.write_json(base / "configs/models.json", {"models": [spec]})
    prepare.write_json(
        base / "models/manifest.json",
        {
            "models": {
                "m": {
                    "status": "verified",
                    "spec_hash": prepare.object_hash(spec),
                    "files": [{"path": "weights.bin", "size": 3, "sha256": prepare.sha256(path)}],
                }
            }
        },
    )
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="pinned artifact"):
        prepare.verify_weights("detection", "m")


def test_data_path_cannot_escape_data_directory():
    with pytest.raises(ValueError):
        prepare.data_path("../secrets/weights")


def test_archive_checksums_are_full_sha256():
    assert set(prepare.COCO_ARCHIVES) == {"annotations_trainval2017.zip", "val2017.zip"}
    assert all(len(spec["sha256"]) == 64 for spec in prepare.COCO_ARCHIVES.values())
