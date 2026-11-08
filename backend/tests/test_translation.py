"""Russian captions supplement, never replace, Florence's source text."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from PIL import Image

from backend import models
from backend.schemas import AnnotationPayload, ClassifiedObject


@pytest.mark.parametrize("stage", ["annotation", "classification"])
def test_predict_adds_translation_and_preserves_source(monkeypatch, tmp_path, stage):
    original = "  A spotted cat.  "
    object_id = str(uuid4())
    payload = (
        {"raw_caption": original}
        if stage == "annotation"
        else {"objects": [{"object_id": object_id, "object_caption": original}]}
    )
    monkeypatch.setattr(models, "annotate", MagicMock(return_value=payload))
    monkeypatch.setattr(models, "classify", MagicMock(return_value=payload))
    translate = MagicMock(return_value="Пятнистая кошка.")
    monkeypatch.setattr(models, "translate_caption", translate)
    translator = object()
    result = models.predict(
        {"stage": stage, "parameters": {}, "detection": {}},
        tmp_path,
        loaded=object(),
        translator=translator,
        image=Image.new("RGB", (4, 4)),
    )["payload"]
    translate.assert_called_once_with(original, translator)
    if stage == "annotation":
        parsed = AnnotationPayload.model_validate(result)
        assert parsed.raw_caption == original
        assert parsed.russian_caption == "Пятнистая кошка."
    else:
        parsed = ClassifiedObject.model_validate(result["objects"][0])
        assert parsed.object_caption == original
        assert parsed.russian_caption == "Пятнистая кошка."


def test_translation_failure_does_not_publish_english_as_russian(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "annotate", MagicMock(return_value={"raw_caption": "A cat."}))
    monkeypatch.setattr(models, "translate_caption", MagicMock(side_effect=ValueError))
    result = models.predict(
        {"stage": "annotation", "parameters": {}},
        tmp_path,
        loaded=object(),
        translator=object(),
        image=Image.new("RGB", (4, 4)),
    )
    assert result == {"error": {"code": "model_error", "type": "ValueError"}}


@pytest.mark.parametrize("stage", ["annotation", "classification"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_predict_loads_translator_on_requested_device(monkeypatch, tmp_path, stage, device):
    monkeypatch.setattr(models, "annotate", MagicMock(return_value={"raw_caption": "A cat."}))
    monkeypatch.setattr(models, "classify", MagicMock(return_value={"objects": []}))
    loader = MagicMock()
    monkeypatch.setattr(models, "load_translator", loader)
    monkeypatch.setattr(models, "translate_caption", MagicMock(return_value="Кошка."))
    result = models.predict(
        {"stage": stage, "parameters": {"device": device}, "detection": {}},
        tmp_path,
        loaded=object(),
        image=Image.new("RGB", (4, 4)),
    )
    assert "payload" in result
    loader.assert_called_once_with(tmp_path.parent / models.TRANSLATION_SPEC["model_name"], device)


def test_unanswerable_is_not_translated():
    tokenizer = MagicMock()
    assert models.translate_caption(" unanswerable ", (tokenizer, object())) is None
    tokenizer.assert_not_called()


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "cuda:1"])
@pytest.mark.parametrize("mode", ["valid", "long", "empty", "unfinished", "limit"])
def test_translation_checks_context_and_complete_output(monkeypatch, mode, device):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    tokenizer, model = MagicMock(), MagicMock()
    model.device = device
    tokenizer.model_max_length = 512
    ids = MagicMock()
    ids.shape = (1, 513 if mode == "long" else 6)
    ids.to.return_value = ids
    mask = MagicMock()
    tokenizer.return_value = {"input_ids": ids, "attention_mask": mask}
    model.config.max_position_embeddings = 512
    model.config.eos_token_id = 0
    model.generate.return_value[0][-1].item.return_value = 1 if mode == "unfinished" else 0
    model.generate.return_value[0].__len__.return_value = 512 if mode == "limit" else 10
    tokenizer.decode.return_value = " " if mode == "empty" else "Кошка."
    if mode == "valid":
        assert models.translate_caption("A cat.", (tokenizer, model)) == "Кошка."
        tokenizer.assert_called_once_with(">>rus<< A cat.", return_tensors="pt", truncation=False)
        ids.to.assert_called_once_with(device)
        mask.to.assert_called_once_with(device)
        assert model.generate.call_args.kwargs["num_beams"] == 4
    else:
        with pytest.raises(ValueError):
            models.translate_caption("A cat.", (tokenizer, model))
        if mode == "long":
            model.generate.assert_not_called()


def test_russian_caption_requires_original_object():
    with pytest.raises(ValueError):
        ClassifiedObject(object_id=uuid4(), error={"code": "crop_error"}, russian_caption="Кот")
    with pytest.raises(ValueError):
        AnnotationPayload(raw_caption="A cat.", russian_caption="   ")


def test_translation_removes_florence_introduction_only(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    tokenizer, model = MagicMock(), MagicMock()
    tokenizer.model_max_length = 512
    ids = MagicMock()
    ids.shape = (1, 6)
    ids.to.return_value = ids
    tokenizer.return_value = {"input_ids": ids}
    model.config.max_position_embeddings = 512
    model.config.eos_token_id = 0
    model.generate.return_value[0][-1].item.return_value = 0
    model.generate.return_value[0].__len__.return_value = 10
    tokenizer.decode.return_value = "Кошка."
    assert (
        models.translate_caption("In this image, I can see a cat.", (tokenizer, model)) == "Кошка."
    )
    tokenizer.assert_called_once_with(">>rus<< a cat.", return_tensors="pt", truncation=False)


def test_translation_whole_caption_keeps_context(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    tokenizer, model = MagicMock(), MagicMock()
    tokenizer.model_max_length = 512
    ids = MagicMock()
    ids.shape = (1, 6)
    ids.to.return_value = ids
    tokenizer.return_value = {"input_ids": ids}
    model.config.max_position_embeddings = 1024
    model.config.eos_token_id = 0
    model.generate.return_value[0][-1].item.return_value = 0
    model.generate.return_value[0].__len__.return_value = 10
    tokenizer.decode.return_value = "Птица на земле. Она смотрит вдаль."
    source = "In this image we can see a bird on the ground. It looks into the distance."
    assert models.translate_caption(source, (tokenizer, model)) == tokenizer.decode.return_value
    tokenizer.assert_called_once_with(
        ">>rus<< a bird on the ground. It looks into the distance.",
        return_tensors="pt",
        truncation=False,
    )
    assert model.generate.call_args.kwargs["num_beams"] == 4


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:1"])
def test_translator_loads_verified_safetensors_on_selected_device(tmp_path, monkeypatch, device):
    import hashlib

    (tmp_path / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(
        models,
        "TRANSLATION_SPEC",
        {
            "files": [
                {"path": "model.safetensors", "sha256": hashlib.sha256(b"weights").hexdigest()}
            ]
        },
    )
    tokenizer_loader, model_loader = MagicMock(), MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            MarianTokenizer=tokenizer_loader,
            MarianMTModel=model_loader,
        ),
    )
    tokenizer, model = models.load_translator(tmp_path, device)
    tokenizer_loader.from_pretrained.assert_called_once_with(str(tmp_path), local_files_only=True)
    model_loader.from_pretrained.assert_called_once_with(
        str(tmp_path),
        local_files_only=True,
        torch_dtype="float16" if device.startswith("cuda") else "float32",
        use_safetensors=True,
    )
    model_loader.from_pretrained.return_value.to.assert_called_once_with(device)
    assert model is model_loader.from_pretrained.return_value.to.return_value.eval.return_value
    assert tokenizer is tokenizer_loader.from_pretrained.return_value
    (tmp_path / "model.safetensors").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="Translation checkpoint checksum mismatch"):
        models.load_translator(tmp_path, device)
    assert model_loader.from_pretrained.call_count == 1


@pytest.mark.parametrize("tokenizer_limit,model_limit", [(512, 1024), (1024, 512)])
def test_translation_context_includes_target_language_token(
    monkeypatch, tokenizer_limit, model_limit
):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    tokenizer, model = MagicMock(), MagicMock()
    tokenizer.model_max_length = tokenizer_limit
    model.config.max_position_embeddings = model_limit
    tokenizer.return_value = {"input_ids": SimpleNamespace(shape=(1, 513))}
    with pytest.raises(ValueError, match="Caption exceeds translation model context"):
        models.translate_caption("A cat.", (tokenizer, model))
    tokenizer.assert_called_once_with(">>rus<< A cat.", return_tensors="pt", truncation=False)
    model.generate.assert_not_called()
