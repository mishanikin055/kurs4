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


def test_unanswerable_is_not_translated():
    tokenizer = MagicMock()
    assert models.translate_caption(" unanswerable ", (tokenizer, object())) is None
    tokenizer.assert_not_called()


@pytest.mark.parametrize("mode", ["valid", "long", "empty", "unfinished", "limit"])
def test_translation_checks_context_and_complete_output(monkeypatch, mode):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    tokenizer, model = MagicMock(), MagicMock()
    ids = SimpleNamespace(shape=(1, 513 if mode == "long" else 6))
    tokenizer.return_value = {"input_ids": ids}
    model.config.max_position_embeddings = 512
    model.config.eos_token_id = 0
    model.generate.return_value[0][-1].item.return_value = 1 if mode == "unfinished" else 0
    model.generate.return_value[0].__len__.return_value = 512 if mode == "limit" else 10
    tokenizer.decode.return_value = " " if mode == "empty" else "Кошка."
    if mode == "valid":
        assert models.translate_caption("A cat.", (tokenizer, model)) == "Кошка."
        tokenizer.assert_called_once_with("A cat.", return_tensors="pt", truncation=False)
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
    tokenizer.return_value = {"input_ids": SimpleNamespace(shape=(1, 6))}
    model.config.max_position_embeddings = 512
    model.config.eos_token_id = 0
    model.generate.return_value[0][-1].item.return_value = 0
    model.generate.return_value[0].__len__.return_value = 10
    tokenizer.decode.return_value = "Кошка."
    assert (
        models.translate_caption("In this image, I can see a cat.", (tokenizer, model)) == "Кошка."
    )
    tokenizer.assert_called_once_with("a cat.", return_tensors="pt", truncation=False)


def test_translation_keeps_each_sentence(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    tokenizer, model = MagicMock(), MagicMock()
    tokenizer.return_value = {"input_ids": SimpleNamespace(shape=(1, 6))}
    model.config.max_position_embeddings = 512
    model.config.eos_token_id = 0
    model.generate.return_value[0][-1].item.return_value = 0
    model.generate.return_value[0].__len__.return_value = 10
    tokenizer.decode.side_effect = ["Автомобиль на земле.", "На фоне деревья."]
    source = "In this image we can see a vehicle on the ground. In the background there are trees."
    assert (
        models.translate_caption(source, (tokenizer, model))
        == "Автомобиль на земле. На фоне деревья."
    )
    assert [call.args[0] for call in tokenizer.call_args_list] == [
        "a vehicle on the ground.",
        "In the background there are trees.",
    ]
