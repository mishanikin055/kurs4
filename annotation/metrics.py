"""Official COCO scorers and CHAIR lexical objects against captions+instances GT."""

from collections import defaultdict
from typing import Any

from annotation.evaluation.chair_words import CHAIR
from detection.common import ROOT, read_json, sha256


def quality(
    samples: list[dict[str, Any]], images: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[float]]:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    gt = {r["image_id"]: [{"caption": c} for c in r["references"]] for r in images}
    predictions = {r["image_id"]: [{"caption": r["raw_caption"]}] for r in samples}
    tokenizer = PTBTokenizer()
    gt_tokens, pred_tokens = tokenizer.tokenize(gt), tokenizer.tokenize(predictions)
    bleu, _ = Bleu(4).compute_score(gt_tokens, pred_tokens)
    cider, per_cider = Cider().compute_score(gt_tokens, pred_tokens)
    rouge, _ = Rouge().compute_score(gt_tokens, pred_tokens)
    chair, details = chair_score(samples, images)
    return (
        {"CIDEr": cider, "BLEU_4": bleu[3], "ROUGE_L": rouge, **chair},
        details,
        per_cider.tolist(),
    )


def chair_score(
    samples: list[dict[str, Any]], images: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # Same word-tokenization, pattern singularization, synonyms and special phrase
    # rules as the authors. val2017 replaces their 2014 train/val loader because
    # every evaluated image and all its GT are contained in this single partition.
    tokenizer_source = read_json(ROOT / "annotation/evaluation/sources.json")["nltk_punkt_tab"][
        "files"
    ]
    for name, digest in tokenizer_source.items():
        if (
            sha256(ROOT / "data/annotation_evaluator/nltk_data/tokenizers/punkt_tab/english" / name)
            != digest
        ):
            raise ValueError("NLTK tokenizer resource changed")
    evaluator = CHAIR([r["image_id"] for r in images], None)
    gt = defaultdict(set)
    instances = read_json(ROOT / "data/coco/annotations/instances_val2017.json")
    categories = {r["id"]: r["name"] for r in instances["categories"]}
    for a in instances["annotations"]:
        gt[a["image_id"]].add(evaluator.inverse_synonym_dict[categories[a["category_id"]]])
    for image in images:
        for caption in image["references"]:
            gt[image["image_id"]].update(evaluator.caption_to_words(caption)[1])
    details = []
    total = hallucinated = sentences = 0
    recalled, gt_count = 0, 0
    for sample in samples:
        objects = evaluator.caption_to_words(sample["raw_caption"])[1]
        incorrect = [o for o in objects if o not in gt[sample["image_id"]]]
        total += len(objects)
        hallucinated += len(incorrect)
        sentences += bool(incorrect)
        recalled += len(set(objects) & gt[sample["image_id"]])
        gt_count += len(gt[sample["image_id"]])
        details.append(
            {
                "image_id": sample["image_id"],
                "objects": objects,
                "gt_objects": sorted(gt[sample["image_id"]]),
                "hallucinated": incorrect,
            }
        )
    return {
        "CHAIRs": sentences / len(samples),
        "CHAIRi": hallucinated / total if total else None,
        "object_recall_micro": recalled / gt_count if gt_count else None,
    }, details
