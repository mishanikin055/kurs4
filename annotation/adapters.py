"""Native Transformers adapters; image only, independent of detector outputs."""

import time
from typing import Any

from PIL import Image

from annotation.common import MODELS


class Adapter:
    def __init__(self, model_id: str, config: dict[str, Any]):
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoProcessor,
            BlipForConditionalGeneration,
            Qwen3VLForConditionalGeneration,
            SmolVLMForConditionalGeneration,
        )

        self.model_id, self.config = model_id, config
        self.torch = torch
        self.device = config["device"]
        self.dtype = torch.float16
        path = str(MODELS / model_id)
        classes = {
            "blip_base": BlipForConditionalGeneration,
            "florence2_base_ft": AutoModelForCausalLM,
            "qwen3_vl_2b": Qwen3VLForConditionalGeneration,
            "smolvlm2_500m": SmolVLMForConditionalGeneration,
        }
        kwargs = {"local_files_only": True, "trust_remote_code": False, "use_fast": False}
        processor_class = AutoProcessor
        if model_id == "florence2_base_ft":
            kwargs["trust_remote_code"] = True
            kwargs["use_fast"] = (
                True  # official Florence lacks merges.txt; tokenizer.json is complete
            )
        if model_id == "qwen3_vl_2b":
            kwargs.update(
                min_pixels=config["qwen_min_pixels"], max_pixels=config["qwen_max_pixels"]
            )
        self.processor = processor_class.from_pretrained(path, **kwargs)
        model_kwargs = {
            "torch_dtype": self.dtype,
            "local_files_only": True,
            "trust_remote_code": model_id == "florence2_base_ft",
            "low_cpu_mem_usage": True,
            "attn_implementation": config["attn_implementation"][model_id],
        }
        # Direct CUDA placement avoids an extra complete CPU copy of Qwen's weights.
        model_kwargs["device_map"] = {"": self.device}
        self.model = classes[model_id].from_pretrained(path, **model_kwargs).eval()
        if model_id == "florence2_base_ft":
            # Microsoft code uses tuple caches. Disable new DynamicCache initialization
            # without modifying the pinned model source, weights or decoder math.
            self.model.language_model._supports_default_dynamic_cache = lambda: False
        self.parameters = sum(p.numel() for p in self.model.parameters())

    def sync(self) -> None:
        self.torch.cuda.synchronize()

    def predict(self, image: Image.Image) -> dict[str, Any]:
        self.sync()
        start = time.perf_counter()
        rendered = None
        if self.model_id == "blip_base":
            inputs = self.processor(images=image, return_tensors="pt")
            prompt = None
        elif self.model_id == "florence2_base_ft":
            prompt = self.config["florence_task"]
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        else:
            prompt = self.config["prompt"]
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
            ]
            rendered = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=rendered, images=[image], return_tensors="pt")
        inputs = inputs.to(self.device)
        for key, value in inputs.items():
            if value.is_floating_point():
                inputs[key] = value.to(self.dtype)
        self.sync()
        pre_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.config["max_new_tokens"],
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
        self.sync()
        gen_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        ids = output[0]
        if self.model_id in {"qwen3_vl_2b", "smolvlm2_500m"}:
            ids = ids[inputs["input_ids"].shape[-1] :]
        else:
            ids = ids[1:]  # encoder-decoder initial token is not newly generated
        raw = self.processor.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        tokens = len(ids)
        self.sync()
        post_ms = (time.perf_counter() - start) * 1000
        return {
            "raw_caption": raw,
            "output_token_ids": ids.tolist(),
            "output_tokens": tokens,
            "prompt": prompt,
            "rendered_prompt": rendered,
            "input_tokens": inputs["input_ids"].shape[-1] if "input_ids" in inputs else None,
            "timing": {
                "preprocess_ms": pre_ms,
                "generate_ms": gen_ms,
                "postprocess_ms": post_ms,
                "total_ms": pre_ms + gen_ms + post_ms,
                "tokens_per_second": tokens / (gen_ms / 1000),
            },
        }
