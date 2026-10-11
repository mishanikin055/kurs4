"""Офлайн blip_base: готовые веса, штатный processor, FP16 greedy caption."""

import time
from typing import Any

try:
    from .common import MODELS, decode_result, main, move_inputs, sync
except ImportError:  # python scripts/annotation/blip_base.py
    from common import MODELS, decode_result, main, move_inputs, sync


class Model:
    def __init__(self, config: dict[str, Any]):
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.torch, self.config = torch, config
        self.device = config["device"]
        path = str(MODELS / "blip_base")
        self.processor = BlipProcessor.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, use_fast=False
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": self.device},
            attn_implementation="eager",
        ).eval()

    def predict(self, image: Any) -> dict[str, Any]:
        torch = self.torch
        sync(torch)
        started = time.perf_counter()
        prompt, rendered = None, None
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = move_inputs(inputs, self.device, torch)
        sync(torch)
        pre_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.config["max_new_tokens"],
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
        sync(torch)
        gen_ms = (time.perf_counter() - started) * 1000
        # Encoder-decoder initial token is not a newly generated completion token.
        ids = output[0][1:]
        return decode_result(
            self.processor,
            ids,
            prompt,
            rendered,
            inputs["input_ids"].shape[-1] if "input_ids" in inputs else None,
            pre_ms,
            gen_ms,
            torch,
        )


if __name__ == "__main__":
    main(Model, "blip_base")
