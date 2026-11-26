"""Load the document VLM and build model inputs for one or many pages."""
from __future__ import annotations

import os

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

DEFAULT_MODEL = os.environ.get("UOCR_MODEL", "ibm-granite/granite-docling-258M")
PAGE_PROMPT = "Convert this page to docling."


class DocVLM:
    """Thin wrapper: processor + model + prompt building. CPU, fp32 by default."""

    def __init__(self, model_id: str = DEFAULT_MODEL, dtype=torch.float32,
                 threads: int | None = None, longest_edge: int | None = None):
        if threads:
            torch.set_num_threads(threads)
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        if longest_edge:  # lower = fewer image tiles = shorter prefix (faster, less accurate)
            self.processor.image_processor.size = {"longest_edge": longest_edge}
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, attn_implementation="sdpa").eval()
        gen = self.model.generation_config
        eos = gen.eos_token_id
        self.eos_token_ids = eos if isinstance(eos, list) else [eos]
        tok = self.processor.tokenizer
        for t in ("<end_of_utterance>", "<|end_of_text|>"):
            tid = tok.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid != tok.unk_token_id and tid not in self.eos_token_ids:
                self.eos_token_ids.append(tid)

    def build_inputs(self, images: list[Image.Image], prompt: str = PAGE_PROMPT) -> dict:
        """All page images go into the reference prefix, followed by the prompt."""
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        return dict(self.processor(text=text, images=[images], return_tensors="pt"))

    def decode_ids(self, ids: list[int]) -> str:
        return self.processor.tokenizer.decode(ids, skip_special_tokens=False)

    def config_summary(self) -> dict:
        c = self.model.config.text_config
        head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        return {
            "model": self.model_id,
            "params_M": round(sum(p.numel() for p in self.model.parameters()) / 1e6, 1),
            "layers": c.num_hidden_layers, "q_heads": c.num_attention_heads,
            "kv_heads": c.num_key_value_heads, "head_dim": head_dim,
            # bytes of K+V per token across all layers (fp32 = 4 bytes)
            "kv_bytes_per_token": 2 * c.num_hidden_layers * c.num_key_value_heads * head_dim
                                  * torch.finfo(self.model.dtype).bits // 8,
        }
