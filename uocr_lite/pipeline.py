"""PDF -> page images -> VLM (DocTags) -> Markdown.

Two modes, mirroring the reference repo:
  * page-wise      : one request per page, then stitch pages into one document.
  * long-horizon   : a window of several pages in ONE request (all images in the
                     reference prefix), decoded continuously. Like the reference's
                     SGLANG_UOCR_DOC_*_PAGE_WINDOW, a page window bounds the prefix.
Either mode can run with full attention (window=None) or R-SWA (window=n).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pymupdf as fitz
from PIL import Image

from .model import DocVLM, PAGE_PROMPT
from .rswa import NoRepeatNGram, greedy_decode

PAGE_BREAK = "<page_break>"


def pdf_to_images(pdf_path: str | Path, dpi: int = 144, first: int = 1,
                  last: int | None = None) -> list[Image.Image]:
    doc = fitz.open(pdf_path)
    last = last or doc.page_count
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    imgs = []
    for i in range(first - 1, min(last, doc.page_count)):
        pix = doc[i].get_pixmap(matrix=mat)
        imgs.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    doc.close()
    return imgs


def clean_doctags(text: str) -> str:
    text = text.strip()
    for t in ("<end_of_utterance>", "<|end_of_text|>"):
        text = text.replace(t, "")
    return text.strip()


def doctags_to_markdown(doctags_pages: list[str], images: list[Image.Image]) -> str:
    """Use docling-core to turn DocTags (+ page images for picture crops) into Markdown."""
    from docling_core.types.doc import DoclingDocument
    from docling_core.types.doc.document import DocTagsDocument

    dt = DocTagsDocument.from_doctags_and_image_pairs(doctags_pages, images)
    doc = DoclingDocument.load_from_doctags(dt, document_name="Document")
    return doc.export_to_markdown()


def split_pages(doctags: str, n_pages: int) -> list[str]:
    """Split a long-horizon output on <page_break>; pad/merge to exactly n_pages."""
    parts = [p.strip() for p in doctags.split(PAGE_BREAK)]
    if len(parts) > n_pages:
        parts = parts[:n_pages - 1] + ["\n".join(parts[n_pages - 1:])]
    parts += [""] * (n_pages - len(parts))
    # every page must be wrapped in <doctag>...</doctag> for docling-core
    out = []
    for p in parts:
        p = re.sub(r"</?doctag>", "", p).strip()
        out.append(f"<doctag>{p}</doctag>")
    return out


class Parser:
    def __init__(self, vlm: DocVLM, window: int | None = None, max_new_tokens: int = 4096,
                 no_repeat: NoRepeatNGram | None = NoRepeatNGram(35, 1024)):
        self.vlm, self.window = vlm, window
        self.max_new_tokens, self.no_repeat = max_new_tokens, no_repeat

    def _run(self, images, prompt, max_new_tokens):
        inputs = self.vlm.build_inputs(images, prompt)
        ids, stats = greedy_decode(self.vlm.model, inputs, max_new_tokens,
                                   self.vlm.eos_token_ids, window=self.window,
                                   no_repeat=self.no_repeat)
        return clean_doctags(self.vlm.decode_ids(ids)), stats

    def parse_pagewise(self, images: list[Image.Image]) -> tuple[list[str], list[dict]]:
        pages, stats = [], []
        for i, img in enumerate(images, 1):
            t = time.perf_counter()
            dt, st = self._run([img], PAGE_PROMPT, self.max_new_tokens)
            pages.append(dt)
            stats.append({"page": i, **st.summary(), "wall_s": round(time.perf_counter() - t, 1)})
            print(json.dumps(stats[-1]), flush=True)
        return pages, stats

    def parse_long_horizon(self, images: list[Image.Image], page_window: int = 4
                           ) -> tuple[list[str], list[dict]]:
        pages, stats = [], []
        for s in range(0, len(images), page_window):
            chunk = images[s:s + page_window]
            prompt = PAGE_PROMPT if len(chunk) == 1 else \
                f"Convert these {len(chunk)} pages to docling, separating pages with <page_break>."
            t = time.perf_counter()
            dt, st = self._run(chunk, prompt, self.max_new_tokens * len(chunk))
            pages += split_pages(dt, len(chunk))
            stats.append({"pages": f"{s + 1}-{s + len(chunk)}", **st.summary(),
                          "page_breaks": dt.count(PAGE_BREAK),
                          "wall_s": round(time.perf_counter() - t, 1)})
            print(json.dumps(stats[-1]), flush=True)
        return pages, stats
