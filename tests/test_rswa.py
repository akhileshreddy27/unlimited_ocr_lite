"""Correctness tests for R-SWA on the real model.

1. n = infinity  -> identical tokens to vanilla full attention.
2. small n       -> identical tokens until the first eviction; cache never exceeds L_m + n.
3. The incremental R-SWA decoder (cache eviction) gives the same logits as one
   teacher-forced forward pass with the explicit R-SWA attention mask. This shows
   the cache trick computes exactly the attention pattern in the paper's Eq. (1)-(4).

Run:  .venv/Scripts/python -m pytest tests -q -s
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uocr_lite.model import DocVLM  # noqa: E402
from uocr_lite.pipeline import pdf_to_images  # noqa: E402
from uocr_lite.rswa import (get_cache, greedy_decode, kv_cache_tokens,  # noqa: E402
                            rswa_attention_mask, step_kwargs, trim_cache)

SAMPLE = ROOT / "data" / "samples" / "unlimited_ocr_paper.pdf"
N_TOKENS = 96


@pytest.fixture(scope="module")
def setup():
    vlm = DocVLM(longest_edge=512)  # one tile per page -> fast tests
    img = pdf_to_images(SAMPLE, dpi=72, first=1, last=1)
    return vlm, vlm.build_inputs(img)


def _decode(vlm, inputs, window):
    return greedy_decode(vlm.model, inputs, N_TOKENS, eos_token_ids=[-1], window=window)


def test_infinite_window_equals_full_attention(setup):
    vlm, inputs = setup
    full, _ = _decode(vlm, inputs, None)
    rswa, _ = _decode(vlm, inputs, 10_000)
    assert full == rswa


def test_small_window_bounds_cache(setup):
    vlm, inputs = setup
    n = 16
    full, _ = _decode(vlm, inputs, None)
    rswa, st = _decode(vlm, inputs, n)
    L = inputs["input_ids"].shape[1]
    assert max(st.kv_tokens) <= L + n
    # before the window fills, nothing is evicted, so outputs must agree
    assert full[:n + 1] == rswa[:n + 1]


def test_ring_buffer_matches_trim(setup):
    """Optimized in-place ring cache == simple cat-and-evict cache, token for token,
    and its memory never grows after prefill."""
    vlm, inputs = setup
    n = 16
    trim, _ = greedy_decode(vlm.model, inputs, N_TOKENS, eos_token_ids=[-1], window=n)
    ring, st = greedy_decode(vlm.model, inputs, N_TOKENS, eos_token_ids=[-1], window=n,
                             cache_impl="ring")
    assert ring == trim
    assert len(set(st.kv_bytes)) == 1          # preallocated once, constant afterwards


@torch.inference_mode()
def test_cache_eviction_matches_explicit_mask(setup):
    vlm, inputs = setup
    model, n, steps = vlm.model, 8, 40
    L = inputs["input_ids"].shape[1]

    # incremental R-SWA: record logits for each decode step
    out = model(**inputs, use_cache=True)
    cache, tok = get_cache(out), int(out.logits[0, -1].argmax())
    toks, inc_logits = [tok], []
    for i in range(steps):
        out = model(input_ids=torch.tensor([[tok]]), **step_kwargs(cache, L + i))
        cache = get_cache(out)
        trim_cache(cache, L, n)
        assert kv_cache_tokens(cache) <= L + n
        inc_logits.append(out.logits[0, -1])
        tok = int(out.logits[0, -1].argmax())
        toks.append(tok)

    # one teacher-forced pass over prefix + generated tokens with the explicit mask
    ids = torch.cat([inputs["input_ids"], torch.tensor([toks[:steps]])], dim=1)
    T = ids.shape[1]
    mask = rswa_attention_mask(L, T, n)
    tf_inputs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    tf = model(input_ids=ids, attention_mask=mask, position_ids=torch.arange(T)[None],
               use_cache=False, **tf_inputs)
    tf_logits = tf.logits[0, L:L + steps]
    inc = torch.stack(inc_logits)
    assert torch.allclose(inc, tf_logits, atol=2e-3, rtol=1e-3), (inc - tf_logits).abs().max()
