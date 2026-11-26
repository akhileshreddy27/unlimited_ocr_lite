"""Reference Sliding Window Attention (R-SWA) for a pretrained decoder.

Paper rule (Unlimited OCR Works, Sec. 3.4): a decode token at step t attends to
  * every prefix token  P = {1..L_m}          (image tokens + prompt, "the book")
  * the last n generated tokens D_n(t)        (a causal sliding window)
and nothing else. Hence the KV cache never exceeds L_m + n entries:
  C_MHA(T)   = L_m + T
  C_RSWA(T)  = L_m + min(n, T)  <= L_m + n

Implementation notes
--------------------
* Keys are cached *after* RoPE has been applied, so evicting middle entries is
  safe: surviving keys keep their original absolute positions and nothing has
  to be re-rotated. New tokens keep getting increasing absolute positions.
* We drive decoding with our own greedy loop (not ``model.generate``), passing
  ``cache_position``/``position_ids`` explicitly, because the cache length no
  longer equals the number of tokens seen.
* ``window=None`` means vanilla full attention (the baseline).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


# --------------------------------------------------------------------------- #
# KV-cache helpers (work across transformers cache layouts)
# --------------------------------------------------------------------------- #
def get_cache(out):
    """transformers 5.x names the output field ``cache``; 4.x used ``past_key_values``."""
    c = getattr(out, "cache", None)
    return c if c is not None else out.past_key_values


def step_kwargs(cache, pos: int) -> dict:
    """Arguments for one incremental decode step at absolute position ``pos``."""
    return {"cache": cache, "cache_position": torch.tensor([pos]),
            "position_ids": torch.tensor([[pos]]), "use_cache": True}


def _layers(cache):
    """Yield (get_k, get_v, set_kv) accessors for each layer of an HF cache."""
    if hasattr(cache, "layers"):  # transformers >= 4.56: DynamicCache.layers[i].keys/values
        for layer in cache.layers:
            def set_kv(k, v, layer=layer):
                layer.keys, layer.values = k, v
            yield layer.keys, layer.values, set_kv
    else:  # older: key_cache / value_cache lists
        for i in range(len(cache.key_cache)):
            def set_kv(k, v, i=i):
                cache.key_cache[i], cache.value_cache[i] = k, v
            yield cache.key_cache[i], cache.value_cache[i], set_kv


def kv_cache_tokens(cache) -> int:
    for k, _, _ in _layers(cache):
        return 0 if k is None else k.shape[-2]
    return 0


def kv_cache_bytes(cache) -> int:
    total = 0
    for k, v, _ in _layers(cache):
        if k is not None:
            total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total


def trim_cache(cache, prefix_len: int, window: int) -> None:
    """Keep the first ``prefix_len`` (reference) entries + the last ``window`` entries."""
    for k, v, set_kv in _layers(cache):
        if k is None or k.shape[-2] <= prefix_len + window:
            continue
        # [batch, kv_heads, seq, head_dim]; .contiguous() so freed memory is really released
        k = torch.cat([k[..., :prefix_len, :], k[..., -window:, :]], dim=-2).contiguous()
        v = torch.cat([v[..., :prefix_len, :], v[..., -window:, :]], dim=-2).contiguous()
        set_kv(k, v)


# --------------------------------------------------------------------------- #
# Optimized R-SWA cache: preallocated ring buffer, in-place writes
# --------------------------------------------------------------------------- #
from transformers.cache_utils import Cache, DynamicLayer  # noqa: E402


class RingRSWALayer(DynamicLayer):
    """One layer's KV cache as [prefix | ring of n+1 output slots], preallocated once.

    ``trim_cache`` (above) is the simple version: append with torch.cat, then cut the
    middle out with another torch.cat, i.e. two full-cache copies per token per layer.
    This layer writes each new token into slot ``P + t % (n+1)`` in place: O(1) work,
    no reallocation, constant memory. Attention is permutation-invariant over keys
    (RoPE is already baked into each key and no mask is needed inside the window),
    so the ring's wrap-around order does not matter.

    Ring size n+1 = the current token + the previous n, exactly what the incremental
    trim version attends to, so both produce identical tokens (tests/test_rswa.py).
    """

    is_croppable = False

    def __init__(self, window: int):
        super().__init__()
        self.window, self.size = window, window + 1
        self.P = 0       # prefix length (fixed after prefill)
        self.t = 0       # decode tokens written so far

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:  # prefill: allocate prefix + ring once
            B, H, S, D = key_states.shape
            self.keys = key_states.new_empty(B, H, S + self.size, D)
            self.values = value_states.new_empty(B, H, S + self.size, D)
            self.keys[:, :, :S] = key_states
            self.values[:, :, :S] = value_states
            self.P, self.t, self.is_initialized = S, 0, True
            self.device = key_states.device
            return self.keys[:, :, :S], self.values[:, :, :S]
        for i in range(key_states.shape[-2]):  # decode: normally one token
            slot = self.P + self.t % self.size
            self.keys[:, :, slot] = key_states[:, :, i]
            self.values[:, :, slot] = value_states[:, :, i]
            self.t += 1
        valid = self.P + min(self.t, self.size)
        return self.keys[:, :, :valid], self.values[:, :, :valid]   # views, no copy

    def get_seq_length(self) -> int:
        return self.P + min(self.t, self.size) if self.is_initialized else 0

    def get_mask_sizes(self, query_length: int, *args, **kwargs) -> tuple[int, int]:
        if not self.is_initialized:
            return query_length, 0
        return self.P + min(self.t + query_length, self.size), 0


def make_ring_cache(model, window: int) -> Cache:
    n_layers = model.config.text_config.num_hidden_layers
    return Cache(layers=[RingRSWALayer(window) for _ in range(n_layers)])


def rswa_attention_mask(prefix_len: int, total_len: int, window: int | None,
                        dtype=torch.float32, device="cpu") -> torch.Tensor:
    """Additive 4D mask [1,1,T,T] for teacher-forced (training / verification) passes.

    Row q may see column k iff k <= q and (k < prefix_len or q - k <= window).
    The ``q - k <= window`` term means "itself + the previous ``window`` tokens",
    which is exactly what the incremental decoder sees after ``trim_cache``.
    """
    q = torch.arange(total_len, device=device)[:, None]
    k = torch.arange(total_len, device=device)[None, :]
    allowed = k <= q
    if window is not None:
        allowed &= (k < prefix_len) | (q - k <= window)
    mask = torch.zeros(total_len, total_len, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask[None, None]


# --------------------------------------------------------------------------- #
# Greedy decoding with per-step instrumentation
# --------------------------------------------------------------------------- #
@dataclass
class DecodeStats:
    prefix_len: int = 0
    window: int | None = None
    prefill_s: float = 0.0
    step_ms: list[float] = field(default_factory=list)      # latency of each decode step
    kv_tokens: list[int] = field(default_factory=list)      # cache length after each step
    kv_bytes: list[int] = field(default_factory=list)       # cache bytes after each step
    stop_reason: str = ""

    @property
    def n_generated(self) -> int:
        return len(self.step_ms)

    @property
    def decode_tps(self) -> float:
        s = sum(self.step_ms) / 1000
        return self.n_generated / s if s else 0.0

    def summary(self) -> dict:
        n = self.n_generated
        return {
            "window": self.window, "prefix_len": self.prefix_len, "generated": n,
            "prefill_s": round(self.prefill_s, 2),
            "decode_tps": round(self.decode_tps, 2),
            "first_100_ms": round(sum(self.step_ms[:100]) / max(1, min(n, 100)), 2),
            "last_100_ms": round(sum(self.step_ms[-100:]) / max(1, min(n, 100)), 2),
            "peak_kv_MB": round(max(self.kv_bytes, default=0) / 2**20, 2),
            "final_kv_tokens": self.kv_tokens[-1] if self.kv_tokens else 0,
            "stop": self.stop_reason,
        }


class NoRepeatNGram:
    """Ban a token that would repeat an n-gram seen in the last ``lookback`` tokens.

    Same idea as the paper's DeepseekOCRNoRepeatNGramLogitProcessor (ngram 35,
    window 128/1024). Small models are prone to loops on long outputs, and with
    R-SWA the model literally cannot see text older than the window.
    """

    def __init__(self, n: int = 35, lookback: int = 1024):
        self.n, self.lookback = n, lookback

    def banned(self, tokens: list[int]) -> set[int]:
        n = self.n
        if len(tokens) < n:
            return set()
        hist = tokens[-self.lookback:]
        prefix = tuple(hist[-(n - 1):])
        out = set()
        for i in range(len(hist) - n + 1):
            if tuple(hist[i:i + n - 1]) == prefix:
                out.add(hist[i + n - 1])
        return out


@torch.inference_mode()
def greedy_decode(model, inputs: dict, max_new_tokens: int, eos_token_ids,
                  window: int | None = None, no_repeat: NoRepeatNGram | None = None,
                  on_token=None, cache_impl: str = "trim") -> tuple[list[int], DecodeStats]:
    """Prefill the reference (image + prompt) once, then decode token by token.

    window=None -> vanilla full attention; window=n -> R-SWA with n output tokens.
    cache_impl: "trim" (cat + evict, simple) or "ring" (preallocated, in-place).
    Returns (generated_token_ids, stats).
    """
    eos = set([eos_token_ids] if isinstance(eos_token_ids, int) else eos_token_ids)
    prefix_len = inputs["input_ids"].shape[1]
    stats = DecodeStats(prefix_len=prefix_len, window=window)
    ring = window is not None and cache_impl == "ring"

    t0 = time.perf_counter()
    extra = {"cache": make_ring_cache(model, window)} if ring else {}
    out = model(**inputs, use_cache=True, **extra)
    stats.prefill_s = time.perf_counter() - t0
    cache = get_cache(out)
    logits = out.logits[:, -1, :]

    generated: list[int] = []
    pos = prefix_len  # absolute position of the next token
    for _ in range(max_new_tokens):
        if no_repeat is not None:
            for b in no_repeat.banned(generated):
                logits[0, b] = float("-inf")
        tok = int(logits.argmax(-1))
        generated.append(tok)
        if on_token:
            on_token(tok)
        if tok in eos:
            stats.stop_reason = "eos"
            break

        t = time.perf_counter()
        out = model(input_ids=torch.tensor([[tok]]), **step_kwargs(cache, pos))
        cache = get_cache(out)
        if window is not None and not ring:
            trim_cache(cache, prefix_len, window)
        stats.step_ms.append((time.perf_counter() - t) * 1000)
        stats.kv_tokens.append(kv_cache_tokens(cache))
        stats.kv_bytes.append(kv_cache_bytes(cache))
        logits = out.logits[:, -1, :]
        pos += 1
    else:
        stats.stop_reason = "max_new_tokens"
    return generated, stats
