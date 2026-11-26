"""Point 3: long-context inference profiling: full attention vs R-SWA (trim vs ring cache).

Decodes a fixed number of tokens per setting (EOS ignored, repetition banned) so
every run is the same length, and records per-step latency + KV-cache size.
Each setting is repeated (--repeats) and reported as mean ± std.

Settings:  full | <n> (R-SWA, simple cat+evict cache) | ring<n> (R-SWA, in-place ring buffer)

Outputs (in --out):
  steps_<setting>.csv     per-step latency_ms, kv_tokens, kv_MB (first repeat)
  summary.json / summary.md
  latency_vs_step.png     ~ paper Fig. 3
  kv_vs_step.png          KV-cache memory growth
  complexity.png          analytical attention cost  O(T(L+T)) vs O(T(L+n))

Example (run with nothing else using the CPU):
  python scripts/profile_decode.py --tokens 8000 --settings full 512 ring512 ring128 --repeats 3
"""
import argparse
import csv
import json
import statistics as stats_mod
import sys
from pathlib import Path

import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uocr_lite.model import DocVLM, DEFAULT_MODEL  # noqa: E402
from uocr_lite.pipeline import pdf_to_images  # noqa: E402
from uocr_lite.rswa import NoRepeatNGram, greedy_decode  # noqa: E402


def smooth(xs, k=50):
    out, s = [], 0.0
    for i, x in enumerate(xs):
        s += x
        if i >= k:
            s -= xs[i - k]
        out.append(s / min(i + 1, k))
    return out


def parse_setting(s: str):
    if s == "full":
        return None, "trim"
    if s.startswith("ring"):
        return int(s[4:]), "ring"
    return int(s), "trim"


def mean_std(xs):
    return (round(stats_mod.mean(xs), 2), round(stats_mod.stdev(xs), 2) if len(xs) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="data/samples/unlimited_ocr_paper.pdf")
    ap.add_argument("--pages", type=int, default=1, help="pages in the reference prefix")
    ap.add_argument("--tokens", type=int, default=3000)
    ap.add_argument("--settings", "--windows", nargs="+", default=["full", "512", "ring512"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None, help="torch threads (pin for stable timing)")
    ap.add_argument("--longest-edge", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="outputs/profile")
    args = ap.parse_args()

    if sys.platform == "win32":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from keep_awake import hold
        hold()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vlm = DocVLM(args.model, threads=args.threads, longest_edge=args.longest_edge)
    cfg = {**vlm.config_summary(), "torch_threads": torch.get_num_threads()}
    print(json.dumps(cfg))
    images = pdf_to_images(args.pdf, 144, 1, args.pages)
    inputs = vlm.build_inputs(images)
    L = inputs["input_ids"].shape[1]
    print(f"prefix L_m = {L} tokens for {args.pages} page(s)")

    proc = psutil.Process()
    rows, curves = [], {}
    # warm-up so the first measured setting doesn't pay one-time costs
    greedy_decode(vlm.model, inputs, 32, eos_token_ids=[-1])
    for name in args.settings:
        window, impl = parse_setting(name)
        runs = []
        for rep in range(args.repeats):
            rss0 = proc.memory_info().rss
            ids, st = greedy_decode(vlm.model, inputs, args.tokens, eos_token_ids=[-1],
                                    window=window, no_repeat=NoRepeatNGram(35, 1024),
                                    cache_impl=impl)
            s = st.summary()
            s["rss_delta_MB"] = round((proc.memory_info().rss - rss0) / 2**20, 1)
            runs.append(s)
            print(json.dumps({"setting": name, "rep": rep, **s}), flush=True)
            if rep == 0:
                curves[name] = st
                with open(out / f"steps_{name}.csv", "w", newline="") as f:
                    wr = csv.writer(f)
                    wr.writerow(["step", "latency_ms", "kv_tokens", "kv_MB"])
                    for i, (ms, kt, kb) in enumerate(zip(st.step_ms, st.kv_tokens, st.kv_bytes)):
                        wr.writerow([i, round(ms, 3), kt, round(kb / 2**20, 3)])
        rows.append({"setting": name, "window": window, "cache": impl, "repeats": args.repeats,
                     "decode_tps": mean_std([r["decode_tps"] for r in runs]),
                     "first_100_ms": mean_std([r["first_100_ms"] for r in runs]),
                     "last_100_ms": mean_std([r["last_100_ms"] for r in runs]),
                     "peak_kv_MB": runs[0]["peak_kv_MB"],
                     "final_kv_tokens": runs[0]["final_kv_tokens"]})

    (out / "summary.json").write_text(json.dumps({"config": cfg, "prefix_len": L,
                                                  "tokens": args.tokens, "runs": rows}, indent=2))
    md = [f"Prefix L_m = {L}, {args.tokens} decoded tokens, {args.repeats} repeat(s), "
          f"{cfg['torch_threads']} threads\n",
          "| Setting | Cache | Decode tok/s | First-100 ms/tok | Last-100 ms/tok | Peak KV MB |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['setting']} | {r['cache'] if r['window'] else 'dynamic'} | "
                  f"{r['decode_tps'][0]} ± {r['decode_tps'][1]} | {r['first_100_ms'][0]} ± {r['first_100_ms'][1]} | "
                  f"{r['last_100_ms'][0]} ± {r['last_100_ms'][1]} | {r['peak_kv_MB']} |")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, st in curves.items():
        ax.plot(smooth(st.step_ms, 200), label=name)
    ax.set(xlabel="decode step", ylabel="per-step latency (ms, 200-step mean)",
           title=f"Decode latency vs step (L_m={L})")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "latency_vs_step.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, st in curves.items():
        ax.plot([b / 2**20 for b in st.kv_bytes], label=name)
    ax.set(xlabel="decode step", ylabel="KV cache (MB)", title="KV-cache memory vs step")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "kv_vs_step.png", dpi=150)

    # analytical attention cost per generated token: keys attended x 4*heads*head_dim*layers FLOPs
    T = args.tokens
    flops_per_key = 4 * cfg["q_heads"] * cfg["head_dim"] * cfg["layers"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ts = range(1, T + 1)
    ax.plot(ts, [flops_per_key * (L + t) / 1e6 for t in ts], label="full: L_m + t keys")
    for w in sorted({parse_setting(s)[0] for s in args.settings} - {None}):
        ax.plot(ts, [flops_per_key * (L + min(w, t)) / 1e6 for t in ts], label=f"R-SWA n={w}")
    ax.set(xlabel="decode step t", ylabel="attention MFLOPs per token",
           title="Attention cost per generated token")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "complexity.png", dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
