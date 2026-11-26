"""Run the VLM over the OmniDocBench subset for each attention setting.

Saves one JSON line per page to outputs/eval/<setting>/preds.jsonl:
  {image, doctags, token_ids, stats}
Resumable: pages already in preds.jsonl are skipped.

Shortcut (exact, not an approximation): R-SWA with window n only differs from
full attention once more than n tokens have been generated (first eviction).
tests/test_rswa.py::test_small_window_bounds_cache checks this. So if the
full-attention output has <= n+1 tokens, the R-SWA output is identical and we
copy it (marked "reused": true). Run "full" first.

  python scripts/run_eval_inference.py --settings full 512 256 128
"""
import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uocr_lite.model import DocVLM, PAGE_PROMPT  # noqa: E402
from uocr_lite.pipeline import clean_doctags  # noqa: E402
from uocr_lite.rswa import NoRepeatNGram, greedy_decode  # noqa: E402

DATA = ROOT / "data" / "omnidocbench"


def load_done(path: Path) -> dict:
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["image"]] = r
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=["full", "512", "256", "128"])
    ap.add_argument("--max-new-tokens", type=int, default=6000)
    ap.add_argument("--limit", type=int, default=0, help="only the first N pages (smoke test)")
    ap.add_argument("--out", default="outputs/eval")
    args = ap.parse_args()

    if sys.platform == "win32":  # long CPU job: don't let the laptop idle into Modern Standby
        sys.path.insert(0, str(ROOT / "scripts"))
        from keep_awake import hold
        hold()

    subset = json.load(open(DATA / "subset.json", encoding="utf-8"))
    if args.limit:
        subset = subset[:args.limit]
    vlm = DocVLM()
    no_repeat = NoRepeatNGram(35, 1024)
    full_path = Path(args.out) / "full" / "preds.jsonl"

    for setting in args.settings:
        window = None if setting == "full" else int(setting)
        path = Path(args.out) / setting / "preds.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        done = load_done(path)
        full = load_done(full_path) if window is not None else {}
        for i, page in enumerate(subset, 1):
            name = page["page_info"]["image_path"]
            if name in done:
                continue
            f = full.get(name)
            if f is not None and len(f["token_ids"]) <= window + 1:
                rec = {**f, "stats": {**f["stats"], "window": window, "reused": True}}
            else:
                img = Image.open(DATA / "images" / name).convert("RGB")
                t = time.perf_counter()
                ids, st = greedy_decode(vlm.model, vlm.build_inputs([img], PAGE_PROMPT),
                                        args.max_new_tokens, vlm.eos_token_ids,
                                        window=window, no_repeat=no_repeat)
                rec = {"image": name, "doctags": clean_doctags(vlm.decode_ids(ids)),
                       "token_ids": ids,
                       "stats": {**st.summary(), "wall_s": round(time.perf_counter() - t, 1),
                                 "reused": False}}
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            s = rec["stats"]
            print(f"[{setting}] {i}/{len(subset)} {name[:40]} gen={s['generated']} "
                  f"stop={s['stop']} reused={s['reused']} wall={s.get('wall_s')}", flush=True)


if __name__ == "__main__":
    main()
