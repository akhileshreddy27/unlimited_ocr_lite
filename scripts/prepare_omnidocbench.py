"""Build a small, stratified, reproducible OmniDocBench subset for CPU evaluation.

* English pages only (granite-docling is an English-first model).
* Document types with >= 20 English pages (7 types); `per_type` pages each.
* Within each type, up to `min_table` table pages and `min_eq` formula pages are
  sampled first, so tables and display formulas get enough samples.
* Writes data/omnidocbench/subset.json (annotations) + downloads only those images.

  python scripts/prepare_omnidocbench.py --per-type 8 --seed 0
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "opendatalab/OmniDocBench"
ROOT = Path(__file__).resolve().parents[1] / "data" / "omnidocbench"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=8)
    ap.add_argument("--min-table", type=int, default=2)
    ap.add_argument("--min-eq", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ann = hf_hub_download(REPO, "OmniDocBench.json", repo_type="dataset", local_dir=ROOT)
    pages = json.load(open(ann, encoding="utf-8"))
    en = [p for p in pages if p["page_info"]["page_attribute"].get("language") == "english"]
    by_type = defaultdict(list)
    for p in en:
        by_type[p["page_info"]["page_attribute"]["data_source"]].append(p)

    rng = random.Random(args.seed)
    subset = []
    for src in sorted(by_type):
        group = sorted(by_type[src], key=lambda p: p["page_info"]["image_path"])
        if len(group) < 20:
            print(f"skip {src}: only {len(group)} English pages")
            continue
        has = lambda p, c: any(e["category_type"] == c for e in p["layout_dets"])  # noqa: E731
        chosen = []
        for cat, k in (("table", args.min_table), ("equation_isolated", args.min_eq)):
            pool = [p for p in group if has(p, cat) and p not in chosen]
            chosen += rng.sample(pool, min(k, len(pool)))
        rest = [p for p in group if p not in chosen]
        chosen += rng.sample(rest, args.per_type - len(chosen))
        subset += chosen
        print(f"{src:22s} {len(chosen)} pages "
              f"(tables {sum(has(p, 'table') for p in chosen)}, "
              f"formulas {sum(has(p, 'equation_isolated') for p in chosen)})")

    for i, p in enumerate(subset, 1):
        hf_hub_download(REPO, f"images/{p['page_info']['image_path']}", repo_type="dataset",
                        local_dir=ROOT)
        print(f"\rimages {i}/{len(subset)}", end="", flush=True)
    print()
    out = ROOT / "subset.json"
    out.write_text(json.dumps(subset, ensure_ascii=False), encoding="utf-8")
    print(f"saved {len(subset)} pages -> {out}")


if __name__ == "__main__":
    main()
