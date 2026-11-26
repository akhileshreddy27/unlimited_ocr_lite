"""Score predictions in outputs/eval/<setting>/preds.jsonl against OmniDocBench.

Writes outputs/eval/results.md (tables for the report) and results.json.

Per page:
  text_ed     page text (title+text classes, GT reading order vs predicted order),
              normalized edit distance
  table       each GT table matched to the predicted table with best TEDS;
              unmatched GT tables score 0
  formula_ed  page-level: all display formulas concatenated in reading order, LaTeX
              normalized (no \tag, alignment envs, & or \\), normalized edit distance
  order_ed    text blocks matched by text similarity (>= 0.5); edit distance between
              the GT order of matched blocks and the order they were predicted in
  layout      F1 @ IoU 0.5 per class, micro-averaged over pages
  distinct-20/35 on generated token ids
Overall* = mean(100*(1-text_ed), TEDS, 100*(1-formula_ed))  (formula ED instead of CDM)

  python scripts/evaluate.py --settings full 512 256 128
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uocr_lite.docbench import drop_ignored, gt_elements, gt_ignored_boxes, pred_elements  # noqa: E402
from uocr_lite.metrics import (distinct_n, edit_distance, greedy_match, layout_counts,  # noqa: E402
                               norm_latex, norm_text, teds)

DATA = ROOT / "data" / "omnidocbench"
TEXTY = ("title", "text")


def score_page(gt_page: dict, rec: dict) -> dict:
    img = Image.open(DATA / "images" / rec["image"]).convert("RGB")
    gt = gt_elements(gt_page)
    pr = drop_ignored(pred_elements(rec["doctags"], img), gt_ignored_boxes(gt_page))
    r = {"image": rec["image"], "type": gt_page["page_info"]["page_attribute"]["data_source"],
         "generated": rec["stats"]["generated"], "stop": rec["stats"]["stop"]}

    # text
    g_txt = [e for e in gt if e["cls"] in TEXTY]
    p_txt = [e for e in pr if e["cls"] in TEXTY]
    r["text_ed"] = edit_distance("".join(norm_text(e["text"]) for e in g_txt),
                                 "".join(norm_text(e["text"]) for e in p_txt))

    # reading order over matched text blocks
    g_n = [norm_text(e["text"]) for e in g_txt]
    p_n = [norm_text(e["text"]) for e in p_txt]
    m = [(i, j) for i, j, s in greedy_match(g_n, p_n, lambda a, b: 1 - edit_distance(a, b))
         if s >= 0.5 and g_n[i]]
    if len(m) >= 2:
        gt_rank = [i for i, _ in sorted(m, key=lambda x: x[1])]  # GT index in predicted order
        r["order_ed"] = edit_distance("".join(chr(0x4E00 + i) for i in gt_rank),
                                      "".join(chr(0x4E00 + i) for i in sorted(gt_rank)))
    else:
        r["order_ed"] = math.nan

    # tables
    g_tab = [e["html"] for e in gt if e["cls"] == "table" and e["html"]]
    p_tab = [e["html"] for e in pr if e["cls"] == "table"]
    if g_tab:
        tm = {i: j for i, j, _ in greedy_match(g_tab, p_tab, lambda g, p: teds(p, g, True))}
        r["teds"] = mean(teds(p_tab[tm[i]], g, False) if i in tm else 0.0 for i, g in enumerate(g_tab))
        r["teds_s"] = mean(teds(p_tab[tm[i]], g, True) if i in tm else 0.0 for i, g in enumerate(g_tab))
    else:
        r["teds"] = r["teds_s"] = math.nan

    # formulas: page-level (all display formulas concatenated in reading order).
    # Granularity-invariant: the model often emits a multi-line `aligned` block where
    # the GT has one formula per line; one-to-one matching would score those lines as
    # missed although they were transcribed. Truly missing formulas still cost, since
    # their content is absent from the predicted string.
    g_f = [norm_latex(e["latex"]) for e in gt if e["cls"] == "formula"]
    p_f = [norm_latex(e["latex"]) for e in pr if e["cls"] == "formula"]
    r["formula_ed"] = edit_distance("".join(g_f), "".join(p_f)) if g_f else math.nan
    r["n_formula_gt"], r["n_formula_pred"] = len(g_f), len(p_f)

    # layout
    r["layout"] = layout_counts([(e["cls"], e["box"]) for e in gt],
                                [(e["cls"], e["box"]) for e in pr if e["box"]])
    ids = rec.get("token_ids", [])
    r["distinct20"], r["distinct35"] = distinct_n(ids, 20), distinct_n(ids, 35)
    return r


def nanmean(xs):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return mean(xs) if xs else math.nan


def aggregate(rows: list[dict]) -> dict:
    a = {"pages": len(rows),
         "text_ed": nanmean(r["text_ed"] for r in rows),
         "teds": 100 * nanmean(r["teds"] for r in rows),
         "teds_s": 100 * nanmean(r["teds_s"] for r in rows),
         "formula_ed": nanmean(r["formula_ed"] for r in rows),
         "order_ed": nanmean(r["order_ed"] for r in rows),
         "distinct35": 100 * nanmean(r["distinct35"] for r in rows),
         "hit_max_tokens": sum(r["stop"] == "max_new_tokens" for r in rows),
         "avg_generated": mean(r["generated"] for r in rows)}
    parts = [100 * (1 - a["text_ed"]), a["teds"], 100 * (1 - a["formula_ed"])]
    a["overall"] = nanmean(parts)
    tot = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        for cls, (tp, npred, ngt) in r["layout"].items():
            t = tot[cls]; t[0] += tp; t[1] += npred; t[2] += ngt
    f1 = {}
    for cls, (tp, npred, ngt) in tot.items():
        p, rc = (tp / npred if npred else 0), (tp / ngt if ngt else 0)
        f1[cls] = 100 * (2 * p * rc / (p + rc) if p + rc else 0)
    a["layout_f1"] = f1
    alltp = sum(t[0] for t in tot.values()); allp = sum(t[1] for t in tot.values())
    allg = sum(t[2] for t in tot.values())
    p, rc = alltp / max(1, allp), alltp / max(1, allg)
    a["layout_f1_all"] = 100 * (2 * p * rc / (p + rc) if p + rc else 0)
    return a


def fmt(x, d=3):
    return "—" if isinstance(x, float) and math.isnan(x) else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=["full", "512", "256", "128"])
    ap.add_argument("--eval-dir", default="outputs/eval")
    args = ap.parse_args()

    gt = {p["page_info"]["image_path"]: p
          for p in json.load(open(DATA / "subset.json", encoding="utf-8"))}
    results, per_page = {}, {}
    for s in args.settings:
        path = Path(args.eval_dir) / s / "preds.jsonl"
        if not path.exists():
            print(f"skip {s}: no predictions"); continue
        recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = [score_page(gt[r["image"]], r) for r in recs]
        per_page[s] = rows
        results[s] = {"all": aggregate(rows)}
        for t in sorted({r["type"] for r in rows}):
            results[s][t] = aggregate([r for r in rows if r["type"] == t])
        print(s, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in results[s]["all"].items() if k != "layout_f1"}))

    # only compare settings on the pages they all share
    out = Path(args.eval_dir)
    (out / "results.json").write_text(json.dumps({"results": results, "per_page": per_page},
                                                 indent=1, default=str), encoding="utf-8")
    L = ["# Evaluation on OmniDocBench (English subset)\n",
         "Overall* = mean(100·(1−TextED), TEDS, 100·(1−FormulaED)); FormulaED replaces CDM.\n",
         "| Setting | Pages | Text ED↓ | TEDS↑ | TEDS-S↑ | Formula ED↓ | Order ED↓ | Layout F1↑ | Distinct-35↑ | Overall*↑ | Hit max tokens |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, res in results.items():
        a = res["all"]
        L.append(f"| {s} | {a['pages']} | {fmt(a['text_ed'])} | {fmt(a['teds'], 1)} | {fmt(a['teds_s'], 1)} | "
                 f"{fmt(a['formula_ed'])} | {fmt(a['order_ed'])} | {fmt(a['layout_f1_all'], 1)} | "
                 f"{fmt(a['distinct35'], 2)} | {fmt(a['overall'], 1)} | {a['hit_max_tokens']} |")
    first = next(iter(results), None)
    if first:
        L += ["", f"## Per document type ({first})\n",
              "| Type | Pages | Text ED↓ | TEDS↑ | Formula ED↓ | Order ED↓ |", "|---|---|---|---|---|---|"]
        for t, a in results[first].items():
            if t != "all":
                L.append(f"| {t} | {a['pages']} | {fmt(a['text_ed'])} | {fmt(a['teds'], 1)} | "
                         f"{fmt(a['formula_ed'])} | {fmt(a['order_ed'])} |")
        L += ["", f"## Layout F1 @ IoU 0.5 per class ({first})\n", "| Class | F1 |", "|---|---|"]
        for c, v in sorted(results[first]["all"]["layout_f1"].items()):
            L.append(f"| {c} | {v:.1f} |")
    (out / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"saved -> {out / 'results.md'}")


if __name__ == "__main__":
    main()
