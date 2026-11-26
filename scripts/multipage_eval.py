"""Point 1: multi-page / long-horizon parsing experiment.

Synthetic multi-page documents are built by concatenating OmniDocBench pages of the
same type (the paper built its multi-page data the same way). Documents of 2, 4 and
8 pages, for two document types (book, academic_literature).

Modes (all at longest_edge=1024 -> ~340 prefix tokens/page, like the paper's lower-
resolution "Base" mode for multi-page input):
  pagewise      each page parsed alone, outputs concatenated (baseline)
  long_full     all pages in ONE request, one continuous decode, full attention
  long_ring512  all pages in ONE request, R-SWA (ring cache, n=512)

The model was trained on single pages and an 8192-token context: 8-page documents
deliberately exceed 8192 positions (prefix + output), the limit the paper removes
by training at 32K. That is reported, not hidden.

  python scripts/multipage_eval.py run      # inference (resumable)
  python scripts/multipage_eval.py score    # -> outputs/multipage/results.md
"""
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uocr_lite.docbench import gt_elements, pred_elements  # noqa: E402
from uocr_lite.metrics import (distinct_n, edit_distance, greedy_match, norm_latex,  # noqa: E402
                               norm_text, teds)

DATA = ROOT / "data" / "omnidocbench"
OUT = ROOT / "outputs" / "multipage"
TYPES = ["book", "academic_literature"]
SIZES = [2, 4, 8]
MODES = {"pagewise": None, "long_full": None, "long_ring512": 512}
LONGEST_EDGE = 1024
PROMPT = "Convert this page to docling."
TOK_PER_PAGE = 2500
CTX = 8192


def docs():
    sub = json.load(open(DATA / "subset.json", encoding="utf-8"))
    out = []
    for t in TYPES:
        pages = [p for p in sub if p["page_info"]["page_attribute"]["data_source"] == t]
        for n in SIZES:
            out.append({"id": f"{t}_{n}p", "type": t, "pages": pages[:n]})
    return out


def load(path):
    if not path.exists():
        return {}
    return {r["key"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


def run():
    if sys.platform == "win32":
        sys.path.insert(0, str(ROOT / "scripts"))
        from keep_awake import hold
        hold()
    from uocr_lite.model import DocVLM
    from uocr_lite.rswa import NoRepeatNGram, greedy_decode

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "preds.jsonl"
    done = load(path)
    vlm = DocVLM(longest_edge=LONGEST_EDGE)
    nr = NoRepeatNGram(35, 1024)

    def decode(images, window, max_new):
        inputs = vlm.build_inputs(images, PROMPT)
        t = time.perf_counter()
        ids, st = greedy_decode(vlm.model, inputs, max_new, vlm.eos_token_ids, window=window,
                                no_repeat=nr, cache_impl="ring")
        s = {**st.summary(), "wall_s": round(time.perf_counter() - t, 1),
             "max_position": st.prefix_len + st.n_generated}
        return vlm.decode_ids(ids), ids, s

    def save(rec):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        s = rec["stats"]
        print(f"[{rec['key']}] gen={s['generated']} stop={s['stop']} maxpos={s['max_position']} "
              f"peakKV={s['peak_kv_MB']}MB tps={s['decode_tps']} wall={s['wall_s']}s", flush=True)

    # pagewise: each unique page once
    for d in docs():
        for p in d["pages"]:
            key = f"pagewise|{p['page_info']['image_path']}"
            if key in done:
                continue
            img = Image.open(DATA / "images" / p["page_info"]["image_path"]).convert("RGB")
            text, ids, s = decode([img], None, TOK_PER_PAGE)
            done[key] = rec = {"key": key, "doctags": text, "token_ids": ids, "stats": s}
            save(rec)
    # long-horizon: whole document in one request
    for mode in ("long_full", "long_ring512"):
        for d in docs():
            key = f"{mode}|{d['id']}"
            if key in done:
                continue
            imgs = [Image.open(DATA / "images" / p["page_info"]["image_path"]).convert("RGB")
                    for p in d["pages"]]
            text, ids, s = decode(imgs, MODES[mode], TOK_PER_PAGE * len(imgs))
            done[key] = rec = {"key": key, "doctags": text, "token_ids": ids, "stats": s}
            save(rec)


# --------------------------------------------------------------------------- #
# scoring (document level)
# --------------------------------------------------------------------------- #
def _pred_doc_elements(doctags, images):
    text = doctags.replace("<end_of_utterance>", "").replace("</doctag>", "").replace("<doctag>", "")
    pieces = text.split("<page_break>")
    els = []
    for i, piece in enumerate(pieces):
        els += pred_elements(f"<doctag>{piece.strip()}</doctag>", images[min(i, len(images) - 1)])
    return els, len(pieces) - 1


def _ignored_texts(pages):
    ign = []
    for p in pages:
        for e in p["layout_dets"]:
            if e["category_type"] in ("header", "footer", "page_number", "page_footnote", "abandon"):
                t = norm_text(e.get("text", ""))
                if t:
                    ign.append(t)
    return ign


def score_doc(pages, pred_els):
    """Doc-level metrics. Predictions matching GT-ignored text (headers, page numbers)
    are dropped by text, since long-horizon output can't be aligned to pages by box."""
    ign = _ignored_texts(pages)
    pred_els = [e for e in pred_els if not (e["cls"] in ("title", "text") and norm_text(e["text"])
                and any(edit_distance(norm_text(e["text"]), t) <= 0.3 for t in ign))]
    gt = [e for p in pages for e in gt_elements(p)]
    r = {"text_ed": edit_distance("".join(norm_text(e["text"]) for e in gt if e["cls"] in ("title", "text")),
                                  "".join(norm_text(e["text"]) for e in pred_els if e["cls"] in ("title", "text")))}
    g_tab = [e["html"] for e in gt if e["cls"] == "table" and e["html"]]
    p_tab = [e["html"] for e in pred_els if e["cls"] == "table"]
    if g_tab:
        tm = {i: j for i, j, _ in greedy_match(g_tab, p_tab, lambda g, p: teds(p, g, True))}
        r["teds"] = 100 * mean(teds(p_tab[tm[i]], g) if i in tm else 0.0 for i, g in enumerate(g_tab))
    else:
        r["teds"] = math.nan
    g_f = [norm_latex(e["latex"]) for e in gt if e["cls"] == "formula"]
    p_f = [norm_latex(e["latex"]) for e in pred_els if e["cls"] == "formula"]
    r["formula_ed"] = edit_distance("".join(g_f), "".join(p_f)) if g_f else math.nan
    return r


def score():
    recs = load(OUT / "preds.jsonl")
    rows = []
    for d in docs():
        imgs = [Image.open(DATA / "images" / p["page_info"]["image_path"]).convert("RGB") for p in d["pages"]]
        for mode in MODES:
            if mode == "pagewise":
                keys = [f"pagewise|{p['page_info']['image_path']}" for p in d["pages"]]
                if not all(k in recs for k in keys):
                    continue
                els, ids = [], []
                for k, img in zip(keys, imgs):
                    els += _pred_doc_elements(recs[k]["doctags"], [img])[0]
                    ids += recs[k]["token_ids"]
                st = [recs[k]["stats"] for k in keys]
                stats = {"generated": sum(s["generated"] for s in st),
                         "peak_kv_MB": max(s["peak_kv_MB"] for s in st),
                         "max_position": max(s["max_position"] for s in st),
                         "wall_s": round(sum(s["wall_s"] for s in st), 1), "stop": "eos"}
                breaks = len(d["pages"]) - 1
            else:
                rec = recs.get(f"{mode}|{d['id']}")
                if rec is None:
                    continue
                els, breaks = _pred_doc_elements(rec["doctags"], imgs)
                ids, stats = rec["token_ids"], rec["stats"]
            m = score_doc(d["pages"], els)
            rows.append({"doc": d["id"], "type": d["type"], "pages": len(d["pages"]), "mode": mode, **m,
                         "distinct35": 100 * distinct_n(ids, 35), "page_breaks": breaks,
                         "generated": stats["generated"], "max_position": stats["max_position"],
                         "peak_kv_MB": stats["peak_kv_MB"], "wall_s": stats["wall_s"],
                         "stop": stats["stop"]})

    def f(x, d=3):
        return "—" if isinstance(x, float) and math.isnan(x) else (f"{x:.{d}f}" if isinstance(x, float) else str(x))

    L = ["# Multi-page / long-horizon parsing (synthetic documents from OmniDocBench pages)\n",
         f"All modes at longest_edge={LONGEST_EDGE}. Model context = {CTX} positions; "
         "'Max pos' > 8192 means beyond the trained range.\n",
         "| Doc | Pages | Mode | Text ED↓ | TEDS↑ | Formula ED↓ | Distinct-35↑ | Page breaks (expected) | Generated | Max pos | Peak KV MB | Wall s | Stop |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['doc']} | {r['pages']} | {r['mode']} | {f(r['text_ed'])} | {f(r['teds'], 1)} | "
                 f"{f(r['formula_ed'])} | {r['distinct35']:.1f} | {r['page_breaks']} ({r['pages'] - 1}) | "
                 f"{r['generated']} | {r['max_position']} | {r['peak_kv_MB']} | {r['wall_s']} | {r['stop']} |")
    L += ["", "## Mean over documents, by size\n",
          "| Pages | Mode | Text ED↓ | Distinct-35↑ | Peak KV MB |", "|---|---|---|---|---|"]
    for n in SIZES:
        for mode in MODES:
            sel = [r for r in rows if r["pages"] == n and r["mode"] == mode]
            if sel:
                L.append(f"| {n} | {mode} | {mean(r['text_ed'] for r in sel):.3f} | "
                         f"{mean(r['distinct35'] for r in sel):.1f} | {max(r['peak_kv_MB'] for r in sel)} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    {"run": run, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
