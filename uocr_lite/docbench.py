"""Turn OmniDocBench ground truth and DocTags predictions into the same element list.

Element = dict(cls, text, latex, html, box, order)
  cls in {title, text, table, formula, figure}; headers/footers/page numbers/
  page footnotes/masks are ignored on both sides (as OmniDocBench end-to-end does).
"""
from __future__ import annotations

from PIL import Image

GT_IGNORE = {"header", "footer", "page_number", "page_footnote", "abandon", "list_group"}
GT_CLASS = {
    "title": "title",
    "text_block": "text", "figure_caption": "text", "table_caption": "text",
    "table_footnote": "text", "figure_footnote": "text", "equation_caption": "text",
    "reference": "text", "code_txt": "text", "code_txt_caption": "text",
    "equation_semantic": "text", "equation_explanation": "text",
    "table": "table", "equation_isolated": "formula", "figure": "figure",
}
PRED_CLASS = {
    "title": "title", "section_header": "title",
    "text": "text", "paragraph": "text", "list_item": "text", "caption": "text",
    "footnote": "text", "code": "text", "reference": "text",
    "table": "table", "formula": "formula", "picture": "figure", "chart": "figure",
}


def _poly_to_box(poly):
    xs, ys = poly[0::2], poly[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def gt_elements(page: dict) -> list[dict]:
    out = []
    for e in page["layout_dets"]:
        cat = e["category_type"]
        if e.get("ignore") or cat in GT_IGNORE or cat.endswith("_mask") or cat not in GT_CLASS:
            continue
        out.append({"cls": GT_CLASS[cat], "text": e.get("text", ""), "latex": e.get("latex", ""),
                    "html": e.get("html", ""), "box": _poly_to_box(e["poly"]),
                    "order": e.get("order") if e.get("order") is not None else 10**6})
    out.sort(key=lambda x: x["order"])
    return out


def gt_ignored_boxes(page: dict) -> list[list[float]]:
    """Regions OmniDocBench does not score (headers, footers, page numbers, masks...)."""
    return [_poly_to_box(e["poly"]) for e in page["layout_dets"]
            if e.get("ignore") or e["category_type"] in GT_IGNORE
            or e["category_type"].endswith("_mask")]


def drop_ignored(pred: list[dict], ignored: list[list[float]], thr: float = 0.5) -> list[dict]:
    """Remove predictions that mostly lie inside an ignored GT region, so reading a
    header the benchmark ignores is not penalized (OmniDocBench does the same)."""
    def inside(b, r):
        ix = max(0.0, min(b[2], r[2]) - max(b[0], r[0]))
        iy = max(0.0, min(b[3], r[3]) - max(b[1], r[1]))
        area = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
        return ix * iy / area >= thr
    return [e for e in pred if not (e["box"] and any(inside(e["box"], r) for r in ignored))]


def pred_elements(doctags: str, image: Image.Image) -> list[dict]:
    from docling_core.types.doc import DoclingDocument
    from docling_core.types.doc.document import DocTagsDocument

    if not doctags.startswith("<doctag>"):
        doctags = f"<doctag>{doctags}</doctag>"
    try:
        dt = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
        doc = DoclingDocument.load_from_doctags(dt, document_name="pred")
    except Exception as ex:  # malformed output (e.g. truncated) counts as empty page
        print("  [warn] doctags parse failed:", str(ex)[:120])
        return []
    out = []
    for order, (item, _lvl) in enumerate(doc.iterate_items()):
        label = str(getattr(item.label, "value", item.label))
        cls = PRED_CLASS.get(label)
        if cls is None:
            continue
        prov = item.prov[0] if getattr(item, "prov", None) else None
        box = [prov.bbox.l, prov.bbox.t, prov.bbox.r, prov.bbox.b] if prov else None
        el = {"cls": cls, "text": getattr(item, "text", "") or "", "latex": "", "html": "",
              "box": box, "order": order}
        if cls == "formula":
            el["latex"] = el["text"]
        elif cls == "table":
            try:
                el["html"] = item.export_to_html(doc=doc)
            except Exception:
                el["html"] = ""
        out.append(el)
    return out
