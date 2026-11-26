"""Document-level OCR metrics (OmniDocBench-style, CPU-friendly).

  text_ed       normalized edit distance of page text            (lower better)
  table TEDS    tree-edit-distance similarity of table HTML      (higher better)
  table TEDS-S  same, structure only (cell text ignored)         (higher better)
  formula_ed    normalized edit distance of display-formula LaTeX (lower better)
                (OmniDocBench uses CDM, which needs a LaTeX renderer; we use edit
                 distance on normalized LaTeX, the metric it reported before CDM)
  order_ed      reading-order edit distance over matched blocks  (lower better)
  layout F1     box detection F1 @ IoU 0.5 per coarse class      (higher better)
  distinct-n    unique n-grams / all n-grams of output tokens    (higher better;
                catches repetition loops, as in the paper's Table 3)
"""
from __future__ import annotations

import re
import unicodedata

from apted import APTED, Config
from apted.helpers import Tree
from lxml import html as lxml_html
from rapidfuzz.distance import Levenshtein


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #
INLINE_MATH = re.compile(r"\$\$(.*?)\$\$|\$(.*?)\$|\\\((.*?)\\\)", re.S)
LATEX_CMD = re.compile(r"\\[a-zA-Z]+")


def _math_to_plain(m: re.Match) -> str:
    body = next(g for g in m.groups() if g is not None)
    return LATEX_CMD.sub("", body)


def norm_text(s: str) -> str:
    """NFKC; inline math -> its plain symbols (GT writes `$x_1$`, the model often
    writes `x 1`), then drop LaTeX commands, markup and all whitespace. Applied
    identically to GT and prediction."""
    s = unicodedata.normalize("NFKC", s or "")
    s = INLINE_MATH.sub(_math_to_plain, s)
    s = LATEX_CMD.sub("", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"[#*_`|{}^$\\]", "", s)
    return re.sub(r"\s+", "", s)


def edit_distance(a: str, b: str) -> float:
    """Normalized Levenshtein in [0,1]; two empty strings -> 0."""
    if not a and not b:
        return 0.0
    return Levenshtein.normalized_distance(a, b)


# --------------------------------------------------------------------------- #
# formulas
# --------------------------------------------------------------------------- #
ALIGN_ENVS = (r"aligned|align\*?|split|gathered|gather\*?|eqnarray\*?|equation\*?|array|"
              r"[pbBvV]?matrix|cases")


def norm_latex(s: str) -> str:
    """Normalize LaTeX so layout-only differences don't count as errors:
    equation numbers (\\tag), alignment environments, `&` and `\\\\` line breaks are
    removed, so one `aligned` block and the same lines as separate formulas compare equal."""
    s = (s or "").strip()
    s = re.sub(r"^\$\$|\$\$$|^\\\[|\\\]$|^\$|\$$", "", s.strip())
    s = re.sub(r"\\tag\*?\{[^}]*\}", "", s)
    s = re.sub(r"\\(begin|end)\{(" + ALIGN_ENVS + r")\}(\{[^}]*\})?", "", s)
    s = s.replace("\\\\", "").replace("&", "")
    s = re.sub(r"\\(left|right|big|Big|bigg|Bigg)(?![a-zA-Z])", "", s)
    s = re.sub(r"\\(displaystyle|textstyle|mathrm|operatorname)(?![a-zA-Z])", "", s)
    s = s.replace("\\,", "").replace("\\;", "").replace("\\!", "").replace("\\quad", "")
    s = re.sub(r"\s+", "", s)
    s = s.replace("{", "").replace("}", "")   # brace placement is not semantic here
    return s


# --------------------------------------------------------------------------- #
# matching helper
# --------------------------------------------------------------------------- #
def greedy_match(gt: list, pred: list, score) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching by highest score. Returns (gt_idx, pred_idx, score)."""
    pairs = sorted(((score(g, p), i, j) for i, g in enumerate(gt) for j, p in enumerate(pred)),
                   reverse=True)
    used_g, used_p, out = set(), set(), []
    for s, i, j in pairs:
        if i not in used_g and j not in used_p:
            used_g.add(i); used_p.add(j); out.append((i, j, s))
    return out


# --------------------------------------------------------------------------- #
# tables: TEDS (Zhong et al., 2020, PubTabNet)
# --------------------------------------------------------------------------- #
class _TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag, self.colspan, self.rowspan, self.content = tag, colspan, rowspan, content
        self.children = list(children)

    def bracket(self):
        return "{%s%s}" % (self.tag, "".join(c.bracket() for c in self.children))


class _TedsConfig(Config):
    def rename(self, n1, n2):
        if n1.tag != n2.tag or n1.colspan != n2.colspan or n1.rowspan != n2.rowspan:
            return 1.0
        if n1.tag == "td" and (n1.content or n2.content):
            return Levenshtein.normalized_distance(n1.content or "", n2.content or "")
        return 0.0


def _cell_text(td) -> str:
    return norm_text("".join(td.itertext()))


def _to_tree(node, structure_only: bool) -> _TableTree:
    if node.tag == "td":
        content = "" if structure_only else _cell_text(node)
        return _TableTree("td", int(node.attrib.get("colspan", 1)),
                          int(node.attrib.get("rowspan", 1)), content)
    t = _TableTree(node.tag)
    for child in node:
        if child.tag in ("tr", "td"):
            t.children.append(_to_tree(child, structure_only))
    return t


def _parse_table(html_str: str):
    """Parse HTML, flatten thead/tbody, th->td. Returns the <table> element or None."""
    if not html_str or "<t" not in html_str:
        return None
    s = re.sub(r"</?(thead|tbody|tfoot)[^>]*>", "", html_str)
    s = re.sub(r"<th(\s|>)", r"<td\1", s).replace("</th>", "</td>")
    try:
        root = lxml_html.fromstring(f"<div>{s}</div>")
    except Exception:
        return None
    tables = root.xpath(".//table")
    return tables[0] if tables else None


def _count(t: _TableTree) -> int:
    return 1 + sum(_count(c) for c in t.children)


def teds(pred_html: str, gt_html: str, structure_only: bool = False) -> float:
    gt = _parse_table(gt_html)
    if gt is None:
        return float("nan")
    pred = _parse_table(pred_html)
    if pred is None:
        return 0.0
    tg, tp = _to_tree(gt, structure_only), _to_tree(pred, structure_only)
    n = max(_count(tg), _count(tp))
    dist = APTED(tp, tg, _TedsConfig()).compute_edit_distance()
    return max(0.0, 1.0 - dist / n)


# --------------------------------------------------------------------------- #
# layout: box F1 @ IoU
# --------------------------------------------------------------------------- #
def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def layout_counts(gt_boxes, pred_boxes, thr=0.5) -> dict:
    """gt/pred: lists of (cls, [x0,y0,x1,y1]). Returns {cls: [tp, n_pred, n_gt]}."""
    out = {}
    for cls in {c for c, _ in gt_boxes} | {c for c, _ in pred_boxes}:
        g = [b for c, b in gt_boxes if c == cls]
        p = [b for c, b in pred_boxes if c == cls]
        tp = sum(1 for _, _, s in greedy_match(g, p, iou) if s >= thr)
        out[cls] = [tp, len(p), len(g)]
    return out


# --------------------------------------------------------------------------- #
# repetition
# --------------------------------------------------------------------------- #
def distinct_n(tokens: list[int], n: int) -> float:
    if len(tokens) < n:
        return 1.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)
