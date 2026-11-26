"""Unit tests for the evaluation metrics (no model needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uocr_lite.metrics import distinct_n, edit_distance, layout_counts, norm_latex, norm_text, teds  # noqa: E402

G = "<table><tr><td>a</td><td>b</td></tr><tr><td>1</td><td>2</td></tr></table>"


def test_teds():
    assert teds(G, G) == 1.0
    th = ("<table><thead><tr><th>a</th><th>b</th></tr></thead>"
          "<tbody><tr><td>1</td><td>2</td></tr></tbody></table>")
    assert teds(th, G) == 1.0                      # th/thead are formatting only
    wrong = G.replace(">2<", ">9<")
    assert 0.8 < teds(wrong, G) < 1.0 and teds(wrong, G, structure_only=True) == 1.0
    assert teds("", G) == 0.0


def test_latex_granularity_invariant():
    block = r"\begin{aligned} V_s &= V_{s0} \\ V_{s0} &= 2 \end{aligned}"
    lines = norm_latex(r"V_s=V_{s0}") + norm_latex(r"V_{s0}=2 \tag{3}")
    assert norm_latex(block) == lines


def test_latex_matrix_forms_close():
    a = norm_latex(r"\left[\begin{array}{cc} 1 & 2 \\ 3 & 4\end{array}\right]")
    b = norm_latex(r"\begin{bmatrix}1&2\\3&4\end{bmatrix}")
    assert edit_distance(a, b) <= 0.34


def test_text_inline_math():
    assert norm_text(r"Thus $x_1 = \frac{4}{3}x_3$, and") == norm_text("Thus x 1 = 4 3 x 3, and")


def test_distinct_and_layout():
    assert distinct_n([1, 2, 3] * 20, 5) < 0.1 and distinct_n(list(range(60)), 5) == 1.0
    c = layout_counts([("text", [0, 0, 10, 10]), ("table", [0, 20, 10, 30])],
                      [("text", [1, 1, 10, 10]), ("text", [50, 50, 60, 60])])
    assert c["text"] == [1, 2, 1] and c["table"] == [0, 0, 1]
