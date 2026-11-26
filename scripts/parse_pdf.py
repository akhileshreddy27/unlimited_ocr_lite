"""Parse a PDF to Markdown.

Examples:
  python scripts/parse_pdf.py data/samples/unlimited_ocr_paper.pdf --pages 1-1
  python scripts/parse_pdf.py doc.pdf --pages 1-4 --mode long --page-window 4 --window 256
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uocr_lite.model import DocVLM, DEFAULT_MODEL  # noqa: E402
from uocr_lite.pipeline import Parser, doctags_to_markdown, pdf_to_images  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--pages", default="1-1", help="e.g. 1-5")
    ap.add_argument("--mode", choices=["page", "long"], default="page")
    ap.add_argument("--page-window", type=int, default=4, help="pages per request in long mode")
    ap.add_argument("--window", type=int, default=None, help="R-SWA output window n (omit = full attention)")
    ap.add_argument("--max-new-tokens", type=int, default=4096, help="per page")
    ap.add_argument("--dpi", type=int, default=144)
    ap.add_argument("--longest-edge", type=int, default=None, help="image processor longest edge")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    first, last = (int(x) for x in args.pages.split("-"))
    images = pdf_to_images(args.pdf, args.dpi, first, last)
    vlm = DocVLM(args.model, longest_edge=args.longest_edge)
    print(json.dumps(vlm.config_summary()))
    parser = Parser(vlm, window=args.window, max_new_tokens=args.max_new_tokens)
    if args.mode == "page":
        pages, stats = parser.parse_pagewise(images)
    else:
        pages, stats = parser.parse_long_horizon(images, args.page_window)

    tag = f"{Path(args.pdf).stem}_p{first}-{last}_{args.mode}_{'full' if args.window is None else f'rswa{args.window}'}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.doctags.txt").write_text("\n<page_break>\n".join(pages), encoding="utf-8")
    (out / f"{tag}.stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    md = doctags_to_markdown(pages, images)
    (out / f"{tag}.md").write_text(md, encoding="utf-8")
    print(f"saved -> {out / tag}.md")


if __name__ == "__main__":
    main()
