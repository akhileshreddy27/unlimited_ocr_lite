# Unlimited-OCR-Lite

> **Results:** [docs/RESULTS.md](docs/RESULTS.md)

Long-horizon document parsing on a CPU laptop. This is a small-scale
re-implementation of the core idea of *Unlimited OCR Works* (Baidu, 2026):
**Reference Sliding Window Attention (R-SWA)**.

| Resume point | Where it lives |
|---|---|
| VLM-based OCR pipeline for multi-page parsing | `uocr_lite/model.py`, `uocr_lite/pipeline.py`, `scripts/parse_pdf.py` |
| R-SWA to restrict decoder attention and bound KV-cache growth | `uocr_lite/rswa.py`, `tests/test_rswa.py` |
| Long-context inference analysis (complexity, KV memory, latency, throughput) | `scripts/profile_decode.py` |
| Document-level evaluation (text / tables / formulas / layout) | `uocr_lite/metrics.py`, `uocr_lite/docbench.py`, `scripts/evaluate.py` |

## Hardware

Intel Core Ultra 5 225H, 15 GB RAM, no NVIDIA GPU. Everything runs on CPU in fp32.

## Model

`ibm-granite/granite-docling-258M`: an Idefics3-style VLM (SigLIP2 vision
encoder + a small Granite decoder) that outputs **DocTags**: text, section headers,
tables (OTSL), formulas (LaTeX), code, and layout boxes, with `<page_break>`
between pages. We do **not** retrain it from scratch. R-SWA is applied to its
decoder at inference time, with optional light LoRA adaptation later.

## R-SWA in one paragraph

During decoding, each new token attends to **all reference tokens** (the page
images plus the prompt, length `L_m`) and **only the last `n` generated tokens**.
The KV cache is therefore bounded at `L_m + n`, instead of `L_m + T` for full
attention. Per-token attention cost stops growing with output length `T`.
Our implementation evicts the middle of the KV cache after each step. Keys are
stored after RoPE, so surviving keys keep their absolute positions.
`tests/test_rswa.py` checks that this cache eviction gives the same logits as a
teacher-forced pass with the explicit R-SWA attention mask.

## Usage

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt

# parse a PDF (page-wise, full attention)
.\.venv\Scripts\python scripts\parse_pdf.py data\samples\unlimited_ocr_paper.pdf --pages 1-3

# same with R-SWA, window n = 256
.\.venv\Scripts\python scripts\parse_pdf.py data\samples\unlimited_ocr_paper.pdf --pages 1-3 --window 256

# long-horizon: 4 pages per request in one continuous decode
.\.venv\Scripts\python scripts\parse_pdf.py data\samples\unlimited_ocr_paper.pdf --pages 1-4 --mode long --page-window 4 --window 256

# correctness tests
.\.venv\Scripts\python -m pytest tests -q

# profiling: latency / KV memory vs decode step
.\.venv\Scripts\python scripts\profile_decode.py --tokens 3000 --windows full 128 512
```

Outputs go to `outputs/` (DocTags, Markdown, per-page stats JSON, plots).

Everything after the eval sweep can run unattended (keeps the laptop awake, logs each stage to
`outputs/overnight/`):

```powershell
.\.venv\Scripts\python scripts\overnight.py --wait-pid <sweep PID>
```

## Evaluation (point 4)

```powershell
# 1. 56-page stratified English subset of OmniDocBench (8 pages x 7 document types)
.\.venv\Scripts\python scripts\prepare_omnidocbench.py --per-type 8 --seed 0
# 2. inference for full attention and R-SWA n = 512/256/128 (resumable; ~1-2 h on CPU)
.\.venv\Scripts\python scripts\run_eval_inference.py --settings full 512 256 128
# 3. score -> outputs/eval/results.md
.\.venv\Scripts\python scripts\evaluate.py --settings full 512 256 128
```

Methodology choices, and why:

| Choice | Reason |
|---|---|
| English pages only | granite-docling is English-first; Chinese pages would measure language coverage, not R-SWA |
| Stratified by document type, with table/formula pages sampled first | 56 pages is small; this guarantees ~18 table and ~17 formula pages |
| Headers, footers, page numbers ignored; predictions inside those regions dropped | same as OmniDocBench end-to-end evaluation |
| Inline math normalized to plain symbols on both sides | GT writes `$x_1$`, the model writes `x 1`; same content should not count as an error |
| Formula edit distance on normalized LaTeX instead of CDM | CDM needs a LaTeX renderer + image matching; ED is the earlier OmniDocBench metric |
| TEDS / TEDS-S with greedy table matching, unmatched GT table = 0 | standard PubTabNet TEDS (`apted`), `th`/`thead` normalized away |
| Layout F1 @ IoU 0.5 on 5 classes (title, text, table, formula, figure) | box-level check that the model finds the right regions |
| Distinct-20/35 on output tokens | catches repetition loops, as in the paper's Table 3 |
| R-SWA pages whose full-attention output is ≤ n+1 tokens reuse that output | exact: no KV entry is evicted before token n+1 (verified in `tests/test_rswa.py`) |

## Roadmap

1. [x] Pipeline: PDF → images → DocTags → Markdown (page-wise and long-horizon)
2. [x] R-SWA cache + greedy decoder with per-step instrumentation
3. [x] Correctness tests
4. [ ] Profiling sweep: n ∈ {64,128,256,512,full} × prefix of {1,2,4} pages
5. [x] Evaluation pipeline on an OmniDocBench subset: text edit distance, table TEDS,
       formula edit distance, reading order, layout F1, Distinct-n
6. [x] Synthetic multi-page documents (2/4/8 pages) for long-horizon evaluation
       (`scripts/multipage_eval.py run|score`): page-wise vs one-request full attention vs one-request R-SWA
7. [x] Ring-buffer KV cache (`RingRSWALayer`: preallocated, in-place writes, no torch.cat),
       token-identical to the simple cache (`tests/test_rswa.py::test_ring_buffer_matches_trim`)
8. [x] Repeated 7k-token latency runs: full vs trim vs ring
       (`profile_decode.py --tokens 8000 --settings full 512 ring512 ring128 --repeats 3 --threads 12`)
Out of scope (by design): retraining / LoRA adaptation. Training-free R-SWA loses accuracy on
long pages; the paper fixes this with continued training, and we report the gap instead.
