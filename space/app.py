"""DocuMind — grounded document intelligence, with the retrieval visible.

The metaphor is a mind, not a filing cabinet. A document is read once and
becomes a set of **memory nodes**. A question propagates as a **signal**; the
nodes it reaches **fire**, and those are the citations. When no pathway
activates strongly enough, the mind says so rather than confabulating — which
is the behaviour this project is actually built around.

Every number on screen is measured. Where something is not measured it shows
an em-dash, and where the backend is a stand-in the page says so. There is no
confidence percentage anywhere, deliberately.

This file is presentation only. All retrieval logic lives in `app.*` and is
reached through `pipeline.py`.
"""

from __future__ import annotations

import os
from pathlib import Path

# This Space has decided to generate via Groq. Settings' own default is "auto",
# which deliberately never selects the Groq backend — a key present in the
# environment must not silently change which backend answers. Declaring it here,
# in the deployment's own entry point, keeps that choice explicit and visible
# rather than implicit. setdefault so an operator can still override it.
os.environ.setdefault("MODEL_BACKEND", "groq")

import gradio as gr  # noqa: E402

import pipeline  # noqa: E402

# ZeroGPU requires at least one decorated entry point to schedule a Space.
# Nothing here actually needs a GPU: generation is an API call and both local
# models (MiniLM embedder ~90 MB, cross-encoder ~90 MB) run acceptably on CPU.
# The decorated function is therefore a no-op that is never called on the hot
# path, so no visitor GPU quota is ever consumed.
try:
    import spaces

    @spaces.GPU(duration=1)
    def _zerogpu_probe() -> str:  # pragma: no cover - only meaningful on Spaces
        return "ok"

except ImportError:  # running locally
    spaces = None


DEMO_DIR = Path(__file__).parent / "corpus"
EM_DASH = "—"

CSS = """
:root {
  --dm-bg: #070b12;
  --dm-panel: rgba(12, 20, 32, 0.86);
  --dm-line: rgba(126, 178, 214, 0.16);
  --dm-dim: #4d6b80;
  --dm-text: #9ed6ec;
  --dm-live: #eafaff;
  --dm-signal: #6ee7d7;
  --dm-quiet: #3a6271;
}
.gradio-container { background: var(--dm-bg) !important; max-width: 1180px !important; }
#dm-head { border-bottom: 1px solid var(--dm-line); padding: 18px 4px 14px; margin-bottom: 8px; }
#dm-head h1 {
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  letter-spacing: 0.34em; font-size: 1.45rem; margin: 0; color: #fff; font-weight: 500;
}
#dm-head p { color: var(--dm-dim); font-size: 0.8rem; margin: 6px 0 0; letter-spacing: 0.04em; }
.dm-note {
  font-family: ui-monospace, monospace; font-size: 0.74rem; color: var(--dm-dim);
  letter-spacing: 0.06em; text-transform: uppercase;
}
.dm-card {
  background: var(--dm-panel); border: 1px solid var(--dm-line);
  border-radius: 3px; padding: 14px 16px;
}
.dm-kv { display: flex; flex-wrap: wrap; gap: 10px 26px; margin-top: 4px; }
.dm-kv div { font-family: ui-monospace, monospace; font-size: 0.74rem; }
.dm-kv span.k { color: var(--dm-quiet); letter-spacing: 0.08em; text-transform: uppercase; }
.dm-kv span.v { color: var(--dm-live); margin-left: 8px; }
.dm-warn {
  border-left: 2px solid #ffb454; padding: 9px 13px; margin: 10px 0;
  background: rgba(255, 180, 84, 0.07); color: #ffd79a;
  font-family: ui-monospace, monospace; font-size: 0.76rem;
}
.dm-abstain {
  border-left: 2px solid var(--dm-quiet); padding: 12px 15px;
  background: rgba(58, 98, 113, 0.12); color: #b9d4e0;
}
"""


def _nodes_svg(citations, total_chunks: int, page_count: int) -> str:
    """Activation map: every memory node in the index, those that fired lit.

    Brightness is the node's measured relevance score. Nothing here is
    decorative — an unlit field means nothing activated, which is exactly
    what an abstention looks like.
    """
    if not total_chunks:
        return ""

    fired = {c.chunk_id: c.relevance_score for c in citations}
    width, per_row = 940, 46
    radius, gap = 4, 20
    rows = (total_chunks + per_row - 1) // per_row
    height = max(rows * gap + 24, 60)

    dots = []
    for i in range(total_chunks):
        cx = 16 + (i % per_row) * gap
        cy = 18 + (i // per_row) * gap
        dots.append(
            f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="#12202e" stroke="#1d3446" stroke-width="1"/>'
        )

    # Fired nodes are drawn as a labelled row rather than pinned into the grid
    # above: a chunk id alone does not give its index position, and guessing a
    # coordinate would put a made-up location on screen.
    legend = []
    if fired:
        for i, (_cid, score) in enumerate(sorted(fired.items(), key=lambda kv: -kv[1])):
            alpha = 0.35 + 0.65 * min(max(score, 0.0), 1.0)
            legend.append(
                f'<circle cx="{16 + i * 26}" cy="{height + 16}" r="6" fill="#6ee7d7" fill-opacity="{alpha:.2f}"/>'
                f'<text x="{16 + i * 26}" y="{height + 34}" fill="#4d6b80" font-size="8" '
                f'font-family="monospace" text-anchor="middle">{score:.2f}</text>'
            )

    legend_label = (
        f'<text x="16" y="{height + 2}" fill="#3a6271" font-size="9" font-family="monospace" '
        f'letter-spacing="1.5">NODES FIRED — RELEVANCE</text>'
        if fired
        else f'<text x="16" y="{height + 2}" fill="#3a6271" font-size="9" font-family="monospace" '
        f'letter-spacing="1.5">NO PATHWAY ACTIVATED</text>'
    )

    return (
        f'<div class="dm-card"><div class="dm-note" style="margin-bottom:8px">'
        f'MEMORY NODES · {total_chunks} indexed · {page_count} pages</div>'
        f'<svg viewBox="0 0 {width} {height + 44}" width="100%" style="display:block">'
        f'{"".join(dots)}{legend_label}{"".join(legend)}</svg></div>'
    )


def _ms(value: float | None) -> str:
    """A stage that did not run renders as an em-dash, never as 0 ms."""
    return EM_DASH if value is None else f"{value:.0f} ms"


def _status_bar() -> str:
    info = pipeline.device_info()
    rows = "".join(
        f'<div><span class="k">{k}</span><span class="v">{v}</span></div>' for k, v in info.items()
    )
    warn = ""
    if pipeline.is_mock_embeddings():
        warn = (
            '<div class="dm-warn">STAND-IN EMBEDDINGS ACTIVE — retrieval scores on this '
            "page are not meaningful quantities and abstention will not trigger reliably. "
            "Set MODEL_BACKEND and GROQ_API_KEY for real behaviour.</div>"
        )
    return (
        f'<div class="dm-card"><div class="dm-note">BACKEND</div>'
        f'<div class="dm-kv"><div><span class="k">model</span>'
        f'<span class="v">{pipeline.backend_name()}</span></div>{rows}</div></div>{warn}'
    )


def do_ingest(file_obj, state):
    if file_obj is None:
        return state, gr.update(value="Choose a document first.", visible=True), "", ""
    try:
        result = pipeline.ingest(file_obj.name)
    except Exception as exc:
        return state, gr.update(value=f"Could not read that document: {exc}", visible=True), "", ""

    state = {"document_id": result.document_id, "title": result.title}
    summary = (
        f'<div class="dm-card"><div class="dm-note">READ INTO MEMORY</div><div class="dm-kv">'
        f'<div><span class="k">document</span><span class="v">{result.title}</span></div>'
        f'<div><span class="k">pages</span><span class="v">{result.pages}</span></div>'
        f'<div><span class="k">memory nodes</span><span class="v">{result.chunks}</span></div>'
        f'<div><span class="k">words</span><span class="v">{result.words:,}</span></div>'
        f"</div></div>"
    )
    return state, gr.update(visible=False), summary, _nodes_svg([], result.chunks, result.pages)


def do_ask(question, state):
    if not state or "document_id" not in state:
        return "Load a document first.", "", ""
    if not question or not question.strip():
        return "Ask something.", "", ""

    document_id = state["document_id"]
    try:
        result = pipeline.ask(document_id, question.strip())
    except Exception as exc:
        return f"That request failed: {exc}", "", ""

    if result.abstained:
        answer = (
            f'<div class="dm-card dm-abstain"><div class="dm-note" style="margin-bottom:6px">'
            f"NO PATHWAY ACTIVATED</div>{result.answer}<div class='dm-note' style='margin-top:8px'>"
            f"This is a designed response, not an error. The strongest signal reached "
            f"{result.top_score:.4f}, below the {pipeline._settings.min_relevance_score} floor.</div></div>"
        )
        citations_md = ""
    else:
        answer = f'<div class="dm-card">{result.answer}</div>'
        lines = []
        for i, c in enumerate(result.citations, start=1):
            page = f"page {c.page_number}" if c.page_number else EM_DASH
            lines.append(f"**[{i}]** {page} · relevance `{c.relevance_score:.4f}`\n\n> {c.snippet}\n")
        citations_md = "\n".join(lines)

    telemetry = (
        f'<div class="dm-card"><div class="dm-note">TELEMETRY</div><div class="dm-kv">'
        f'<div><span class="k">grounding</span><span class="v">{result.grounding.value}</span></div>'
        f'<div><span class="k">top signal</span><span class="v">{result.top_score:.4f}</span></div>'
        f'<div><span class="k">nodes fired</span><span class="v">{len(result.citations)}</span></div>'
        f'<div><span class="k">embed</span><span class="v">{_ms(result.embed_ms)}</span></div>'
        f'<div><span class="k">search</span><span class="v">{_ms(result.search_ms)}</span></div>'
        f'<div><span class="k">rerank</span><span class="v">{_ms(result.rerank_ms)}</span></div>'
        f'<div><span class="k">generate</span><span class="v">{_ms(result.generate_ms)}</span></div>'
        f'<div><span class="k">total</span><span class="v">{result.total_ms:.0f} ms</span></div>'
        f'<div><span class="k">model</span><span class="v">{result.model_used}</span></div>'
        f"</div></div>"
    )
    nodes = _nodes_svg(
        result.citations, pipeline.chunk_count(document_id), pipeline.page_count(document_id)
    )
    return answer + telemetry + nodes, citations_md, ""


def do_retrieve(question, state):
    if not state or "document_id" not in state:
        return [], "Load a document first."
    if not question or not question.strip():
        return [], "Ask something."
    rows, elapsed = pipeline.candidate_rows(state["document_id"], question.strip())
    note = (
        f'<div class="dm-note">{len(rows)} candidates retrieved in {elapsed:.0f} ms · '
        f'"survives" shows which reached the generator after reranking</div>'
    )
    return rows, note


with gr.Blocks(css=CSS, title="DocuMind", theme=gr.themes.Base()) as demo:
    state = gr.State({})

    gr.HTML(
        '<div id="dm-head"><h1>DOCUMIND</h1>'
        "<p>Grounded document intelligence — a document becomes memory, a question becomes a signal, "
        "and the nodes it reaches are your citations. It declines when nothing activates.</p></div>"
    )
    status = gr.HTML(_status_bar())

    with gr.Row():
        upload = gr.File(
            label="Document (PDF, TXT, PNG, JPG)",
            file_types=[".pdf", ".txt", ".png", ".jpg", ".jpeg"],
            scale=3,
        )
        ingest_btn = gr.Button("Read into memory", variant="primary", scale=1)
    ingest_error = gr.Markdown(visible=False)
    ingest_summary = gr.HTML()
    ingest_nodes = gr.HTML()

    with gr.Tabs():
        with gr.Tab("Recall"):
            gr.Markdown(
                "Ask a question. The answer is generated **only** from retrieved passages — "
                "if they don't support an answer, it declines instead of guessing."
            )
            question = gr.Textbox(
                label="Question", placeholder="What does this document say about…", lines=2
            )
            ask_btn = gr.Button("Ask", variant="primary")
            answer_out = gr.HTML()
            citations_out = gr.Markdown(label="Citations")

        with gr.Tab("Activation"):
            gr.Markdown(
                "The retrieval pool with the lid off — every candidate the index returned, its "
                "score, and whether it survived reranking. Retrieval only: no generation, so this "
                "is the cheapest way to see the mechanics."
            )
            retr_question = gr.Textbox(label="Question", lines=2)
            retr_btn = gr.Button("Retrieve", variant="primary")
            retr_note = gr.HTML()
            retr_table = gr.Dataframe(
                headers=["rank", "score", "survives", "page", "passage"],
                datatype=["number", "str", "str", "str", "str"],
                wrap=True,
            )

        with gr.Tab("Evidence"):
            gr.HTML(
                '<div class="dm-card">'
                '<div class="dm-note" style="margin-bottom:10px">EVALUATION — NOT YET MEASURED</div>'
                "<p style='color:#9ed6ec;font-size:0.86rem;line-height:1.6'>"
                "This tab will publish a before/after table generated from committed result "
                "files: recall@4, nDCG@10, false-answer rate and p50 latency across dense-only, "
                "+BM25 hybrid, +cross-encoder rerank and calibrated abstention.</p>"
                "<p style='color:#4d6b80;font-size:0.8rem;line-height:1.6'>"
                "It is empty because the evaluation harness has not been run yet, and this "
                "project does not print a number it has not measured. Every cell below stays an "
                "em-dash until a committed, SHA-stamped result file fills it.</p>"
                "<table style='width:100%;margin-top:12px;font-family:ui-monospace,monospace;"
                "font-size:0.78rem;color:#9ed6ec;border-collapse:collapse'>"
                "<tr style='color:#3a6271;text-align:left'>"
                "<th style='padding:6px 4px'>CONFIG</th><th>RECALL@4</th><th>NDCG@10</th>"
                "<th>FALSE-ANSWER</th><th>P50</th></tr>"
                + "".join(
                    f"<tr style='border-top:1px solid rgba(126,178,214,0.12)'>"
                    f"<td style='padding:6px 4px'>{c}</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>"
                    for c in [
                        "Dense only (baseline)",
                        "+ BM25 hybrid",
                        "+ cross-encoder rerank",
                        "+ calibrated abstention",
                    ]
                )
                + "</table></div>"
            )

    gr.HTML(
        '<div class="dm-note" style="margin-top:18px;padding-top:12px;'
        'border-top:1px solid rgba(126,178,214,0.16)">'
        "Uploads are session-scoped and are not persisted. · The full visualized experience — "
        "a Schwarzschild raytracer rendering the same backend — runs in the project's Colab "
        "notebook, because a WebGL frontend cannot be hosted on this tier. Neither surface is a "
        "cut-down version of the other.</div>"
    )

    ingest_btn.click(
        do_ingest, [upload, state], [state, ingest_error, ingest_summary, ingest_nodes]
    )
    ask_btn.click(do_ask, [question, state], [answer_out, citations_out, ingest_error])
    retr_btn.click(do_retrieve, [retr_question, state], [retr_table, retr_note])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
