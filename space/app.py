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

# Force CPU, and mean it. ZeroGPU runs torch in CUDA-emulation mode where
# torch.cuda.is_available() answers True, so auto-detection picks "cuda",
# sentence-transformers then triggers a real CUDA init outside any
# @spaces.GPU function, and ZeroGPU aborts the call with
# "Low-level CUDA init reached". Both local models are ~90 MB and run fine on
# CPU, and generation is an HTTP call, so nothing here wants a GPU anyway.
os.environ.setdefault("MODEL_DEVICE", "cpu")

# Translate via Groq rather than deep_translator's Google endpoint. That
# endpoint rate-limits by source IP and Spaces share egress IPs, so it returns
# "too many requests" here no matter how little this app sends — verified
# against the live Space, not assumed.
os.environ.setdefault("TRANSLATION_PROVIDER", "groq")

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

# Gradio 6 renders server-side by default. SSR re-executes the module per
# request, which makes hidden gr.State the least reliable place to keep
# document identity — so identity lives in a visible textbox instead (see
# below) and SSR is switched off, because this app is a stateful session
# rather than a static page.
os.environ.setdefault("GRADIO_SSR_MODE", "false")

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


def _evidence_tab() -> str:
    """The published evaluation table — empty until the harness has run.

    Every cell is an em-dash on purpose. The evaluation has not been executed
    yet, and this project does not print a number it has not measured. Filling
    these with plausible values would be the exact failure it exists to delete.
    """
    rows = "".join(
        f"<tr style='border-top:1px solid rgba(126,178,214,0.12)'>"
        f"<td style='padding:6px 4px'>{config}</td>"
        f"<td>{EM_DASH}</td><td>{EM_DASH}</td><td>{EM_DASH}</td><td>{EM_DASH}</td></tr>"
        for config in (
            "Dense only (baseline)",
            "+ BM25 hybrid",
            "+ cross-encoder rerank",
            "+ calibrated abstention",
        )
    )
    return (
        '<div class="dm-card">'
        '<div class="dm-note" style="margin-bottom:10px">EVALUATION — NOT YET MEASURED</div>'
        "<p style='color:#9ed6ec;font-size:0.86rem;line-height:1.6'>This tab will publish a "
        "before/after table generated from committed, SHA-stamped result files: recall@4, "
        "nDCG@10, false-answer rate and p50 latency across dense-only, +BM25 hybrid, "
        "+cross-encoder rerank and calibrated abstention.</p>"
        "<p style='color:#4d6b80;font-size:0.8rem;line-height:1.6'>It is empty because the "
        "evaluation harness has not been run yet. Every cell stays an em-dash until a "
        "committed result file fills it.</p>"
        "<table style='width:100%;margin-top:12px;font-family:ui-monospace,monospace;"
        "font-size:0.78rem;color:#9ed6ec;border-collapse:collapse'>"
        "<tr style='color:#3a6271;text-align:left'><th style='padding:6px 4px'>CONFIG</th>"
        "<th>RECALL@4</th><th>NDCG@10</th><th>FALSE-ANSWER</th><th>P50</th></tr>"
        f"{rows}</table></div>"
    )


def do_ingest(file_obj):
    """Read a document into memory and hand back its id.

    The id is returned into a *visible* textbox rather than a hidden
    gr.State. gr.State is per-browser-session, which made the Space's HTTP
    API unusable (every call got a fresh state and every answer was "Load a
    document first") and is fragile under Gradio's SSR mode. An explicit
    handle works identically in the browser and over the API, and it shows
    the user that a document really is loaded.
    """
    if file_obj is None:
        return "", "Choose a document first.", "", ""
    try:
        result = pipeline.ingest(file_obj.name)
    except Exception as exc:
        return "", f"Could not read that document: {exc}", "", ""

    summary = (
        f'<div class="dm-card"><div class="dm-note">READ INTO MEMORY</div><div class="dm-kv">'
        f'<div><span class="k">document</span><span class="v">{result.title}</span></div>'
        f'<div><span class="k">pages</span><span class="v">{result.pages}</span></div>'
        f'<div><span class="k">memory nodes</span><span class="v">{result.chunks}</span></div>'
        f'<div><span class="k">words</span><span class="v">{result.words:,}</span></div>'
        f"</div></div>"
    )
    return (
        result.document_id,
        "",
        summary,
        _nodes_svg([], result.chunks, result.pages),
    )


def _need_document(document_id) -> str | None:
    if not document_id or not str(document_id).strip():
        return '<div class="dm-card dm-abstain">Read a document into memory first.</div>'
    return None


def do_ask(question, document_id):
    guard = _need_document(document_id)
    if guard:
        return guard, ""
    if not question or not question.strip():
        return '<div class="dm-card dm-abstain">Ask a question first.</div>', ""

    try:
        result = pipeline.ask(document_id.strip(), question.strip())
    except Exception as exc:
        return f'<div class="dm-card dm-warn">That request failed: {exc}</div>', ""

    if result.abstained:
        answer = (
            f'<div class="dm-card dm-abstain"><div class="dm-note" style="margin-bottom:6px">'
            f"NO PATHWAY ACTIVATED</div>{result.answer}"
            f"<div class='dm-note' style='margin-top:8px'>A designed response, not an error. "
            f"The strongest signal reached {result.top_score:.4f}.</div></div>"
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
        f'<div><span class="k">total</span><span class="v">{_ms(result.total_ms)}</span></div>'
        f'<div><span class="k">model</span><span class="v">{result.model_used}</span></div>'
        f"</div></div>"
    )
    nodes = _nodes_svg(
        result.citations,
        pipeline.chunk_count(document_id.strip()),
        pipeline.page_count(document_id.strip()),
    )
    return answer + telemetry + nodes, citations_md


def do_retrieve(question, document_id):
    guard = _need_document(document_id)
    if guard:
        return [], guard
    if not question or not question.strip():
        return [], '<div class="dm-note">Ask a question first.</div>'
    try:
        rows, elapsed = pipeline.candidate_rows(document_id.strip(), question.strip())
    except Exception as exc:
        return [], f'<div class="dm-card dm-warn">Retrieval failed: {exc}</div>'
    note = (
        f'<div class="dm-note">{len(rows)} candidates in {elapsed:.0f} ms · '
        f'"survives" shows which reached the generator after reranking</div>'
    )
    return rows, note


def do_summarize(document_id):
    guard = _need_document(document_id)
    if guard:
        return guard, ""
    try:
        summary, strategy, elapsed = pipeline.summarize(document_id.strip())
    except Exception as exc:
        return f'<div class="dm-card dm-warn">Summarization failed: {exc}</div>', ""

    meta = (
        f'<div class="dm-card"><div class="dm-note">SUMMARY</div><div class="dm-kv">'
        f'<div><span class="k">strategy</span><span class="v">{strategy}</span></div>'
        f'<div><span class="k">elapsed</span><span class="v">{_ms(elapsed)}</span></div>'
        f'<div><span class="k">model</span><span class="v">{pipeline.backend_name()}</span></div>'
        f"</div></div>"
    )

    parts = [f"### Executive summary\n\n{summary.executive_summary}"]
    if summary.key_findings:
        parts.append("### Key findings\n\n" + "\n".join(f"- {k}" for k in summary.key_findings))
    if summary.important_numbers:
        parts.append(
            "### Important numbers\n\n" + "\n".join(f"- {n}" for n in summary.important_numbers)
        )
    # Absent sections render as an em-dash rather than being hidden, so the
    # reader can tell "the document has no methodology section" from "we did
    # not look".
    parts.append(f"### Methodology\n\n{summary.methodology or EM_DASH}")
    parts.append(f"### Limitations\n\n{summary.limitations or EM_DASH}")
    return meta, "\n\n".join(parts)


def do_translate(document_id, target_language, source_language):
    guard = _need_document(document_id)
    if guard:
        return guard, ""
    if not target_language or not target_language.strip():
        return '<div class="dm-card dm-abstain">Choose a target language.</div>', ""

    target = target_language.strip().split()[0].lower()
    try:
        result, source, detected, provider, elapsed = pipeline.translate(
            document_id.strip(), target, (source_language or "").strip() or None
        )
    except Exception as exc:
        return f'<div class="dm-card dm-warn">Translation failed: {exc}</div>', ""

    origin = "detected" if detected else ("declared" if source != "auto" else "undetermined")
    dropped = (
        '<div class="dm-warn">CONTENT DROPPED — some segments returned nothing.</div>'
        if result.content_dropped
        else ""
    )
    meta = (
        f'<div class="dm-card"><div class="dm-note">TRANSLATION</div><div class="dm-kv">'
        f'<div><span class="k">source</span><span class="v">{source} ({origin})</span></div>'
        f'<div><span class="k">target</span><span class="v">{target}</span></div>'
        f'<div><span class="k">segments</span>'
        f'<span class="v">{result.segments_translated}/{result.segments_total}</span></div>'
        f'<div><span class="k">elapsed</span><span class="v">{_ms(elapsed)}</span></div>'
        f'<div><span class="k">provider</span><span class="v">{provider}</span></div>'
        f"</div></div>{dropped}"
    )
    return meta, result.text


with gr.Blocks(title="DocuMind") as demo:
    gr.HTML(
        '<div id="dm-head"><h1>DOCUMIND</h1>'
        "<p>Grounded document intelligence — a document becomes memory, a question becomes a "
        "signal, and the nodes it reaches are your citations. It declines when nothing "
        "activates.</p></div>"
    )
    gr.HTML(_status_bar())

    with gr.Row():
        upload = gr.File(
            label="Document (PDF, TXT, PNG, JPG)",
            file_types=[".pdf", ".txt", ".png", ".jpg", ".jpeg"],
            scale=3,
        )
        ingest_btn = gr.Button("Read into memory", variant="primary", scale=1)

    document_id = gr.Textbox(
        label="Document handle",
        info="Filled in when a document is read. Every tab below works off this value.",
        interactive=False,
    )
    ingest_error = gr.Markdown()
    ingest_summary = gr.HTML()
    ingest_nodes = gr.HTML()

    with gr.Tabs():
        with gr.Tab("Recall"):
            gr.Markdown(
                "Ask a question. The answer is generated **only** from retrieved passages — "
                "if they don't support one, it declines instead of guessing."
            )
            question = gr.Textbox(label="Question", lines=2)
            ask_btn = gr.Button("Ask", variant="primary")
            answer_out = gr.HTML()
            citations_out = gr.Markdown()

        with gr.Tab("Activation"):
            gr.Markdown(
                "The retrieval pool with the lid off — every candidate, its score, and whether "
                "it survived reranking. Retrieval only, no generation."
            )
            retr_question = gr.Textbox(label="Question", lines=2)
            retr_btn = gr.Button("Retrieve", variant="primary")
            retr_note = gr.HTML()
            retr_table = gr.Dataframe(
                headers=["rank", "score", "survives", "page", "passage"],
                datatype=["number", "str", "str", "str", "str"],
                wrap=True,
            )

        with gr.Tab("Summarize"):
            gr.Markdown(
                "Short documents are summarized in one pass; long ones go through map-reduce "
                "over page groups. The strategy that actually ran is reported."
            )
            sum_btn = gr.Button("Summarize", variant="primary")
            sum_meta = gr.HTML()
            sum_out = gr.Markdown()

        with gr.Tab("Translate"):
            gr.Markdown(
                "Whole-document translation, segmented on sentence boundaries. Segment counts "
                "are reported so silent truncation would be visible."
            )
            with gr.Row():
                target_lang = gr.Dropdown(
                    label="Target language",
                    choices=["hi Hindi", "es Spanish", "fr French", "de German",
                             "ja Japanese", "ar Arabic", "zh-CN Chinese", "en English"],
                    value="hi Hindi",
                )
                source_lang = gr.Textbox(
                    label="Source language (optional)",
                    placeholder="blank = detect offline",
                )
            tr_btn = gr.Button("Translate", variant="primary")
            tr_meta = gr.HTML()
            tr_out = gr.Textbox(label="Translated text", lines=12)

        with gr.Tab("Evidence"):
            gr.HTML(_evidence_tab())

    gr.HTML(
        '<div class="dm-note" style="margin-top:18px;padding-top:12px;'
        'border-top:1px solid rgba(126,178,214,0.16)">'
        "Uploads are session-scoped and are not persisted. · The full visualized experience — "
        "a Schwarzschild raytracer rendering the same backend — runs in the project's Colab "
        "notebook, because a WebGL frontend cannot be hosted on this tier. Neither surface is "
        "a cut-down version of the other.</div>"
    )

    ingest_btn.click(
        do_ingest, [upload], [document_id, ingest_error, ingest_summary, ingest_nodes]
    )
    ask_btn.click(do_ask, [question, document_id], [answer_out, citations_out])
    retr_btn.click(do_retrieve, [retr_question, document_id], [retr_table, retr_note])
    sum_btn.click(do_summarize, [document_id], [sum_meta, sum_out])
    tr_btn.click(do_translate, [document_id, target_lang, source_lang], [tr_meta, tr_out])


if __name__ == "__main__":
    # css and theme belong to launch() on Gradio 6; passing them to Blocks is
    # accepted with a warning and then ignored, which silently dropped the
    # entire DocuMind palette on the deployed Space.
    demo.launch(
        css=CSS,
        theme=gr.themes.Base(),
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
