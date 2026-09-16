---
title: DocuMind
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.27.0
app_file: app.py
python_version: "3.12"
short_description: Grounded document Q&A that declines when evidence is thin
license: mit
pinned: false
---

# DocuMind

Ask questions of a document and get answers grounded in retrieved passages,
with page-anchored citations — or an honest refusal when the document does not
support an answer.

A document is read once and becomes a set of **memory nodes**. A question
propagates as a **signal**; the nodes it reaches **fire**, and those are your
citations. When nothing activates strongly enough, the system says so instead
of confabulating. That refusal is a designed response, not an error state, and
it is the behaviour the project is built around.

## Tabs

| Tab | What it does | Cost |
|---|---|---|
| **Recall** | Question in, grounded answer out, with citations, measured per-stage timings and the abstention reading | Groq API call |
| **Activation** | The retrieval pool with the lid off — every candidate, its score, and whether it survived reranking. No generation. | CPU only |
| **Evidence** | Where the published before/after evaluation table will live | none |

## Honesty rules this app follows

- **Every number on screen is measured.** Anything unmeasured renders as an
  em-dash. The Evidence tab is deliberately all em-dashes: the evaluation
  harness has not been run yet, and the project does not print a number it has
  not measured.
- **No confidence percentage, anywhere.** Retrieval relevance is a retrieval
  score, not a probability that the answer is correct, and it is never dressed
  up as one.
- **The backend identifies itself.** If stand-in models are active, the page
  says so in a banner rather than quietly returning weaker answers.
- **Abstention is first-class.** It gets the same visual weight as an answer.

## How it runs

Generation is a [Groq](https://groq.com) API call. Embedding
(`all-MiniLM-L6-v2`, ~90 MB) and reranking (`ms-marco-MiniLM-L-6-v2`, ~90 MB)
run locally on CPU. Nothing in the pipeline needs a GPU, so no visitor GPU
quota is consumed.

Set `GROQ_API_KEY` as a Space secret. Without it the app still loads and
retrieval still works; generation reports that it is unavailable rather than
falling back to something weaker without saying so.

Uploads are session-scoped and are not persisted.

## This is not the whole project

The full visualized experience — a Schwarzschild null-geodesic raytracer that
renders retrieval as light bending around a black hole, running the same
backend — lives in the project's Colab notebook. A WebGL frontend cannot be
hosted on the Gradio SDK at all, so this Space is a deliberately different
surface built for this tier, not a reduced version of that one. Neither is a
cut-down of the other.

Source: [Arnavdsp/DocuMind](https://github.com/Arnavdsp/DocuMind)

## Licence

MIT.
