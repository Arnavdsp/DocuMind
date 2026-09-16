# Verification tools

These are not tests you run in CI; they are instruments for looking at a
renderer that is otherwise very easy to be wrong about. Each one caught a real
defect during the build.

Run them from `frontend/`, with `playwright` installed (`npm i -D playwright`).
`ui-check.mjs` and `uniform-probe.mjs` need the API running on :8000.

| Tool | What it answers |
|---|---|
| `render-check.mjs` | Does the shader compile, and what does each simulation state actually look like? Renders eight representative states (poster, edge-on, empty index, mid-collapse, abstained, low-integrity, redshifted, lowest quality tier) to PNG with lit/hot/mean-luma measurements, so state differences are quantitative rather than a matter of opinion. |
| `frame-probe.mjs` | How large is the disk in frame at a given camera distance, and does the quality tier change *what* is drawn? Sweeps camera distance and step count and measures the disk's on-screen extent. Confirmed that 110 and 460 steps produce the same image at different fidelity. |
| `ui-check.mjs` | Does the whole app work against a live backend? Drives a real browser: uploads a document, asks a question, opens the params panel, screenshots each step and dumps every telemetry row. |
| `uniform-probe.mjs` | What is the shader *actually* being handed? Monkey-patches `WebGLRenderingContext` before page load to capture every uniform write. This is what found `uCollapse` stuck at 0 on a re-uploaded document — the render looked wrong and every other explanation was plausible. |

```bash
node tools/render-check.mjs
node tools/frame-probe.mjs
node tools/ui-check.mjs /path/to/document.txt mytag
node tools/uniform-probe.mjs
```
