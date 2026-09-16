# MIGRATION

What changed from the notebook version of the Document Intelligence Suite, and
why. Ordered by how much of an argument it needs.

---

## The intelligence layer: nothing changed

This is the headline, so it goes first.

`app/rag/retrieval.py`, `app/rag/generation.py`, `app/rag/reranker.py`,
`app/rag/chunking.py`, `app/rag/vector_store.py`, `app/services/*`,
`app/ingestion/*`, `app/storage/*`, `app/api/routes/*`, `app/config.py` and
`app/schemas/*` have **zero diff** against the version extracted from
`document_intelligence_suite_final.ipynb`.

All 10 endpoints, all response shapes, `retrieval_top_k=8`,
`rerank_top_k=4`, `min_relevance_score=0.18`, embed-once-at-ingestion,
structured logging with request IDs, and `AppError` → user-safe message while
internals go to logs — all preserved exactly. The test suite still passes
unmodified: **48 tests, 11 modules.**

The rebuild is a frontend replacement. That was the constraint, and it held.

---

## The frontend: replaced entirely

### Deleted

| Removed | Why |
|---|---|
| `src/index.css` (ink/paper/brass/phosphor tokens) | Replaced by `design-tokens.css`. No token from the old palette survives. |
| `src/pages/landing.tsx`, `src/pages/workspace.tsx` | There are no longer two pages. There is one continuous simulation. |
| `src/components/ui/*` (button, card, tabs, badge) | Generic shadcn-style primitives carry visual assumptions — rounded corners, filled surfaces, shadow elevation — that fight the aesthetic at every turn. Replaced by two CSS classes, `.panel` and `.control`. |
| `src/components/workspace/*-tab.tsx` | The five capabilities were designed into the world rather than tabbed beside it. |
| `src/scenes/intelligence-core.tsx` | Superseded by the raytracer. |
| `react-router-dom` | One route. The router was 20 KB to express that. |
| `three`, `@react-three/fiber`, `@react-three/drei` | See below. |
| `framer-motion`, `lucide-react`, `@radix-ui/*` | Nothing left that used them. |

Bundle: **190 KB React + 40 KB app + 33 KB lazy scene chunk** (59/13/12 KB
gzipped). The previous build shipped three.js and R3F on the critical path.

### `three` / react-three-fiber → raw WebGL

The scene is one full-screen triangle running one fragment shader. three.js
contributes a scene graph, a material system, a camera abstraction and a render
loop, none of which this uses, at roughly 600 KB. Its internal `useFrame` loop
would also have had to be fought to implement the visibility-and-offscreen
pausing the performance budget requires.

`src/sim/renderer.ts` is ~250 lines of WebGL and owns its own loop. This is a
deviation from the stack the brief described, and it is the one place the build
chose against the existing dependency set — stated here rather than quietly.

### Palette: sampled, not described

The brief characterised the reference interface as "thin hairline rules at ~8–14%
white" and "everything uppercase, monospace... colour almost entirely absent."

The screenshots say otherwise. Pixel-sampling the reference gives a **cold
desaturated slate-cyan** chrome — `#3a6271` labels, `#9ed6ec` interactive text,
`#eafaff` live values — with **amber** (`#eca74f` / `#ffb454`) marking active
affordances and the mass subtitle. Pure white appears in exactly one place: the
GARGANTUA wordmark. Panel fill is `#050d18` at ~84%, not neutral black.

The brief said to match the screenshots where they conflict with the prose.
They conflicted; the screenshots won. Every value in `design-tokens.css` carries
the measurement it came from.

### Where the cinematic idea and honesty conflicted

The brief asked to be told about these rather than have them quietly resolved.
Three came up.

**1. Chunk count is not exposed by the API.**
"Accretion disk particles = chunks in the vector index" needs a number no
endpoint returns. Options were: add an endpoint (changes the API surface),
render an em-dash (loses the mapping entirely), or estimate.

Resolved by estimating from `character_count` against the configured chunk size
and overlap, and labelling it `~` wherever it appears as text. It is the only
estimated number in the interface and it is marked as one. If you would rather
it were exact, the honest fix is an `index_stats` field on `DocumentRecord`.

**2. The frequency-shift control is a presentation parameter.**
Translate as redshift/blueshift is a good conceit, but "how much shift" is not a
measurement of anything. Rendering it as a telemetry row with units would be
precisely the fabricated-telemetry failure the build exists to avoid.

Resolved by making it a labelled control, not a readout. The slider position
drives the shader in real time; the only measured facts on that panel are the
languages the backend reported and whether it truncated.

**3. Chrome amber sits next to the grounding ramp.**
The reference uses amber for latched toggles. The grounding ramp runs
amber → orange → white-hot. Visually adjacent, semantically unrelated.

Resolved by keeping two token families with different values and a hard rule:
`groundingColor()` is the only function permitted to return a
`--color-grounding-*` token, it returns `null` for an unknown level, and no
grounding-bound element falls back to the chrome accent. The alternative —
recolouring active toggles to avoid the adjacency — would have broken the
reference match for a problem that discipline solves.

### Shader corrections found by rendering it

`frontend/tools/render-check.mjs` compiles the shader in a headless WebGL
context and renders eight representative states to PNG with quantitative
measurements. It caught four real defects that reading the code would not have:

1. **Escape direction quantised to the integration step.** Using the position
   angle `φ` at the escape step meant adjacent pixels snapped to different
   steps, smearing every star into a horizontal dash. Fixed by computing the
   asymptotic tangent instead.
2. **Catastrophic cancellation in that tangent.** The natural form contains
   `-du/u²` and `1/u`, both of which lose all significant bits as `u → 0` —
   exactly when the direction is needed. Rewritten in the scaled form
   `u²·(dP/dφ)`, leaving two small comparable terms.
3. **The disk clamped to white.** A 2.6× gain on the temperature profile pushed
   the entire inner disk to `t = 1`, so the amber and orange halves of the ramp
   never appeared and everything rendered gold.
4. **Doppler asymmetry invisible.** Beaming was scaling luminance, which the
   filmic tonemap then compressed flat. It now shifts emission up the colour
   ramp as well — which is also more physically correct, since the observed
   temperature is Doppler-shifted, so the approaching limb is not merely
   brighter but whiter.

Star field aspect ratio was wrong too (2:1 cells in lat/long space).

### Behaviour that differs from the previous app

- **Tabs are gone as navigation.** The five capabilities are panels in one
  world; the app moves to Query automatically when the index becomes usable.
- **No confidence percentage anywhere.** Deliberate. See README.
- **Empty state shows no disk.** With nothing indexed, `uDiskLuminosity` is 0
  and the disk does not glow — you get the lensed starfield, the shadow and the
  photon ring. Uploading ignites it. This is austere on purpose: a glowing disk
  before ingestion would depict an index that does not exist.
- **Mission elapsed starts when the job succeeds**, not at upload. It measures
  the age of a usable index.
- **The collapse sequence can appear to skip stages.** Short documents finish
  faster than the 900 ms poll interval, so the ladder jumps to `ready`. Correct:
  the alternative is animating stages the backend never reported.

---

## Deployment: Vercel dropped

An earlier revision of this build targeted Vercel serverless functions —
`HostedModelService` behind the `ModelService` ABC, a serverless vector store,
Vercel Blob for documents.

That work was removed. Deploying only the frontend leaves you with an interface
and no intelligence, and a fully-serverless backend means giving up the local
model stack (`torch` + `transformers` alone vastly exceed the 250 MB unzipped
function limit, and there is no GPU), replacing it with paid hosted inference,
and re-architecting ingestion around a 10–60 s function timeout and an ephemeral
filesystem.

The replacement is `notebooks/gargantua.ipynb`: real models on a Colab GPU,
the frontend built and served by FastAPI itself, one public URL. It costs
nothing, needs no accounts, and runs the actual system rather than a reduced
one. `docker-compose.yml` remains the path to real hosting.
