#!/usr/bin/env python3
"""Generate notebooks/gargantua.ipynb.

The notebook embeds the entire project as a base64 gzipped tarball so it is
self-contained: opening it in Colab and running all cells needs no clone, no
credentials and no network access to this repository.

Regenerate it whenever the source changes:

    python tools/build_notebook.py

The archive deliberately excludes node_modules, dist, data and the venv (the
notebook rebuilds those) and docs/ (README screenshots, ~1 MB of PNG that
would triple the notebook's size for no benefit inside Colab).
"""

from __future__ import annotations

import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "gargantua.ipynb"

INCLUDE = [
    "backend/app",
    "backend/tests",
    "backend/requirements.txt",
    "backend/requirements-ml.txt",
    "backend/requirements-dev.txt",
    "backend/pyproject.toml",
    "backend/Dockerfile",
    "frontend/src",
    "frontend/public",
    "frontend/index.html",
    "frontend/package.json",
    "frontend/package-lock.json",
    "frontend/vite.config.ts",
    "frontend/postcss.config.js",
    "frontend/tsconfig.json",
    "frontend/tsconfig.app.json",
    "frontend/tsconfig.node.json",
    "frontend/.oxlintrc.json",
    "frontend/tools",
    "README.md",
    "MIGRATION.md",
    "LICENSE",
    ".env.example",
    ".gitignore",
    "docker-compose.yml",
    "pytest.ini",
]

EXCLUDE_PARTS = {
    "node_modules",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "render-check",
    ".git",
}


def _keep(path: Path) -> bool:
    return not any(part in EXCLUDE_PARTS for part in path.parts)


def build_archive() -> str:
    buf = io.BytesIO()
    count = 0
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for entry in INCLUDE:
            src = ROOT / entry
            if not src.exists():
                raise SystemExit(f"missing from repo: {entry}")
            if src.is_file():
                tar.add(src, arcname=entry)
                count += 1
                continue
            for file in sorted(src.rglob("*")):
                if file.is_file() and _keep(file.relative_to(ROOT)):
                    tar.add(file, arcname=str(file.relative_to(ROOT)))
                    count += 1
    print(f"archived {count} files, {buf.tell() / 1024:.0f} KB gzipped")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def main() -> None:
    archive = build_archive()

    cells: list[dict] = []

    # ---------------------------------------------------------------- intro
    cells.append(
        md(
            """# GARGANTUA — Document Intelligence

**Run all cells.** You will get one public URL serving both the interface and
the API, with real models on this runtime's GPU.

This notebook is self-contained. It carries the entire project as an embedded
archive — no clone, no credentials, no access to GitHub required.

---

### Before you start

Set the runtime to a GPU: **Runtime → Change runtime type → T4 GPU** (or better).

It will run on CPU, but generation on `Phi-3-mini` without a GPU takes minutes
per answer rather than seconds. The notebook tells you which one you got and
the interface reports it on screen — it never claims a capability it doesn't
have.

### What it does

| Cell | |
|---|---|
| 1 | Check the runtime — GPU, CUDA, `torch`, Node |
| 2 | Unpack the project |
| 3 | Install Python dependencies *without* touching Colab's CUDA-matched `torch` |
| 4 | Build the frontend |
| 5 | Verify — 48 tests, mock backend, no downloads |
| 6 | Start the API with the built UI mounted at `/` |
| 7 | Open a public tunnel and print your URL |
| 8 | Smoke-test the live server end to end |

Total: roughly five minutes, most of it model weights downloading on your first
question.

---

### What you're looking at

The document becomes the mass. Its chunks become the accretion disk. Asking a
question traces null geodesics toward the singularity — most fall past the
horizon, a few strike the disk and light up. **Those hot spots are your
citations**, positioned by page number and heated by their real relevance
scores.

Every number on screen is measured. Where a value isn't measured, the HUD shows
an em-dash. There is no confidence percentage anywhere, on purpose — see
`README.md`.
"""
        )
    )

    # ------------------------------------------------------------- cell 1
    cells.append(md("## 1 · Runtime check\n\nNothing is installed yet. This only reports what you have."))
    cells.append(
        code(
            '''import shutil, subprocess, sys, platform

print(f"python   : {platform.python_version()}  ({sys.executable})")

# --- GPU -------------------------------------------------------------------
GPU_NAME = None
try:
    import torch
    if torch.cuda.is_available():
        GPU_NAME = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability()
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"torch    : {torch.__version__}  (CUDA {torch.version.cuda})")
        print(f"gpu      : {GPU_NAME}  compute {major}.{minor}  {total:.1f} GB")
        # bf16 needs Ampere+ (compute 8.x). A T4 is 7.5 and produces garbage
        # or errors with bf16, which is why HFModelService detects capability
        # rather than assuming bf16 for "any GPU".
        print(f"dtype    : {'bfloat16' if major >= 8 else 'float16'} will be selected")
    else:
        print(f"torch    : {torch.__version__}  (no CUDA device)")
except ImportError:
    print("torch    : not installed yet")

if GPU_NAME is None:
    print()
    print("  !! No GPU detected. The app will still run, but generation will be")
    print("     very slow. Runtime -> Change runtime type -> T4 GPU, then")
    print("     re-run from this cell.")

# --- Node ------------------------------------------------------------------
node = shutil.which("node")
if node:
    v = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()
    print(f"node     : {v}  ({node})")
else:
    print("node     : not found — cell 4 will install it")

# --- Tesseract, for scanned pages -----------------------------------------
tess = shutil.which("tesseract")
print(f"tesseract: {'found' if tess else 'not found — cell 3 will install it'}")
'''
        )
    )

    # ------------------------------------------------------------- cell 2
    cells.append(
        md(
            """## 2 · Unpack the project

`_ARCHIVE_B64` below is the whole repository — backend, frontend, tests, docs —
as a base64 gzipped tarball. Nothing is reconstructed from memory or fetched
over the network."""
        )
    )
    cells.append(code(f'_ARCHIVE_B64 = "{archive}"\n'))
    cells.append(
        code(
            '''import base64, io, os, tarfile, pathlib

PROJECT = pathlib.Path("/content/gargantua")
if PROJECT.exists():
    import shutil
    # Preserve build caches across re-runs so a second run is fast.
    for keep in ("frontend/node_modules", "data"):
        src = PROJECT / keep
        if src.exists():
            shutil.move(str(src), f"/content/.keep_{keep.replace('/', '_')}")
    shutil.rmtree(PROJECT)

PROJECT.mkdir(parents=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(_ARCHIVE_B64))) as tar:
    tar.extractall(PROJECT)

# Restore anything we stashed above.
import shutil
for keep in ("frontend/node_modules", "data"):
    stash = pathlib.Path(f"/content/.keep_{keep.replace('/', '_')}")
    if stash.exists():
        (PROJECT / keep).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stash), str(PROJECT / keep))
        print(f"restored cached {keep}")

os.chdir(PROJECT)
files = [p for p in PROJECT.rglob("*") if p.is_file() and "node_modules" not in p.parts]
print(f"extracted {len(files)} files to {PROJECT}")
print()
for top in sorted(p for p in PROJECT.iterdir()):
    print(f"  {top.name}{'/' if top.is_dir() else ''}")
'''
        )
    )

    # ------------------------------------------------------------- cell 3
    cells.append(
        md(
            """## 3 · Python dependencies

**The one thing this cell is careful about:** it does not pin or reinstall
`torch`. Colab ships a build matched to its CUDA driver, and `pip install
torch` on top of it is the single most common way to break GPU support in a
Colab notebook — you end up with a CPU wheel and no error message. The ML
requirements file deliberately omits `torch` for the same reason.

Expect a couple of minutes. Model *weights* are not downloaded here; they load
lazily on your first question."""
        )
    )
    cells.append(
        code(
            '''import subprocess, sys, shutil

def run(cmd, **kw):
    print(f"$ {' '.join(cmd)}")
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        raise SystemExit(f"failed ({r.returncode}): {' '.join(cmd)}")

# --- Tesseract, for OCR on scanned pages ----------------------------------
if not shutil.which("tesseract"):
    run(["apt-get", "-qq", "update"])
    run(["apt-get", "-qq", "install", "-y", "tesseract-ocr"])
    print("tesseract installed")

# --- Core + ML -------------------------------------------------------------
run([sys.executable, "-m", "pip", "install", "-q",
     "-r", "backend/requirements.txt",
     "-r", "backend/requirements-ml.txt",
     "-r", "backend/requirements-dev.txt"])

# --- Confirm we did not clobber the GPU build -----------------------------
import importlib
import torch
importlib.reload(torch)
print()
print(f"torch {torch.__version__} — CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("  !! CUDA is no longer available. Something replaced the Colab torch")
    print("     build. Runtime -> Disconnect and delete runtime, then start over.")
else:
    print(f"  {torch.cuda.get_device_name(0)}")
'''
        )
    )

    # ------------------------------------------------------------- cell 4
    cells.append(
        md(
            """## 4 · Build the frontend

Produces `frontend/dist`, which FastAPI mounts at `/` — **last**, so it never
shadows `/api/*`. That mount is what makes this a single URL rather than two
services and a CORS problem.

The build also verifies the lazy boundary holds: the shader and WebGL host must
land in their own chunk so they never sit on the critical path or block the
upload."""
        )
    )
    cells.append(
        code(
            '''import subprocess, shutil, os, pathlib, glob

# --- Node ------------------------------------------------------------------
if not shutil.which("node"):
    print("installing node 22...")
    subprocess.run(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null 2>&1 "
        "&& apt-get install -y -qq nodejs",
        shell=True, check=True,
    )
print("node", subprocess.run(["node","--version"], capture_output=True, text=True).stdout.strip())

os.chdir("/content/gargantua/frontend")

# npm ci is reproducible from the lockfile; it also wipes a partial
# node_modules from an interrupted earlier run.
install = "ci" if pathlib.Path("package-lock.json").exists() else "install"
print(f"\\n$ npm {install}")
subprocess.run(["npm", install, "--no-audit", "--no-fund"], check=True)

print("\\n$ npm run build")
subprocess.run(["npm", "run", "build"], check=True)

os.chdir("/content/gargantua")

# --- Verify the build -------------------------------------------------------
dist = pathlib.Path("frontend/dist")
assert (dist / "index.html").exists(), "no index.html in dist"

scene_chunks = [p for p in dist.glob("assets/*.js") if "traceGeodesic" in p.read_text(errors="ignore")]
entry_chunks = [p for p in dist.glob("assets/index-*.js")]
print("\\n--- build ---")
for p in sorted(dist.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(dist)}  {p.stat().st_size/1024:.1f} KB")

ok = len(scene_chunks) == 1 and not any(c in scene_chunks for c in entry_chunks)
print()
print(f"lazy boundary: shader is in {scene_chunks[0].name if scene_chunks else 'NOWHERE'}"
      f" — {'held' if ok else 'BROKEN (shader hoisted into the entry chunk)'}")
'''
        )
    )

    # ------------------------------------------------------------- cell 5
    cells.append(
        md(
            """## 5 · Verify the intelligence layer

48 tests across 11 modules, on the deterministic mock backend — no GPU, no
downloads, no network. This is the suite that came with the original build and
it is unmodified: the rebuild replaced the frontend, not the RAG.

If this fails, stop here. Nothing downstream is worth looking at."""
        )
    )
    cells.append(
        code(
            '''import subprocess, sys, os
os.chdir("/content/gargantua")
r = subprocess.run(
    [sys.executable, "-m", "pytest", "backend/tests", "-q", "--no-header"],
    env={**os.environ, "MODEL_BACKEND": "mock"},
    capture_output=True, text=True,
)
print(r.stdout[-2500:])
if r.returncode != 0:
    print(r.stderr[-2000:])
    raise SystemExit("test suite failed — do not continue")
'''
        )
    )

    # ------------------------------------------------------------- cell 6
    cells.append(
        md(
            """## 6 · Start the API

Runs `uvicorn` in the background with the built SPA mounted at `/`.

`MODEL_BACKEND=auto` picks the real Hugging Face models when `torch` and
`transformers` import, and falls back to the mock otherwise. The cell reports
which one you actually got — and so does the interface, on screen, for every
answer.

Weights are **not** downloaded here. `HFModelService` loads lazily on first
use, so the first question you ask will take a minute or two while
`all-MiniLM-L6-v2` and `Phi-3-mini` come down. Everything after that is fast."""
        )
    )
    cells.append(
        code(
            '''import os, subprocess, sys, time, urllib.request, json, signal, pathlib

os.chdir("/content/gargantua")
LOG = pathlib.Path("/content/uvicorn.log")

# Stop a server left over from an earlier run of this cell.
subprocess.run(["pkill", "-f", "uvicorn app.main:app"], capture_output=True)
time.sleep(1)

env = {
    **os.environ,
    "PYTHONPATH": "/content/gargantua/backend",
    "ENVIRONMENT": "colab",
    "DATA_DIR": "/content/gargantua/data",
    "MODEL_BACKEND": "auto",
    "LOG_LEVEL": "INFO",
    "PYTHONUNBUFFERED": "1",
}

with LOG.open("w") as log:
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", "127.0.0.1", "--port", "8000", "--log-level", "info"],
        cwd="/content/gargantua/backend", env=env, stdout=log, stderr=subprocess.STDOUT,
    )

# --- Wait for liveness -----------------------------------------------------
health = None
for attempt in range(60):
    if server.poll() is not None:
        print(LOG.read_text()[-3000:])
        raise SystemExit(f"uvicorn exited with code {server.returncode}")
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as r:
            health = json.loads(r.read())
            break
    except Exception:
        time.sleep(1)

if health is None:
    print(LOG.read_text()[-3000:])
    raise SystemExit("server did not become healthy within 60s")

print(f"health    : {health}")

ready = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/readiness", timeout=5).read())
print(f"readiness : {ready}")

# --- Which model backend did we actually get? -----------------------------
sys.path.insert(0, "/content/gargantua/backend")
for key in ("ENVIRONMENT", "DATA_DIR", "MODEL_BACKEND"):
    os.environ[key] = env[key]
from app.services.model_service import get_model_service
svc = get_model_service()
print(f"backend   : {svc.backend_name}")
print(f"device    : {svc.device_info}")
if svc.backend_name == "mock":
    print()
    print("  !! Running on the MOCK backend. Relevance scores and grounding are")
    print("     NOT meaningful and abstention will not trigger reliably. The UI")
    print("     says so on screen. Check cell 3 for a torch/CUDA problem.")

# --- The SPA is served by the API itself ----------------------------------
page = urllib.request.urlopen("http://127.0.0.1:8000/", timeout=5).read().decode()
print(f"spa at /  : {'GARGANTUA' in page}")
'''
        )
    )

    # ------------------------------------------------------------- cell 7
    cells.append(
        md(
            """## 7 · Your URL

Opens a public tunnel to port 8000.

Defaults to a **Cloudflare quick tunnel**, which needs no account and no token.
If you would rather use ngrok, set `NGROK_AUTHTOKEN` in the cell below.

The URL is ephemeral and dies with this runtime. That is a property of Colab,
not a deployment — `docker-compose.yml` is the path to real hosting."""
        )
    )
    cells.append(
        code(
            '''NGROK_AUTHTOKEN = ""   # optional — leave empty to use Cloudflare

import os, re, subprocess, time, pathlib, urllib.request

PUBLIC_URL = None

if NGROK_AUTHTOKEN.strip():
    subprocess.run(["pip", "install", "-q", "pyngrok"], check=True)
    from pyngrok import ngrok, conf
    conf.get_default().auth_token = NGROK_AUTHTOKEN.strip()
    for t in ngrok.get_tunnels():
        ngrok.disconnect(t.public_url)
    PUBLIC_URL = ngrok.connect(8000, "http").public_url
else:
    # Cloudflare quick tunnel — no account required.
    if not pathlib.Path("/usr/local/bin/cloudflared").exists():
        subprocess.run(
            "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/"
            "cloudflared-linux-amd64 -O /usr/local/bin/cloudflared "
            "&& chmod +x /usr/local/bin/cloudflared",
            shell=True, check=True,
        )
    subprocess.run(["pkill", "-f", "cloudflared"], capture_output=True)
    time.sleep(1)

    tunnel_log = pathlib.Path("/content/cloudflared.log")
    with tunnel_log.open("w") as log:
        subprocess.Popen(
            ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000", "--no-autoupdate"],
            stdout=log, stderr=subprocess.STDOUT,
        )

    for _ in range(45):
        time.sleep(1)
        m = re.search(r"https://[-a-z0-9]+\\.trycloudflare\\.com", tunnel_log.read_text())
        if m:
            PUBLIC_URL = m.group(0)
            break

if not PUBLIC_URL:
    raise SystemExit("tunnel did not come up — re-run this cell")

# Confirm the tunnel actually reaches the app, rather than just existing.
for _ in range(15):
    try:
        body = urllib.request.urlopen(PUBLIC_URL, timeout=10).read().decode()
        if "GARGANTUA" in body:
            break
    except Exception:
        pass
    time.sleep(2)

print("=" * 68)
print()
print(f"    {PUBLIC_URL}")
print()
print("    Open it. Drop in a PDF or a text file.")
print("=" * 68)

try:
    from IPython.display import display, HTML
    display(HTML(
        f'<a href="{PUBLIC_URL}" target="_blank" rel="noopener" '
        f'style="display:inline-block;margin-top:12px;padding:12px 22px;'
        f'background:#000;color:#eca74f;border:1px solid #254353;'
        f'font-family:ui-monospace,monospace;font-size:12px;letter-spacing:.18em;'
        f'text-transform:uppercase;text-decoration:none">Open GARGANTUA &rarr;</a>'
    ))
except Exception:
    pass
'''
        )
    )

    # ------------------------------------------------------------- cell 8
    cells.append(
        md(
            """## 8 · Smoke test

Optional, but it proves the whole path works before you trust the interface:
upload → poll the job through its real stages → ask a grounded question → get
citations back with page numbers and scores.

On the HF backend the first run downloads model weights, so give it a couple of
minutes."""
        )
    )
    cells.append(
        code(
            '''import io, json, time, urllib.request

BASE = "http://127.0.0.1:8000"

def post_file(path, name, data, ctype):
    boundary = "----gargantua"
    body = (
        f"--{boundary}\\r\\nContent-Disposition: form-data; name=\\"file\\"; "
        f"filename=\\"{name}\\"\\r\\nContent-Type: {ctype}\\r\\n\\r\\n"
    ).encode() + data + f"\\r\\n--{boundary}--\\r\\n".encode()
    req = urllib.request.Request(
        BASE + path, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=120).read())

def post_json(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=600).read())

def get(path):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=60).read())

SAMPLE = (
    "Quarterly Operations Review\\n\\n"
    "Revenue for the third quarter reached 42.7 million dollars, an increase of "
    "18 percent year over year. Gross margin was 61 percent, up from 57 percent "
    "in the prior quarter.\\n\\n"
    "Headcount grew from 310 to 366 employees. Customer churn fell to 4.2 percent, "
    "the lowest figure recorded since 2021.\\n\\n"
    "The primary risk identified is revenue concentration: the three largest "
    "customers account for 38 percent of total revenue. Mitigation is under way "
    "but no timeline has been committed.\\n"
).encode()

up = post_file("/api/documents", "ops_review.txt", SAMPLE, "text/plain")
doc_id, job_id = up["document"]["document_id"], up["job_id"]
print(f"uploaded  : {doc_id[:16]}...")

seen = []
for _ in range(600):                       # generous: first run loads weights
    job = get(f"/api/jobs/{job_id}")
    if not seen or seen[-1] != job["stage"]:
        seen.append(job["stage"])
        print(f"  stage   : {job['stage']}  ({job['progress']:.0%})")
    if job["status"] in ("succeeded", "failed"):
        break
    time.sleep(0.9)                        # the same 900ms the UI polls at

if job["status"] != "succeeded":
    raise SystemExit(f"ingestion failed: {job['error_message']}")

doc = get(f"/api/documents/{doc_id}")
m = doc["metrics"]
print(f"\\nindexed   : {m['word_count']} words, {m['page_count']} page(s), "
      f"{m['character_count']} chars")

print("\\n--- a question the document answers ---")
a = post_json(f"/api/documents/{doc_id}/ask", {"question": "What was the gross margin?"})
print(f"grounding : {a['grounding']}   relevance: {a['relevance_score']:.4f}   "
      f"abstained: {a['abstained']}   backend: {a['model_used']}")
print(f"answer    : {a['answer'][:300]}")
for c in a["citations"]:
    print(f"  cite    : page {c['page_number']}  score {c['relevance_score']:.4f}  "
          f"{c['snippet'][:70]}...")

print("\\n--- a question it does not ---")
b = post_json(f"/api/documents/{doc_id}/ask",
              {"question": "What is the melting point of tungsten?"})
print(f"grounding : {b['grounding']}   abstained: {b['abstained']}   "
      f"citations: {len(b['citations'])}")
if b["abstained"]:
    print("            correct — this is SIGNAL LOST in the UI, and it is the")
    print("            system working, not failing.")
elif a["model_used"] == "mock":
    print("            expected on the mock backend: its hashed bag-of-words")
    print("            embeddings give spurious similarity. Not meaningful.")
else:
    print("            did NOT abstain — worth investigating.")

print("\\n--- summarize ---")
s = post_json(f"/api/documents/{doc_id}/summarize", {"force_refresh": False})
sm = s["summary"]
print(f"strategy  : {s['strategy']}  cached: {s['cached']}")
print(f"summary   : {sm['executive_summary'][:280]}")
print(f"numbers   : {sm['important_numbers'][:6]}")
print(f"findings  : {len(sm['key_findings'])}")

print("\\n--- source integrity ---")
pg = get(f"/api/documents/{doc_id}/pages")["pages"][0]
print(f"page 1    : method={pg['extraction_method']}  "
      f"ocr_confidence={pg['ocr_confidence']}  low_quality={pg['is_low_quality']}")

print("\\nall good.")
'''
        )
    )

    # ------------------------------------------------------------- outro
    cells.append(
        md(
            """---

## Using it

Open the URL from cell 7 and drop in a PDF, a text file or a scanned image
(25 MB max).

**Keyboard** — `1`–`4` camera presets, `C` cinematic sequence, `R` auto-orbit,
`P` parameters, `M` sound, `H` hide the HUD. Drag to orbit, scroll to zoom.

**Panels** — `MASS` is the collapse sequence, driven by the real ingestion job.
`QUERY` traces geodesics and returns citations you can click. `BRIEF` is the
structured summary. `SHIFT` is translation as a redshift control. `SOURCE` is
the per-page extraction audit, with OCR confidence and quality flags stated
rather than smoothed over.

**Watch for** — the disk stays dark until something is indexed, because an
empty index has nothing to glow with. Ask something the document cannot support
and the disk goes dark again: *NO GROUNDED SIGNAL*. That is the point of the
system, not a failure of it.

## Notes

- **Colab is a demo environment, not hosting.** The runtime recycles, the URL
  dies with it, and `/content` is not durable storage. Re-running cells 6 and 7
  brings the server and tunnel back; re-running cell 2 keeps `node_modules` and
  `data` so the second run is much faster.
- **The tunnel URL is public** while it lives. Anything you upload is reachable
  by anyone who has it.
- **Stopping:** `Runtime → Disconnect and delete runtime`, or run
  `!pkill -f uvicorn; pkill -f cloudflared`.
- **Server logs** are at `/content/uvicorn.log` — `!tail -50 /content/uvicorn.log`.
- To publish this as a repository, cell 2 has already written the whole thing to
  `/content/gargantua`, `README.md` and `.gitignore` included.
"""
        )
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024 / 1024:.1f} MB, {len(cells)} cells)")


if __name__ == "__main__":
    main()
