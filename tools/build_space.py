#!/usr/bin/env python3
"""Assemble the Hugging Face Space bundle from the real source tree.

A Space is a git repo that must be flat and self-contained: `app.py` at the
root, with everything it imports importable from there. This repository keeps
the application under `backend/app/`, so the bundle is *generated* rather than
maintained by hand — the same approach `tools/build_notebook.py` already takes
for the Colab notebook.

That matters more than it looks. `space/` deliberately contains no retrieval
logic; if fusion, reranking or abstention needs a fix it happens in
`backend/app/rag/` and the Space inherits it on the next build. Two hand-kept
copies of retrieval logic diverge within a week, and then the public demo and
the published evaluation numbers describe different systems.

Usage:

    python tools/build_space.py [--out build/space]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Space-specific presentation layer. No retrieval logic lives here.
# packages.txt is apt-installed by the Space builder; without it pytesseract
# has no binary to call and OCR fails on every scanned document.
#
# It must contain ONLY bare package names, one per line. The builder runs
# `xargs -r -a /tmp/packages.txt apt-get install -y`, so a comment line is
# passed to apt as a list of package names and fails the whole build.
SPACE_FILES = ["app.py", "pipeline.py", "requirements.txt", "packages.txt", "README.md"]

EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".venv", ".git"}


def _keep(path: Path) -> bool:
    return not any(part in EXCLUDE_PARTS for part in path.parts)


def build(out_dir: Path) -> int:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    count = 0

    # 1. The presentation layer, flattened to the Space root.
    for name in SPACE_FILES:
        src = ROOT / "space" / name
        if not src.exists():
            raise SystemExit(f"missing from space/: {name}")
        shutil.copy2(src, out_dir / name)
        count += 1

    # 2. The real application, copied verbatim so `import app.*` resolves.
    #    Never edited in place — this is a copy of the source of truth.
    app_src = ROOT / "backend" / "app"
    if not app_src.exists():
        raise SystemExit("backend/app not found")
    for file in sorted(app_src.rglob("*")):
        if not file.is_file() or not _keep(file.relative_to(ROOT)):
            continue
        dest = out_dir / "app" / file.relative_to(app_src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, dest)
        count += 1

    # 3. A .gitattributes so the Space repo does not try to LFS small files.
    (out_dir / ".gitignore").write_text("__pycache__/\n*.pyc\ndata/\n*.npz\n*.db\n")
    count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="build/space", help="output directory")
    args = parser.parse_args()

    out_dir = (ROOT / args.out).resolve()
    count = build(out_dir)
    print(f"built {count} files into {out_dir}")
    print("push with:  hf upload <namespace>/DocuMind", out_dir, "--repo-type space")


if __name__ == "__main__":
    main()
