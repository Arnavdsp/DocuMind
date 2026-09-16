"""The Space must not own retrieval logic.

`space/` is a presentation layer. Chunking, retrieval, reranking, grounding
classification and abstention all live in `app.*` and are reached through the
adapter. If a retrieval behaviour needs fixing it is fixed in the backend and
the Space inherits it on the next build.

Two hand-maintained copies of retrieval logic diverge within a week, and then
the public demo and the published evaluation numbers describe different
systems — which would be the worst possible version of a fabricated number.
These tests are what stop that happening quietly.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SPACE_DIR = Path(__file__).resolve().parents[2] / "space"
PIPELINE = SPACE_DIR / "pipeline.py"
APP = SPACE_DIR / "app.py"

# Behaviour that must be imported, never redefined, in space/.
FORBIDDEN_DEFINITIONS = {
    "retrieve",
    "rerank",
    "build_reranker",
    "chunk_document",
    "chunk_page",
    "classify_grounding",
    "generate_grounded_answer",
    "build_citations",
    "reciprocal_rank_fusion",
    "weighted_score_fusion",
    "summarize_document",
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _defined_names(path: Path) -> set[str]:
    return {
        node.name
        for node in ast.walk(_tree(path))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def test_space_directory_exists():
    assert PIPELINE.exists(), "space/pipeline.py is missing"
    assert APP.exists(), "space/app.py is missing"


@pytest.mark.parametrize("path", [PIPELINE, APP])
def test_space_defines_no_retrieval_logic(path: Path):
    redefined = _defined_names(path) & FORBIDDEN_DEFINITIONS
    assert not redefined, (
        f"{path.name} defines {sorted(redefined)} itself. Retrieval behaviour belongs "
        f"in backend/app/rag/ so the Space and the eval harness measure the same system."
    )


def test_pipeline_imports_retrieval_from_the_backend():
    modules = _imported_modules(PIPELINE)
    for required in ("app.rag.retrieval", "app.rag.reranker", "app.rag.generation"):
        assert required in modules, f"space/pipeline.py must import {required}"


def test_pipeline_imports_chunking_and_the_vector_store_from_the_backend():
    modules = _imported_modules(PIPELINE)
    assert "app.rag.chunking" in modules
    assert "app.rag.vector_store" in modules


def test_app_layer_does_not_import_rag_internals_directly():
    """app.py talks to pipeline.py, not to the retrieval modules, so there is
    exactly one adapter seam rather than several."""
    rag_imports = {m for m in _imported_modules(APP) if m.startswith("app.rag")}
    assert not rag_imports, f"space/app.py should go through pipeline.py, not {sorted(rag_imports)}"


def test_no_hardcoded_credential_in_the_space():
    for path in (PIPELINE, APP):
        text = path.read_text()
        assert "gsk_" not in text, f"{path.name} appears to contain a Groq key"
        assert "hf_" not in text.replace("hf_hub", ""), f"{path.name} may contain an HF token"


def test_space_readme_has_valid_frontmatter():
    readme = (SPACE_DIR / "README.md").read_text()
    assert readme.startswith("---\n"), "Space README must open with YAML frontmatter"
    block = readme.split("---", 2)[1]
    for key in ("title:", "sdk:", "sdk_version:", "app_file:", "emoji:"):
        assert key in block, f"Space README frontmatter is missing {key}"
    # The server rejects a short_description over 60 characters.
    for line in block.splitlines():
        if line.startswith("short_description:"):
            value = line.split(":", 1)[1].strip()
            assert len(value) <= 60, f"short_description is {len(value)} chars, max 60"
