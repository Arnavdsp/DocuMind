#!/usr/bin/env python3
"""Latency and throughput benchmark for GARGANTUA.

    python tools/bench.py                 # in-process, against the current backend
    python tools/bench.py --http URL      # against a running server over HTTP
    MODEL_BACKEND=mock python tools/bench.py

WHAT THIS MEASURES, AND WHAT IT DOESN'T
---------------------------------------
Run with MODEL_BACKEND=mock and you are measuring the SYSTEM: chunking,
vector search, reranking, serialization, HTTP overhead. Model inference is
replaced by a deterministic stand-in that costs microseconds, so those
numbers isolate the cost the architecture itself imposes. They are a floor.

Run with MODEL_BACKEND=hf on a GPU and you are measuring the PRODUCT, where
embedding and generation dominate everything else by one to two orders of
magnitude.

Both are useful and they answer different questions. The report labels which
one it ran, because a p50 ask latency of 4 ms and one of 4 s are both true
statements about different systems and confusing them is worthless.

Percentiles, not means. A mean latency over a run that includes a cold model
load describes no request that actually happened.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@dataclass
class Samples:
    name: str
    unit: str = "ms"
    values: list[float] = field(default_factory=list)

    def add(self, v: float) -> None:
        self.values.append(v)

    def pct(self, p: float) -> float:
        if not self.values:
            return float("nan")
        s = sorted(self.values)
        # Nearest-rank; with n=20 a p95 is the 19th sample, and saying so is
        # better than interpolating a number no request produced.
        k = max(0, min(len(s) - 1, int(round(p / 100 * len(s) + 0.5)) - 1))
        return s[k]

    def row(self) -> str:
        if not self.values:
            return f"  {self.name:<34} —"
        return (
            f"  {self.name:<34} "
            f"n={len(self.values):<4} "
            f"p50={self.pct(50):>9.2f} "
            f"p95={self.pct(95):>9.2f} "
            f"max={max(self.values):>9.2f}  {self.unit}"
        )


class timer:
    def __init__(self, samples: Samples | None = None):
        self.samples = samples
        self.ms = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self.t0) * 1000
        if self.samples is not None:
            self.samples.add(self.ms)
        return False


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

_PARA = (
    "Revenue for the period reached {n}.{d} million dollars, an increase of "
    "{p} percent year over year. Gross margin was {m} percent and operating "
    "expenses grew {q} percent. Headcount moved from {a} to {b} employees "
    "while customer churn was {c} percent. The board reviewed the capital "
    "allocation plan and approved it without amendment. Management identified "
    "revenue concentration among the largest accounts as the principal risk.\n"
)


def make_document(paragraphs: int, seed: int = 7) -> bytes:
    import random

    rng = random.Random(seed)
    out = []
    for i in range(paragraphs):
        if i % 24 == 0:
            out.append(f"\nSection {i // 24 + 1}. Operating Review\n\n")
        out.append(
            _PARA.format(
                n=rng.randint(10, 900), d=rng.randint(0, 9), p=rng.randint(-20, 40),
                m=rng.randint(30, 70), q=rng.randint(-10, 30), a=rng.randint(200, 900),
                b=rng.randint(200, 900), c=rng.randint(1, 9),
            )
        )
    return "".join(out).encode()


QUESTIONS = [
    "What was the gross margin?",
    "How did headcount change over the period?",
    "What is the principal risk identified?",
    "What happened to customer churn?",
    "Did the board approve the capital allocation plan?",
    "How much did operating expenses grow?",
]


# ---------------------------------------------------------------------------
# In-process benchmark — component level
# ---------------------------------------------------------------------------


def bench_in_process(sizes: list[int], queries: int) -> dict:
    from app.config import get_settings
    from app.ingestion.extractors import extract
    from app.rag.chunking import chunk_document
    from app.rag.reranker import build_reranker
    from app.rag.retrieval import retrieve
    from app.rag.vector_store import NumpyVectorStore
    from app.services.model_service import get_model_service

    settings = get_settings()
    model = get_model_service()
    reranker = build_reranker(reranker_model=settings.reranker_model, model_service=model)

    print(f"\nbackend      : {model.backend_name}")
    print(f"device       : {model.device_info}")
    print(f"retrieval    : top_k={settings.retrieval_top_k} "
          f"rerank_top_k={settings.rerank_top_k} "
          f"min_relevance={settings.min_relevance_score}")
    print(f"chunking     : target={settings.chunk_target_tokens} tok "
          f"overlap={settings.chunk_overlap_tokens} tok")

    # Warm the model once so the first measured sample is not a cold load.
    with timer() as t:
        model.embed(["warmup"])
    print(f"cold embed   : {t.ms:.1f} ms (excluded from all samples below)")

    results: dict = {"backend": model.backend_name, "scale": [], "components": {}}

    store_dir = ROOT / ".bench-index"
    store_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 78)
    print("INGESTION — cost by document size")
    print("=" * 78)
    print(f"  {'paragraphs':>10} {'KB':>8} {'chunks':>7} "
          f"{'chunk ms':>9} {'embed ms':>10} {'index ms':>9} {'chunks/s':>9}")

    indexes: dict[int, tuple] = {}
    for n in sizes:
        data = make_document(n)
        result = extract(extension=".txt", data=data, settings=settings)

        with timer() as t_chunk:
            chunks = chunk_document(
                result.pages,
                document_id=f"bench-{n}",
                target_tokens=settings.chunk_target_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
            )
        texts = [c.text for c in chunks]

        with timer() as t_embed:
            embeddings = model.embed(texts)

        store = NumpyVectorStore(store_dir)
        with timer() as t_index:
            store.add(chunks, embeddings)

        indexes[n] = (store, len(chunks))
        rate = len(chunks) / (t_embed.ms / 1000) if t_embed.ms > 0 else float("inf")
        print(f"  {n:>10} {len(data)/1024:>8.1f} {len(chunks):>7} "
              f"{t_chunk.ms:>9.1f} {t_embed.ms:>10.1f} {t_index.ms:>9.1f} {rate:>9.0f}")
        results["scale"].append({
            "paragraphs": n, "kb": round(len(data) / 1024, 1), "chunks": len(chunks),
            "chunk_ms": round(t_chunk.ms, 2), "embed_ms": round(t_embed.ms, 2),
            "index_ms": round(t_index.ms, 2), "chunks_per_s": round(rate, 1),
        })

    # -- Query path, broken down -------------------------------------------
    print("\n" + "=" * 78)
    print("QUERY — component breakdown, largest index "
          f"({indexes[sizes[-1]][1]} chunks)")
    print("=" * 78)

    store, n_chunks = indexes[sizes[-1]]
    doc_id = f"bench-{sizes[-1]}"

    s_embed = Samples("embed question")
    s_search = Samples("vector search (top_k=8)")
    s_rerank = Samples("rerank (8 -> 4)")
    s_total = Samples("retrieve() total")

    for i in range(queries):
        q = QUESTIONS[i % len(QUESTIONS)]
        with timer(s_embed):
            qe = model.embed([q])[0]
        with timer(s_search):
            cands = store.search(doc_id, qe, top_k=settings.retrieval_top_k)
        with timer(s_rerank):
            reranker.rerank(q, cands, top_k=settings.rerank_top_k)
        with timer(s_total):
            retrieve(document_id=doc_id, question=q, model_service=model,
                     vector_store=store, reranker=reranker, settings=settings)

    for s in (s_embed, s_search, s_rerank, s_total):
        print(s.row())
        results["components"][s.name] = {
            "p50_ms": round(s.pct(50), 3), "p95_ms": round(s.pct(95), 3),
            "n": len(s.values),
        }

    # -- Generation, measured separately ------------------------------------
    print("\n" + "=" * 78)
    print("GENERATION")
    print("=" * 78)
    from app.rag.generation import generate_grounded_answer
    from app.rag.retrieval import retrieve as _retrieve

    r = _retrieve(document_id=doc_id, question=QUESTIONS[0], model_service=model,
                  vector_store=store, reranker=reranker, settings=settings)
    s_gen = Samples("generate grounded answer")
    chars = 0
    for i in range(max(3, queries // 4)):
        with timer(s_gen):
            answer, _ = generate_grounded_answer(
                question=QUESTIONS[i % len(QUESTIONS)], candidates=r.candidates,
                grounding=r.grounding, model_service=model, settings=settings)
        chars += len(answer)
    print(s_gen.row())
    if s_gen.values:
        avg_chars = chars / len(s_gen.values)
        # ~4 chars/token is the standard rough English ratio. Labelled as an
        # estimate because we do not have the tokenizer's count here.
        est_tok = avg_chars / 4
        tps = est_tok / (statistics.median(s_gen.values) / 1000)
        print(f"  answer length: {avg_chars:.0f} chars (~{est_tok:.0f} tokens, estimated)")
        results["generation"] = {
            "p50_ms": round(s_gen.pct(50), 1), "p95_ms": round(s_gen.pct(95), 1),
        }
        # A tokens/s figure for the mock backend is not a slow number or a
        # fast one — it is a meaningless one. MockModelService returns the
        # leading sentences of its input by string slicing; it does not
        # decode, so there is no generation rate to report. Printing
        # "1,126,527 tokens/s" would be exactly the kind of impressive,
        # measured-looking, worthless figure this whole build exists to
        # avoid, so it is suppressed rather than shown with a caveat.
        if model.backend_name == "mock":
            print("  throughput   : not reported — the mock backend does not decode")
        else:
            print(f"  throughput   : ~{tps:.1f} tokens/s at p50 (token count estimated)")
            results["generation"]["est_tokens_per_s"] = round(tps, 1)

    # -- Search scaling -----------------------------------------------------
    print("\n" + "=" * 78)
    print("VECTOR SEARCH — scaling with index size")
    print("=" * 78)
    print(f"  {'chunks':>8} {'p50 us':>9} {'p95 us':>9}")
    results["search_scaling"] = []
    for n in sizes:
        st, cnt = indexes[n]
        qe = model.embed([QUESTIONS[0]])[0]
        s = Samples(f"search {cnt}", "us")
        for _ in range(200):
            with timer() as t:
                st.search(f"bench-{n}", qe, top_k=settings.retrieval_top_k)
            s.add(t.ms * 1000)
        print(f"  {cnt:>8} {s.pct(50):>9.1f} {s.pct(95):>9.1f}")
        results["search_scaling"].append(
            {"chunks": cnt, "p50_us": round(s.pct(50), 1), "p95_us": round(s.pct(95), 1)})

    import shutil
    shutil.rmtree(store_dir, ignore_errors=True)
    return results


# ---------------------------------------------------------------------------
# HTTP benchmark — end to end, including serialization and concurrency
# ---------------------------------------------------------------------------


def _post_json(base: str, path: str, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(base: str, path: str, timeout: int = 60) -> dict:
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read())


def _upload(base: str, name: str, data: bytes) -> dict:
    boundary = "----gargantuabench"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{name}\"\r\nContent-Type: text/plain\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        base + "/api/documents", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def bench_http(base: str, paragraphs: int, queries: int, concurrency: list[int]) -> dict:
    base = base.rstrip("/")
    results: dict = {"base": base}

    s_health = Samples("GET /health")
    for _ in range(50):
        with timer(s_health):
            _get(base, "/health")
    print("\n" + "=" * 78)
    print("HTTP — transport floor")
    print("=" * 78)
    print(s_health.row())
    print("  (this is the overhead every other number below also pays)")

    # -- Ingestion, end to end ---------------------------------------------
    print("\n" + "=" * 78)
    print("INGESTION — end to end, upload to READY")
    print("=" * 78)
    data = make_document(paragraphs, seed=int(time.time()) % 10000)
    t0 = time.perf_counter()
    up = _upload(base, f"bench_{paragraphs}.txt", data)
    upload_ms = (time.perf_counter() - t0) * 1000
    doc_id, job_id = up["document"]["document_id"], up["job_id"]

    stages: list[tuple[str, float]] = []
    last_stage = None
    while True:
        job = _get(base, f"/api/jobs/{job_id}")
        if job["stage"] != last_stage:
            stages.append((job["stage"], (time.perf_counter() - t0) * 1000))
            last_stage = job["stage"]
        if job["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    total_ms = (time.perf_counter() - t0) * 1000

    doc = _get(base, f"/api/documents/{doc_id}")
    m = doc["metrics"] or {}
    print(f"  document     : {len(data)/1024:.1f} KB, {m.get('word_count','?')} words, "
          f"{m.get('page_count','?')} page(s)")
    print(f"  POST returns : {upload_ms:.1f} ms  (the API is async; this is when the")
    print(f"                 caller is unblocked, not when the index is ready)")
    print(f"  ready at     : {total_ms:.1f} ms")
    if total_ms > 0:
        print(f"  throughput   : {(len(data)/1024)/(total_ms/1000):.1f} KB/s")
    print("  stage timeline (ms from upload):")
    for st, ms in stages:
        print(f"    {st:<12} {ms:>9.1f}")
    results["ingestion"] = {
        "kb": round(len(data) / 1024, 1), "upload_return_ms": round(upload_ms, 1),
        "ready_ms": round(total_ms, 1), "stages": [(s, round(v, 1)) for s, v in stages],
    }

    # -- Ask latency --------------------------------------------------------
    print("\n" + "=" * 78)
    print("ASK — end-to-end latency, serial")
    print("=" * 78)
    s_ask = Samples("POST /ask")
    for i in range(queries):
        with timer(s_ask):
            a = _post_json(base, f"/api/documents/{doc_id}/ask",
                           {"question": QUESTIONS[i % len(QUESTIONS)]})
    print(s_ask.row())
    print(f"  last answer  : grounding={a['grounding']} "
          f"relevance={a['relevance_score']:.4f} "
          f"citations={len(a['citations'])} backend={a['model_used']}")
    results["ask_serial"] = {
        "p50_ms": round(s_ask.pct(50), 1), "p95_ms": round(s_ask.pct(95), 1),
        "max_ms": round(max(s_ask.values), 1), "n": len(s_ask.values),
    }

    # -- Throughput under concurrency --------------------------------------
    print("\n" + "=" * 78)
    print("ASK — throughput under concurrency")
    print("=" * 78)
    print(f"  {'workers':>8} {'req':>5} {'wall s':>8} {'req/s':>8} "
          f"{'p50 ms':>9} {'p95 ms':>9}")
    results["ask_concurrent"] = []
    for workers in concurrency:
        n = max(workers * 4, 12)
        lat: list[float] = []

        def one(i: int) -> float:
            t = time.perf_counter()
            _post_json(base, f"/api/documents/{doc_id}/ask",
                       {"question": QUESTIONS[i % len(QUESTIONS)]})
            return (time.perf_counter() - t) * 1000

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            lat = list(ex.map(one, range(n)))
        wall = time.perf_counter() - t0
        s = Samples("c", "ms")
        s.values = lat
        print(f"  {workers:>8} {n:>5} {wall:>8.2f} {n/wall:>8.2f} "
              f"{s.pct(50):>9.1f} {s.pct(95):>9.1f}")
        results["ask_concurrent"].append({
            "workers": workers, "requests": n, "wall_s": round(wall, 2),
            "req_per_s": round(n / wall, 2), "p50_ms": round(s.pct(50), 1),
            "p95_ms": round(s.pct(95), 1),
        })

    # -- Other endpoints ----------------------------------------------------
    print("\n" + "=" * 78)
    print("OTHER ENDPOINTS")
    print("=" * 78)
    s_sum_cold = Samples("POST /summarize (cold)")
    with timer(s_sum_cold):
        _post_json(base, f"/api/documents/{doc_id}/summarize", {"force_refresh": True})
    s_sum_warm = Samples("POST /summarize (cached)")
    for _ in range(5):
        with timer(s_sum_warm):
            _post_json(base, f"/api/documents/{doc_id}/summarize", {"force_refresh": False})
    s_pages = Samples("GET /pages")
    for _ in range(10):
        with timer(s_pages):
            _get(base, f"/api/documents/{doc_id}/pages")
    s_doc = Samples("GET /documents/{id}")
    for _ in range(20):
        with timer(s_doc):
            _get(base, f"/api/documents/{doc_id}")
    for s in (s_sum_cold, s_sum_warm, s_pages, s_doc):
        print(s.row())
    results["endpoints"] = {
        s.name: {"p50_ms": round(s.pct(50), 2)}
        for s in (s_sum_cold, s_sum_warm, s_pages, s_doc)
    }

    return results


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="GARGANTUA latency/throughput benchmark")
    ap.add_argument("--http", metavar="URL",
                    help="benchmark a running server end to end (e.g. http://127.0.0.1:8000)")
    ap.add_argument("--sizes", default="40,200,800,3000",
                    help="document sizes in paragraphs for the in-process scale test")
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--json", metavar="PATH", help="also write results as JSON")
    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")]
    conc = [int(x) for x in args.concurrency.split(",")]

    print("=" * 78)
    print("GARGANTUA — latency and throughput")
    print("=" * 78)
    backend = os.environ.get("MODEL_BACKEND", "auto")
    print(f"MODEL_BACKEND={backend}")
    if backend == "mock":
        print()
        print("  NOTE: the mock backend replaces model inference with a")
        print("  microsecond stand-in. What follows measures the SYSTEM —")
        print("  chunking, search, reranking, HTTP — not the product. It is a")
        print("  floor, not a forecast. Re-run with MODEL_BACKEND=hf on a GPU")
        print("  for numbers that describe what a user experiences.")

    out = {}
    if args.http:
        out = bench_http(args.http, paragraphs=sizes[-1], queries=args.queries,
                         concurrency=conc)
    else:
        out = bench_in_process(sizes, args.queries)

    out["model_backend"] = backend
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
