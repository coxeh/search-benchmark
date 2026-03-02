#!/usr/bin/env python3
"""
Benchmark vector search vs trigram vs ILIKE vs full-text search vs FAISS.
"""

import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
import ollama
import psycopg2
from tabulate import tabulate

EMBEDDING_DIM = 1024

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://vector:vector@localhost:5432/vectorsearch",
)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = "bge-m3"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
WARMUP_RUNS = 2
BENCHMARK_RUNS = 20

TEST_QUERIES = [
    "electronics supplier Texas",
    "ISO 9001 certified",
    "Acme Corp",
    "manufacturing Chicago",
    "chemicals wholesale",
    "industrial metal parts",
    "food beverage distribution",
]


def get_embedding(text: str) -> list[float]:
    if len(text.split()) < 20:
        text = BGE_QUERY_PREFIX + text
    client = ollama.Client(host=OLLAMA_HOST)
    resp = client.embed(model=EMBEDDING_MODEL, input=text)
    return resp["embeddings"][0]


def run_vector(conn, q: str, limit: int = 10) -> tuple[list, float]:
    embedding = get_embedding(q)
    vec_str = "[" + ",".join(str(x) for x in embedding) + "]"
    start = time.perf_counter()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT e.id, 1 - (e.embedding <=> %s::vector) AS score, e.search_text
        FROM entities e
        WHERE e.embedding IS NOT NULL
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
        """,
        (vec_str, vec_str, limit),
    )
    rows = cur.fetchall()
    elapsed = (time.perf_counter() - start) * 1000
    return rows, elapsed


def run_trigram(conn, q: str, limit: int = 10) -> tuple[list, float]:
    cur = conn.cursor()
    cur.execute("SELECT set_limit(0.1)")
    start = time.perf_counter()
    cur.execute(
        """
        SELECT e.id, similarity(e.search_text, %s) AS score, e.search_text
        FROM entities e
        WHERE e.search_text % %s
        ORDER BY similarity(e.search_text, %s) DESC
        LIMIT %s
        """,
        (q, q, q, limit),
    )
    rows = cur.fetchall()
    elapsed = (time.perf_counter() - start) * 1000
    return rows, elapsed


def run_ilike(conn, q: str, limit: int = 10) -> tuple[list, float]:
    pattern = f"%{q}%"
    start = time.perf_counter()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT e.id, 1.0 AS score, e.search_text
        FROM entities e
        WHERE e.search_text ILIKE %s
        LIMIT %s
        """,
        (pattern, limit),
    )
    rows = cur.fetchall()
    elapsed = (time.perf_counter() - start) * 1000
    return rows, elapsed


def run_fulltext(conn, q: str, limit: int = 10) -> tuple[list, float]:
    start = time.perf_counter()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT e.id, ts_rank(e.search_text_tsvector, plainto_tsquery('english', %s)) AS score, e.search_text
        FROM entities e
        WHERE e.search_text_tsvector @@ plainto_tsquery('english', %s)
        ORDER BY score DESC
        LIMIT %s
        """,
        (q, q, limit),
    )
    rows = cur.fetchall()
    elapsed = (time.perf_counter() - start) * 1000
    return rows, elapsed


def build_faiss_index(conn) -> tuple["faiss.Index", list[int]]:
    """Load embeddings from Postgres and build FAISS IndexFlatIP. Returns (index, id_list)."""
    cur = conn.cursor()
    cur.execute("SELECT id, embedding::text FROM entities WHERE embedding IS NOT NULL ORDER BY id")
    rows = cur.fetchall()
    if not rows:
        return None, []
    ids = [r[0] for r in rows]
    vectors = np.array(
        [[float(x) for x in r[1].strip("[]").split(",")] for r in rows],
        dtype=np.float32,
    )
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)
    return index, ids


def make_run_faiss(index: "faiss.Index", id_list: list[int]):
    """Return run_faiss(conn, q, limit) that uses the given FAISS index."""

    def run_faiss(conn, q: str, limit: int = 10) -> tuple[list, float]:
        start = time.perf_counter()
        emb = get_embedding(q)
        query_vec = np.array([emb], dtype=np.float32)
        faiss.normalize_L2(query_vec)
        scores, indices = index.search(query_vec, limit)
        rows = [
            (id_list[i], float(scores[0][j]), "")
            for j, i in enumerate(indices[0])
            if 0 <= i < len(id_list)
        ]
        elapsed = (time.perf_counter() - start) * 1000
        return rows, elapsed

    return run_faiss


def main() -> None:
    conn = psycopg2.connect(DB_URL)

    methods = [
        ("vector", run_vector),
        ("trigram", run_trigram),
        ("ilike", run_ilike),
        ("fulltext", run_fulltext),
    ]

    # Build FAISS index if embeddings exist
    faiss_index, faiss_ids = build_faiss_index(conn)
    if faiss_index is not None and faiss_ids:
        run_faiss = make_run_faiss(faiss_index, faiss_ids)
        methods.append(("faiss", run_faiss))
        print(f"FAISS index loaded: {len(faiss_ids)} vectors")
    else:
        print("Skipping FAISS: no embeddings in database")

    results: dict[str, dict] = {}
    for method_name, fn in methods:
        results[method_name] = {"latencies": [], "counts": [], "sample_results": {}}

    print("Warmup runs...")
    for _ in range(WARMUP_RUNS):
        for method_name, fn in methods:
            try:
                fn(conn, TEST_QUERIES[0], 5)
            except Exception as e:
                print(f"  {method_name} warmup failed: {e}")

    print(f"\nBenchmarking {len(TEST_QUERIES)} queries, {BENCHMARK_RUNS} runs each...\n")

    for query in TEST_QUERIES:
        for method_name, fn in methods:
            latencies = []
            count = 0
            sample = None
            for _ in range(BENCHMARK_RUNS):
                try:
                    rows, elapsed = fn(conn, query, 10)
                    latencies.append(elapsed)
                    count = len(rows)
                    if sample is None and rows:
                        sample = [(r[0], float(r[1]) if r[1] else 0, (r[2] or "")[:80]) for r in rows[:3]]
                except Exception as e:
                    latencies.append(0)
                    sample = str(e)
            results[method_name]["latencies"].extend(latencies)
            results[method_name]["counts"].append(count)
            if query == TEST_QUERIES[0]:
                results[method_name]["sample_results"] = sample

    conn.close()

    # Summary table
    table = []
    for method_name in [m[0] for m in methods]:
        latencies = results[method_name]["latencies"]
        avg_ms = sum(latencies) / len(latencies) if latencies else 0
        table.append([
            method_name,
            f"{avg_ms:.2f}",
            f"{min(latencies):.2f}" if latencies else "-",
            f"{max(latencies):.2f}" if latencies else "-",
        ])

    print(tabulate(table, headers=["Method", "Avg (ms)", "Min (ms)", "Max (ms)"], tablefmt="simple"))
    print()

    # Save results
    out_path = Path("data/benchmark_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save = {
        "queries": TEST_QUERIES,
        "runs_per_query": BENCHMARK_RUNS,
        "methods": {m: {"avg_ms": sum(results[m]["latencies"]) / len(results[m]["latencies"]) if results[m]["latencies"] else 0} for m in results},
        "sample_results": {m: results[m]["sample_results"] for m in results},
    }
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
