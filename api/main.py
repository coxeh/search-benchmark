"""
FastAPI backend for vector search over EAV entities.
"""

import logging
import os
import re
import time
from typing import Any

import asyncpg
import faiss
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Vector Search API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://vector:vector@localhost:5432/vectorsearch",
)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = "bge-m3"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
QUERY_EXPANSION_MODEL = os.environ.get("OLLAMA_QUERY_EXPANSION_MODEL", "llama3.2")
EMBEDDING_DIM = 1024

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)


async def get_db() -> asyncpg.Connection:
    return await asyncpg.connect(DATABASE_URL)


def pool(app: FastAPI):
    async def startup():
        app.state.db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)

    async def shutdown():
        await app.state.db_pool.close()

    return startup, shutdown


@app.on_event("startup")
async def startup():
    app.state.db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    # Build FAISS index for faiss search method
    logger.info("FAISS indexing: starting index build")
    t0 = time.perf_counter()
    try:
        t_fetch = time.perf_counter()
        async with app.state.db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, embedding::text, dataset_id FROM entities WHERE embedding IS NOT NULL ORDER BY id"
            )
        fetch_ms = (time.perf_counter() - t_fetch) * 1000
        logger.info("FAISS indexing: fetched %d rows with embeddings from Postgres in %.1f ms", len(rows), fetch_ms)

        if rows:
            t_parse = time.perf_counter()
            ids = [r["id"] for r in rows]
            app.state.faiss_id_to_dataset = {r["id"]: r["dataset_id"] for r in rows}
            vectors = np.array(
                [[float(x) for x in r["embedding"].strip("[]").split(",")] for r in rows],
                dtype=np.float32,
            )
            parse_ms = (time.perf_counter() - t_parse) * 1000
            logger.info("FAISS indexing: parsed embeddings into shape %s in %.1f ms", vectors.shape, parse_ms)

            t_norm = time.perf_counter()
            faiss.normalize_L2(vectors)
            norm_ms = (time.perf_counter() - t_norm) * 1000
            logger.info("FAISS indexing: L2-normalized vectors in %.1f ms", norm_ms)

            t_build = time.perf_counter()
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
            index.add(vectors)
            build_ms = (time.perf_counter() - t_build) * 1000
            logger.info("FAISS indexing: built IndexFlatIP (dim=%d, ntotal=%d) in %.1f ms", index.d, index.ntotal, build_ms)

            datasets_seen = len(set(app.state.faiss_id_to_dataset.values()))
            logger.info("FAISS indexing: index covers %d datasets", datasets_seen)

            app.state.faiss_index = index
            app.state.faiss_id_list = ids

            total_ms = (time.perf_counter() - t0) * 1000
            logger.info("FAISS indexing: completed in %.1f ms total", total_ms)
        else:
            app.state.faiss_index = None
            app.state.faiss_id_list = []
            app.state.faiss_id_to_dataset = {}
            logger.info("FAISS indexing: no entities with embeddings; FAISS search disabled")
    except Exception as e:
        total_ms = (time.perf_counter() - t0) * 1000
        logger.exception("FAISS indexing: failed after %.1f ms: %s", total_ms, e)
        app.state.faiss_index = None
        app.state.faiss_id_list = []
        app.state.faiss_id_to_dataset = {}


@app.on_event("shutdown")
async def shutdown():
    await app.state.db_pool.close()


async def expand_query(query: str) -> str:
    """Use LLM to expand query with synonyms and related terms. Falls back to original on failure."""
    prompt = f"""Expand this search query with synonyms and related terms. Return only the expanded query as a single line, no other text.

Query: {query}

Expanded query:"""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": QUERY_EXPANSION_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 100},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            expanded = (data.get("response") or "").strip()
            return expanded if expanded else query
    except Exception:
        return query


async def embed_text(text: str, is_query: bool = True) -> list[float]:
    """Embed text. For queries, prepend BGE retrieval instruction."""
    if is_query and len(text.split()) < 20:
        text = BGE_QUERY_PREFIX + text
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{OLLAMA_HOST}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embeddings"][0]


async def fetch_search_texts(pool: asyncpg.Pool, entity_ids: list[int]) -> dict[int, str]:
    """Fetch search_text for entities. Returns {entity_id: search_text}."""
    if not entity_ids:
        return {}
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, COALESCE(search_text, '') AS stext FROM entities WHERE id = ANY($1)",
            entity_ids,
        )
    return {r["id"]: r["stext"] for r in rows}


async def rerank_with_llm(query: str, items: list[tuple[int, str]], top_k: int = 20) -> list[int]:
    """Use LLM to rerank (entity_id, passage) pairs. Returns entity_ids in ranked order."""
    if not items or len(items) == 1:
        return [i[0] for i in items]
    # Truncate passages to stay within context
    max_passage_len = 150
    numbered = [(eid, (text[:max_passage_len] + "..." if len(text) > max_passage_len else text)) for eid, text in items[:50]]
    lines = "\n".join(f"{i+1}. [{eid}] {text}" for i, (eid, text) in enumerate(numbered))
    prompt = f"""Query: {query}

Rank these passages by relevance to the query. Return ONLY the entity IDs in brackets, most relevant first, comma-separated. Example: [42], [7], [3]

Passages:
{lines}

Ranked IDs:"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": QUERY_EXPANSION_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 200},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            raw = (data.get("response") or "").strip()
        # Parse [id], [id], ... from response
        valid_ids = {eid for eid, _ in numbered}
        found = []
        for m in re.findall(r"\[(\d+)\]", raw):
            try:
                eid = int(m)
                if eid in valid_ids:
                    found.append(eid)
            except ValueError:
                pass
        # Preserve order, dedupe, limit to top_k
        seen = set()
        ordered = []
        for eid in found:
            if eid not in seen and eid in valid_ids:
                seen.add(eid)
                ordered.append(eid)
                if len(ordered) >= top_k:
                    break
        # Append any missing (LLM may have skipped some)
        for eid, _ in numbered:
            if eid not in seen:
                ordered.append(eid)
        return ordered[:top_k]
    except Exception:
        return [i[0] for i in items[:top_k]]


async def generate_search_text_from_attributes(attributes: dict[str, Any]) -> str:
    """Use LLM to generate searchable text from attribute key-value pairs."""
    if not attributes:
        return ""
    lines = [f"{k}: {v}" for k, v in attributes.items() if v is not None and str(v).strip()]
    if not lines:
        return ""
    attrs_text = "\n".join(lines)
    prompt = f"""Convert these attribute key-value pairs into a single searchable text block (as used for semantic search embeddings). Be concise. Output only the text, no labels or preamble.

{attrs_text}

Searchable text:"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": QUERY_EXPANSION_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 500},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("response") or "").strip()
            return text if text else attrs_text.replace("\n", ". ")
    except Exception:
        return attrs_text.replace("\n", ". ")


async def fetch_attributes(pool: asyncpg.Pool, entity_ids: list[int]) -> dict[int, dict[str, Any]]:
    """Reconstruct attribute dict for each entity from EAV tables."""
    if not entity_ids:
        return {}
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT ea.entity_id, a.name, ea.value
            FROM entity_attributes ea
            JOIN attributes a ON a.id = ea.attribute_id
            WHERE ea.entity_id = ANY($1)
            ORDER BY a.column_order, a.name
            """,
            entity_ids,
        )
    result: dict[int, dict[str, Any]] = {eid: {} for eid in entity_ids}
    for entity_id, name, value in rows:
        result[entity_id][name] = value
    return result


@app.get("/api/datasets")
async def list_datasets():
    async with app.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT d.id, d.name, d.filename, d.created_at,
                   COUNT(e.id) AS row_count
            FROM datasets d
            LEFT JOIN entities e ON e.dataset_id = d.id
            GROUP BY d.id, d.name, d.filename, d.created_at
            ORDER BY d.created_at DESC
            """
        )
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "filename": r["filename"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "row_count": r["row_count"],
        }
        for r in rows
    ]


class GenerateSearchTextRequest(BaseModel):
    attributes: dict[str, Any]


@app.post("/api/search-text/generate")
async def post_generate_search_text(body: GenerateSearchTextRequest):
    """Generate searchable text from attributes using LLM (fallback when entity search_text is missing)."""
    text = await generate_search_text_from_attributes(body.attributes)
    return {"search_text": text}


@app.get("/api/entities/{entity_id}/search-text")
async def get_entity_search_text(entity_id: int):
    """Return the search_text used for embedding this entity. If missing, generates from attributes via LLM."""
    pool = app.state.db_pool
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT search_text FROM entities WHERE id = $1",
            entity_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Entity not found")
    text = row["search_text"]
    if text and str(text).strip():
        return {"search_text": str(text).strip()}
    # search_text missing: fetch attributes and generate via LLM
    attrs = await fetch_attributes(pool, [entity_id])
    generated = await generate_search_text_from_attributes(attrs.get(entity_id, {}))
    return {"search_text": generated}


@app.get("/api/datasets/{dataset_id}/attributes")
async def get_dataset_attributes(dataset_id: int):
    async with app.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, name, display_name, column_order FROM attributes WHERE dataset_id = $1 ORDER BY column_order",
            dataset_id,
        )
    if not rows:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return [{"id": r["id"], "name": r["name"], "display_name": r["display_name"], "column_order": r["column_order"]} for r in rows]


class SearchResult(BaseModel):
    id: int
    score: float
    attributes: dict[str, Any]
    dataset_name: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
    latency_ms: float
    method: str


RRF_K = 60


def _parse_filters(filters: str) -> list[tuple[str, str]]:
    """Parse filters string 'col:val,col2:val2' into [(col, val), ...]."""
    result = []
    for part in filters.split(","):
        part = part.strip()
        if ":" in part:
            col, _, val = part.partition(":")
            col, val = col.strip(), val.strip()
            if col and val:
                result.append((col, val))
    return result


def _filter_clause(filters: list[tuple[str, str]], param_offset: int) -> tuple[str, list[str]]:
    """Build SQL AND clause for attribute filters. param_offset = next $N to use."""
    if not filters:
        return "", []
    clauses = []
    params = []
    for i, (attr_name, attr_val) in enumerate(filters):
        n, m = param_offset + i * 2 + 1, param_offset + i * 2 + 2
        clauses.append(
            f"""EXISTS (
                SELECT 1 FROM entity_attributes ea
                JOIN attributes a ON a.id = ea.attribute_id AND a.dataset_id = e.dataset_id
                WHERE ea.entity_id = e.id AND a.name = ${n} AND ea.value = ${m}
            )"""
        )
        params.extend([attr_name, attr_val])
    return " AND " + " AND ".join(clauses), params


def _rrf_merge(vector_ids: list[int], fulltext_ids: list[int], limit: int) -> list[int]:
    """Merge two ranked lists using Reciprocal Rank Fusion."""
    rrf_scores: dict[int, float] = {}
    for rank, eid in enumerate(vector_ids, 1):
        rrf_scores[eid] = rrf_scores.get(eid, 0) + 1 / (RRF_K + rank)
    for rank, eid in enumerate(fulltext_ids, 1):
        rrf_scores[eid] = rrf_scores.get(eid, 0) + 1 / (RRF_K + rank)
    return sorted(rrf_scores, key=lambda e: -rrf_scores[e])[:limit]


@app.get("/api/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1),
    method: str = Query("vector", pattern="^(vector|trigram|ilike|fulltext|hybrid|faiss)$"),
    dataset_id: int | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    expand: bool = Query(False, description="Expand query with LLM synonyms before search"),
    rerank: bool = Query(False, description="Rerank top results with LLM for better precision"),
    filters: str = Query("", description="Attribute filters, e.g. country:US,city:London"),
):
    start = time.perf_counter()
    pool = app.state.db_pool
    filter_list = _parse_filters(filters)

    logger.info("Search: method=%s q=%r dataset_id=%s limit=%d", method, q[:80] + "..." if len(q) > 80 else q, dataset_id, limit)

    # Optional query expansion for vector/hybrid/faiss
    search_q = (await expand_query(q)) if expand and method in ("vector", "hybrid", "faiss") else q

    def _apply_filters(sql: str, params: list, param_offset: int) -> tuple[str, list]:
        if not filter_list:
            return sql, params
        filter_sql, filter_params = _filter_clause(filter_list, param_offset)
        # Insert filter into WHERE clause (before ORDER BY or LIMIT)
        if " ORDER BY " in sql:
            sql = sql.replace(" ORDER BY ", filter_sql + " ORDER BY ", 1)
        elif " LIMIT " in sql:
            sql = sql.replace(" LIMIT ", filter_sql + " LIMIT ", 1)
        return sql, params + filter_params

    if method == "hybrid":
        import asyncio

        async def run_vector():
            emb = await embed_text(search_q)
            vec_str = "[" + ",".join(str(x) for x in emb) + "]"
            if dataset_id is not None:
                sql = "SELECT e.id FROM entities e WHERE e.embedding IS NOT NULL AND e.dataset_id = $1 ORDER BY e.embedding <=> $2::vector LIMIT 50"
                params = [dataset_id, vec_str]
                sql, params = _apply_filters(sql, params, 3)
            else:
                sql = "SELECT e.id FROM entities e WHERE e.embedding IS NOT NULL ORDER BY e.embedding <=> $1::vector LIMIT 50"
                params = [vec_str]
                sql, params = _apply_filters(sql, params, 2)
            async with pool.acquire() as conn:
                await conn.execute("SET hnsw.ef_search = 100")
                rows = await conn.fetch(sql, *params)
            return [r["id"] for r in rows]

        async def run_fulltext():
            if dataset_id is not None:
                sql = "SELECT e.id FROM entities e WHERE e.search_text_tsvector @@ plainto_tsquery('english', $2) AND e.dataset_id = $1 ORDER BY ts_rank(e.search_text_tsvector, plainto_tsquery('english', $2)) DESC LIMIT 50"
                params = [dataset_id, search_q]
                sql, params = _apply_filters(sql, params, 3)
            else:
                sql = "SELECT e.id FROM entities e WHERE e.search_text_tsvector @@ plainto_tsquery('english', $1) ORDER BY ts_rank(e.search_text_tsvector, plainto_tsquery('english', $1)) DESC LIMIT 50"
                params = [search_q]
                sql, params = _apply_filters(sql, params, 2)
            async with pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
            return [r["id"] for r in rows]

        vector_ids, fulltext_ids = await asyncio.gather(run_vector(), run_fulltext())
        merged_ids = _rrf_merge(vector_ids, fulltext_ids, limit)
        if not merged_ids:
            return SearchResponse(results=[], latency_ms=0, method="hybrid")

        if rerank and len(merged_ids) > 1:
            search_texts = await fetch_search_texts(pool, merged_ids)
            items = [(eid, search_texts.get(eid, "")) for eid in merged_ids]
            merged_ids = await rerank_with_llm(search_q, items, top_k=limit)

        attrs_map = await fetch_attributes(pool, merged_ids)
        rrf_scores: dict[int, float] = {}
        for rank, eid in enumerate(vector_ids, 1):
            rrf_scores[eid] = rrf_scores.get(eid, 0) + 1 / (RRF_K + rank)
        for rank, eid in enumerate(fulltext_ids, 1):
            rrf_scores[eid] = rrf_scores.get(eid, 0) + 1 / (RRF_K + rank)

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT e.id, d.name AS dataset_name FROM entities e LEFT JOIN datasets d ON d.id = e.dataset_id WHERE e.id = ANY($1)",
                merged_ids,
            )
        id_to_dataset = {r["id"]: r["dataset_name"] for r in rows}

        results = [
            SearchResult(
                id=eid,
                score=round(rrf_scores.get(eid, 0), 4),
                attributes=attrs_map.get(eid, {}),
                dataset_name=id_to_dataset.get(eid),
            )
            for eid in merged_ids
        ]
        latency_ms = (time.perf_counter() - start) * 1000
        return SearchResponse(results=results, latency_ms=round(latency_ms, 2), method="hybrid")

    if method == "faiss":
        faiss_index = getattr(app.state, "faiss_index", None)
        faiss_id_list = getattr(app.state, "faiss_id_list", [])
        faiss_id_to_dataset = getattr(app.state, "faiss_id_to_dataset", {})
        if faiss_index is None or not faiss_id_list:
            return SearchResponse(
                results=[],
                latency_ms=0,
                method="faiss",
            )
        try:
            embedding = await embed_text(search_q)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Embedding failed: {e}")
        query_vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)
        k = min(limit * 4, len(faiss_id_list))  # fetch extra for dataset filtering
        scores_arr, indices_arr = faiss_index.search(query_vec, k)
        rows_data = []
        for j, idx in enumerate(indices_arr[0]):
            if idx < 0 or idx >= len(faiss_id_list):
                continue
            eid = faiss_id_list[idx]
            if dataset_id is not None and faiss_id_to_dataset.get(eid) != dataset_id:
                continue
            score = float(scores_arr[0][j])
            rows_data.append((eid, score))
            if len(rows_data) >= limit:
                break
        entity_ids = [r[0] for r in rows_data]
        if not entity_ids:
            return SearchResponse(results=[], latency_ms=(time.perf_counter() - start) * 1000, method="faiss")
        attrs_map = await fetch_attributes(pool, entity_ids)
        async with pool.acquire() as conn:
            ds_rows = await conn.fetch(
                "SELECT e.id, d.name AS dataset_name FROM entities e LEFT JOIN datasets d ON d.id = e.dataset_id WHERE e.id = ANY($1)",
                entity_ids,
            )
        id_to_dataset = {r["id"]: r["dataset_name"] for r in ds_rows}
        results = [
            SearchResult(
                id=eid,
                score=round(score, 4),
                attributes=attrs_map.get(eid, {}),
                dataset_name=id_to_dataset.get(eid),
            )
            for (eid, score) in rows_data
        ]
        if rerank and len(results) > 1:
            search_texts = await fetch_search_texts(pool, entity_ids)
            items = [(r.id, search_texts.get(r.id, "")) for r in results]
            reranked_ids = await rerank_with_llm(search_q, items, top_k=limit)
            id_to_result = {r.id: r for r in results}
            results = [id_to_result[eid] for eid in reranked_ids if eid in id_to_result]
        latency_ms = (time.perf_counter() - start) * 1000
        return SearchResponse(results=results, latency_ms=round(latency_ms, 2), method="faiss")

    if method == "vector":
        try:
            embedding = await embed_text(search_q)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Embedding failed: {e}")
        vec_str = "[" + ",".join(str(x) for x in embedding) + "]"
        if dataset_id is not None:
            sql = """
                SELECT e.id, 1 - (e.embedding <=> $2::vector) AS score, d.name AS dataset_name
                FROM entities e LEFT JOIN datasets d ON d.id = e.dataset_id
                WHERE e.embedding IS NOT NULL AND e.dataset_id = $1
                ORDER BY e.embedding <=> $2::vector LIMIT $3
            """
            params = [dataset_id, vec_str, limit]
            sql, params = _apply_filters(sql, params, 4)
        else:
            sql = """
                SELECT e.id, 1 - (e.embedding <=> $1::vector) AS score, d.name AS dataset_name
                FROM entities e LEFT JOIN datasets d ON d.id = e.dataset_id
                WHERE e.embedding IS NOT NULL
                ORDER BY e.embedding <=> $1::vector LIMIT $2
            """
            params = [vec_str, limit]
            sql, params = _apply_filters(sql, params, 3)
        async with pool.acquire() as conn:
            if method == "vector":
                await conn.execute("SET hnsw.ef_search = 100")
            rows = await conn.fetch(sql, *params)
        if rerank and rows:
            entity_ids_pre = [r["id"] for r in rows]
            search_texts = await fetch_search_texts(pool, entity_ids_pre)
            items = [(eid, search_texts.get(eid, "")) for eid in entity_ids_pre]
            reranked_ids = await rerank_with_llm(search_q, items, top_k=limit)
            id_to_row = {r["id"]: r for r in rows}
            rows = [id_to_row[eid] for eid in reranked_ids if eid in id_to_row]
    elif method == "trigram":
        if dataset_id is not None:
            sql = """
                SELECT e.id, similarity(e.search_text, $2) AS score, d.name AS dataset_name
                FROM entities e LEFT JOIN datasets d ON d.id = e.dataset_id
                WHERE e.search_text % $2 AND e.dataset_id = $1
                ORDER BY similarity(e.search_text, $2) DESC LIMIT $3
            """
            params = [dataset_id, q, limit]
            sql, params = _apply_filters(sql, params, 4)
        else:
            sql = """
                SELECT e.id, similarity(e.search_text, $1) AS score, d.name AS dataset_name
                FROM entities e LEFT JOIN datasets d ON d.id = e.dataset_id
                WHERE e.search_text % $1
                ORDER BY similarity(e.search_text, $1) DESC LIMIT $2
            """
            params = [q, limit]
            sql, params = _apply_filters(sql, params, 3)
        async with pool.acquire() as conn:
            await conn.execute("SELECT set_limit(0.1)")
            rows = await conn.fetch(sql, *params)
    elif method == "ilike":
        pattern = f"%{q}%"
        if dataset_id is not None:
            sql = """
                SELECT e.id, 1.0 AS score, d.name AS dataset_name
                FROM entities e LEFT JOIN datasets d ON d.id = e.dataset_id
                WHERE e.search_text ILIKE $2 AND e.dataset_id = $1 LIMIT $3
            """
            params = [dataset_id, pattern, limit]
            sql, params = _apply_filters(sql, params, 4)
        else:
            sql = """
                SELECT e.id, 1.0 AS score, d.name AS dataset_name
                FROM entities e LEFT JOIN datasets d ON d.id = e.dataset_id
                WHERE e.search_text ILIKE $1 LIMIT $2
            """
            params = [pattern, limit]
            sql, params = _apply_filters(sql, params, 3)
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
    else:  # fulltext
        if dataset_id is not None:
            sql = """
                SELECT e.id, ts_rank(e.search_text_tsvector, plainto_tsquery('english', $2)) AS score, d.name AS dataset_name
                FROM entities e LEFT JOIN datasets d ON d.id = e.dataset_id
                WHERE e.search_text_tsvector @@ plainto_tsquery('english', $2) AND e.dataset_id = $1
                ORDER BY score DESC LIMIT $3
            """
            params = [dataset_id, q, limit]
            sql, params = _apply_filters(sql, params, 4)
        else:
            sql = """
                SELECT e.id, ts_rank(e.search_text_tsvector, plainto_tsquery('english', $1)) AS score, d.name AS dataset_name
                FROM entities e LEFT JOIN datasets d ON d.id = e.dataset_id
                WHERE e.search_text_tsvector @@ plainto_tsquery('english', $1)
                ORDER BY score DESC LIMIT $2
            """
            params = [q, limit]
            sql, params = _apply_filters(sql, params, 3)
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

    entity_ids = [r["id"] for r in rows]
    attrs_map = await fetch_attributes(pool, entity_ids)
    latency_ms = (time.perf_counter() - start) * 1000

    results = [
        SearchResult(
            id=r["id"],
            score=float(r["score"]) if r["score"] is not None else 0.0,
            attributes=attrs_map.get(r["id"], {}),
            dataset_name=r["dataset_name"] if r.get("dataset_name") else None,
        )
        for r in rows
    ]

    return SearchResponse(results=results, latency_ms=round(latency_ms, 2), method=method)


@app.get("/api/search/compare")
async def search_compare(
    q: str = Query(..., min_length=1),
    dataset_id: int | None = Query(None),
    limit: int = Query(10, ge=1, le=50),
    filters: str = Query("", description="Attribute filters, e.g. country:US,city:London"),
):
    """Run all search methods in parallel and return grouped results."""
    import asyncio

    async def run_one(method: str):
        start = time.perf_counter()
        resp = await search(q=q, method=method, dataset_id=dataset_id, limit=limit, filters=filters)
        return {"method": method, "results": resp.results, "latency_ms": resp.latency_ms}

    methods = ["vector", "faiss", "trigram", "ilike", "fulltext", "hybrid"]
    tasks = [run_one(m) for m in methods]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    result = {}
    for i, m in enumerate(methods):
        if isinstance(outcomes[i], Exception):
            result[m] = {"error": str(outcomes[i]), "results": [], "latency_ms": 0}
        else:
            result[m] = {
                "results": [r.model_dump() for r in outcomes[i]["results"]],
                "latency_ms": outcomes[i]["latency_ms"],
            }
    return result


@app.get("/api/stats")
async def get_stats():
    async with app.state.db_pool.acquire() as conn:
        entity_count = await conn.fetchval("SELECT COUNT(*) FROM entities")
        dataset_count = await conn.fetchval("SELECT COUNT(*) FROM datasets")
        with_embedding = await conn.fetchval("SELECT COUNT(*) FROM entities WHERE embedding IS NOT NULL")
        index_sizes = await conn.fetch(
            """
            SELECT indexrelname AS index_name, pg_size_pretty(pg_relation_size(indexrelid)) AS size
            FROM pg_stat_user_indexes
            WHERE schemaname = 'public' AND relname = 'entities'
            """
        )
    return {
        "entity_count": entity_count,
        "dataset_count": dataset_count,
        "entities_with_embedding": with_embedding,
        "index_sizes": [{"name": r["index_name"], "size": r["size"]} for r in index_sizes],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
