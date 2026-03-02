"""Search quality tests - validates that vector, fulltext, and hybrid return correct matches."""

import os

import ollama
import psycopg2
import pytest

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBEDDING_MODEL = "bge-m3"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _get_embedding(text: str, use_query_prefix: bool = True) -> list[float]:
    """Get embedding, optionally with BGE query prefix."""
    if use_query_prefix and len(text.split()) < 20:
        text = BGE_QUERY_PREFIX + text
    client = ollama.Client(host=OLLAMA_HOST)
    resp = client.embed(model=EMBEDDING_MODEL, input=text)
    return resp["embeddings"][0]


def _run_vector_search(conn, query: str, dataset_id: int, limit: int = 10, use_prefix: bool = True):
    """Run vector search, return list of (entity_id, supplier_id, score)."""
    embedding = _get_embedding(query, use_query_prefix=use_prefix)
    vec_str = "[" + ",".join(str(x) for x in embedding) + "]"
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.id, 1 - (e.embedding <=> %s::vector) AS score
            FROM entities e
            WHERE e.embedding IS NOT NULL AND e.dataset_id = %s
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
            """,
            (vec_str, dataset_id, vec_str, limit),
        )
        rows = cur.fetchall()
    # Fetch supplier_id for each entity
    result = []
    with conn.cursor() as cur:
        for entity_id, score in rows:
            cur.execute(
                """
                SELECT ea.value FROM entity_attributes ea
                JOIN attributes a ON a.id = ea.attribute_id
                WHERE ea.entity_id = %s AND a.name = 'supplier_id'
                """,
                (entity_id,),
            )
            row = cur.fetchone()
            supplier_id = row[0] if row else None
            result.append((entity_id, supplier_id, float(score)))
    return result


def _run_fulltext_search(conn, query: str, dataset_id: int, limit: int = 10):
    """Run full-text search, return list of (entity_id, supplier_id, score)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.id, ts_rank(e.search_text_tsvector, plainto_tsquery('english', %s)) AS score
            FROM entities e
            WHERE e.search_text_tsvector @@ plainto_tsquery('english', %s) AND e.dataset_id = %s
            ORDER BY score DESC
            LIMIT %s
            """,
            (query, query, dataset_id, limit),
        )
        rows = cur.fetchall()
    result = []
    with conn.cursor() as cur:
        for entity_id, score in rows:
            cur.execute(
                """
                SELECT ea.value FROM entity_attributes ea
                JOIN attributes a ON a.id = ea.attribute_id
                WHERE ea.entity_id = %s AND a.name = 'supplier_id'
                """,
                (entity_id,),
            )
            row = cur.fetchone()
            supplier_id = row[0] if row else None
            result.append((entity_id, supplier_id, float(score or 0)))
    return result


def _run_hybrid_search(conn, query: str, dataset_id: int, limit: int = 10, k: int = 60):
    """RRF hybrid: fuse vector + fulltext scores."""
    vector_results = _run_vector_search(conn, query, dataset_id, limit=50, use_prefix=True)
    fulltext_results = _run_fulltext_search(conn, query, dataset_id, limit=50)
    rrf_scores = {}  # entity_id -> (supplier_id, rrf_score)
    for rank, (entity_id, supplier_id, _) in enumerate(vector_results, 1):
        if entity_id not in rrf_scores:
            rrf_scores[entity_id] = [supplier_id, 0.0]
        rrf_scores[entity_id][1] += 1 / (k + rank)
    for rank, (entity_id, supplier_id, _) in enumerate(fulltext_results, 1):
        if entity_id not in rrf_scores:
            rrf_scores[entity_id] = [supplier_id, 0.0]
        rrf_scores[entity_id][1] += 1 / (k + rank)
    sorted_items = sorted(rrf_scores.items(), key=lambda x: -x[1][1])[:limit]
    return [(eid, sid, score) for eid, (sid, score) in sorted_items]


class TestFulltextSearch:
    """Full-text search should find exact matches."""

    def test_fulltext_finds_exact_supplier(self, db_conn, ingested_dataset_id):
        """Query 'Proctor Harris 7257' returns SUP-73405 in top 5."""
        results = _run_fulltext_search(db_conn, "Proctor Harris 7257", ingested_dataset_id, limit=5)
        supplier_ids = [r[1] for r in results]
        assert "SUP-73405" in supplier_ids, f"Expected SUP-73405 in {supplier_ids}"


class TestVectorSearch:
    """Vector search quality tests."""

    def test_vector_finds_exact_supplier(self, db_conn, ingested_dataset_id):
        """Query 'Proctor Harris 7257' returns SUP-73405 in top 10."""
        results = _run_vector_search(db_conn, "Proctor Harris 7257", ingested_dataset_id, limit=10)
        supplier_ids = [r[1] for r in results]
        assert "SUP-73405" in supplier_ids, f"Expected SUP-73405 in {supplier_ids}"

    def test_vector_finds_semantic(self, db_conn, ingested_dataset_id):
        """'electronics supplier Texas' returns at least one Electronics/TX supplier."""
        results = _run_vector_search(db_conn, "electronics supplier Texas", ingested_dataset_id, limit=5)
        assert len(results) > 0
        # SUP-10001 Acme Electronics is in Austin TX
        supplier_ids = [r[1] for r in results]
        assert "SUP-10001" in supplier_ids, f"Expected electronics/TX supplier in {supplier_ids}"


class TestHybridSearch:
    """Hybrid search should combine vector + fulltext for best recall."""

    def test_hybrid_ranks_correctly(self, db_conn, ingested_dataset_id):
        """Hybrid search returns SUP-73405 for 'Proctor Harris, 7257' in top 3."""
        results = _run_hybrid_search(db_conn, "Proctor Harris, 7257", ingested_dataset_id, limit=5)
        supplier_ids = [r[1] for r in results]
        assert "SUP-73405" in supplier_ids[:3], f"Expected SUP-73405 in top 3, got {supplier_ids}"


class TestSearchTextFormat:
    """Search text format validation."""

    def test_search_text_format(self):
        """build_search_text includes supplier_name and address_line_1 prominently."""
        from scripts.ingest import build_search_text

        row = {
            "supplier_id": "SUP-73405",
            "supplier_name": "Proctor Harris",
            "address_line_1": "7257 Oak Avenue",
            "city": "Dallas",
        }
        text = build_search_text(row)
        assert "Proctor Harris" in text
        assert "7257" in text
        assert "SUP-73405" in text
        assert "Oak Avenue" in text
