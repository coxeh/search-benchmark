#!/usr/bin/env python3
"""
Ingest a CSV file into PostgreSQL using the EAV schema.
Auto-discovers columns, builds search_text, embeds via Ollama BGE-M3, and inserts.
"""

import argparse
import os
from pathlib import Path

import ollama
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm

BATCH_SIZE = 32
EMBEDDING_MODEL = "bge-m3"
MAX_SEARCH_TEXT_LEN = 6000


def build_search_text(
    row: dict,
    columns: list[str] | None = None,
    skip_columns: set[str] | None = None,
) -> str:
    """Build search text from any row dict. Uses column order when provided."""
    skip = skip_columns or set()

    def _v(k: str) -> str:
        v = row.get(k)
        return "" if v is None or not str(v).strip() else str(v).strip()

    cols = columns if columns is not None else list(row.keys())
    parts = []
    for col in cols:
        if col in skip:
            continue
        val = _v(col)
        if val:
            parts.append(f"{col.replace('_', ' ').title()}: {val}")
    text = ". ".join(parts)
    if len(text) > MAX_SEARCH_TEXT_LEN:
        text = text[:MAX_SEARCH_TEXT_LEN].rsplit(". ", 1)[0] + "..."
    return text


def get_embedding(text: str, client) -> list[float]:
    """Get embedding from Ollama."""
    resp = client.embed(model=EMBEDDING_MODEL, input=text)
    return resp["embeddings"][0]


def get_embeddings_batch(texts: list[str], client) -> list[list[float]]:
    """Get embeddings for a batch of texts."""
    resp = client.embed(model=EMBEDDING_MODEL, input=texts)
    return [e for e in resp["embeddings"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest CSV into EAV schema with embeddings")
    parser.add_argument("--file", type=Path, required=True, help="Path to CSV file")
    parser.add_argument("--name", type=str, required=True, help="Dataset name")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows (for testing)")
    parser.add_argument(
        "--skip-columns",
        type=str,
        default="",
        help="Comma-separated column names to exclude from search text (e.g. id,uuid)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Override filename stored in dataset (default: use the file's name)",
    )
    args = parser.parse_args()
    skip_columns = {c.strip() for c in args.skip_columns.split(",") if c.strip()}

    db_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://vector:vector@localhost:5432/vectorsearch",
    )
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    client = ollama.Client(host=ollama_host)
    conn = psycopg2.connect(db_url)
    conn.autocommit = False

    try:
        df = pd.read_csv(args.file, dtype=str)
        df = df.fillna("")
        if args.limit:
            df = df.head(args.limit)

        columns = list(df.columns)
        filename = args.filename if args.filename is not None else args.file.name

        # 1. Create dataset
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO datasets (name, filename) VALUES (%s, %s) RETURNING id",
                (args.name, filename),
            )
            dataset_id = cur.fetchone()[0]

        # 2. Create attributes
        attr_id_by_name: dict[str, int] = {}
        with conn.cursor() as cur:
            for i, col in enumerate(columns):
                cur.execute(
                    "INSERT INTO attributes (dataset_id, name, display_name, column_order) VALUES (%s, %s, %s, %s) RETURNING id",
                    (dataset_id, col, col.replace("_", " ").title(), i),
                )
                attr_id_by_name[col] = cur.fetchone()[0]

        # 3. Build search texts and insert entities (without embeddings first)
        entities_data: list[tuple[int, int, str]] = []
        entity_attrs: list[tuple[int, int, str]] = []
        entity_ids: list[int] = []

        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Preparing rows"), start=1):
            row_dict = row.to_dict()
            search_text = build_search_text(row_dict, columns=columns, skip_columns=skip_columns)

            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO entities (dataset_id, source_row, search_text) VALUES (%s, %s, %s) RETURNING id",
                    (dataset_id, i, search_text),
                )
                entity_id = cur.fetchone()[0]
                entity_ids.append(entity_id)
                entities_data.append((entity_id, search_text))

            for col, value in row_dict.items():
                if value is not None and str(value).strip():
                    attr_id = attr_id_by_name[col]
                    entity_attrs.append((entity_id, attr_id, str(value).strip()))

        # 4. Batch embed and update entities
        all_search_texts = [e[1] for e in entities_data]
        all_embeddings: list[list[float]] = []

        for i in tqdm(range(0, len(all_search_texts), BATCH_SIZE), desc="Embedding"):
            batch = all_search_texts[i : i + BATCH_SIZE]
            embs = get_embeddings_batch(batch, client)
            all_embeddings.extend(embs)

        # 5. Update entities with embeddings
        with conn.cursor() as cur:
            for (entity_id, _), emb in zip(entities_data, all_embeddings):
                vec_str = "[" + ",".join(str(x) for x in emb) + "]"
                cur.execute(
                    "UPDATE entities SET embedding = %s::vector WHERE id = %s",
                    (vec_str, entity_id),
                )

        # 6. Insert entity_attributes
        with conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO entity_attributes (entity_id, attribute_id, value) VALUES %s",
                entity_attrs,
            )

        conn.commit()
        with conn.cursor() as cur:
            cur.execute("ANALYZE entities")
            cur.execute("ANALYZE entity_attributes")
        conn.commit()

        print(f"Ingested {len(df)} rows into dataset '{args.name}' (id={dataset_id})")

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


if __name__ == "__main__":
    main()
