# Vector Search with PostgreSQL (pgvector) + BGE-M3

A Dockerized system for semantic and text search over tabular data, using PostgreSQL with pgvector, Ollama (BGE-M3 embeddings), FastAPI, and a React frontend.

## Features

- **EAV schema** – Ingest any CSV with any columns; no schema migrations required
- **Four search methods** – Vector (semantic), trigram (fuzzy), ILIKE (pattern), full-text
- **Compare mode** – Run all methods side-by-side on the same query
- **Benchmarking** – Compare latency across search methods

## Prerequisites

- Docker and Docker Compose
- ~2GB RAM for Ollama + BGE-M3

## Quick Start

```bash
# Start all services (postgres, ollama, api, frontend)
docker compose up -d

# Ollama pulls BGE-M3 automatically on first start (~2–5 min for ~1.2GB)

# Generate sample supplier CSV (10k rows)
docker compose run scripts python scripts/generate_csv.py -n 100000 -o data/suppliers.csv

# Ingest into Postgres (embeds via Ollama; ~5–15 min for 10k rows)
docker compose run scripts python scripts/ingest.py --file data/suppliers.csv --name "Suppliers"

# Open the frontend
open http://localhost:5173
```

## Usage

### Frontend

- **Dataset** – Filter by dataset or search across all
- **Method** – Vector (semantic), Trigram (fuzzy), ILIKE, Full-text, or Compare All
- **Search** – Enter a query and see results with similarity scores

### API Endpoints

- `GET /api/datasets` – List datasets
- `GET /api/datasets/{id}/attributes` – List columns for a dataset
- `GET /api/search?q=...&method=vector|trigram|ilike|fulltext&dataset_id=...&limit=20`
- `GET /api/search/compare?q=...&dataset_id=...&limit=10` – All four methods
- `GET /api/stats` – Entity counts, index sizes

### Scripts

```bash
# Generate CSV (default 10k rows to data/suppliers.csv)
docker compose run scripts python scripts/generate_csv.py -n 5000 -o data/my.csv

# Ingest CSV (see "CSV import" section above for full options)
docker compose run scripts python scripts/ingest.py --file data/my.csv --name "My Dataset"

# Run benchmark
docker compose run scripts python scripts/benchmark.py
```

## Benchmark

Compares latency across vector, trigram, ILIKE, and full-text search. Requires **postgres** and **ollama** to be running (the script calls the API for embeddings).

```bash
# Start dependencies (if not already running)
docker compose up -d postgres ollama

# Run benchmark (runs 5 warmup + 5 benchmark runs per query)
docker compose run --rm scripts python scripts/benchmark.py
```

Output: a table with avg/min/max latency per method, printed to the console. Results are also saved to `data/benchmark_results.json`.

### Database export (backup)

Requires postgres to be running (`docker compose up -d postgres`).

**Using the export script (recommended):**

```bash
# Export to data/vectorsearch_backup_TIMESTAMP.sql
./scripts/export_db.sh

# Export to a specific file
./scripts/export_db.sh my_backup.sql
```

**Using Docker directly:**

```bash
# Export to data/ folder (run from project root)
docker compose run --rm --no-deps -v "$(pwd)/data:/data" -e PGPASSWORD=vector postgres \
  pg_dump -h postgres -U vector vectorsearch -f /data/vectorsearch_backup.sql

# File will appear as data/vectorsearch_backup.sql
```

### Database import (restore)

**Using the import script (recommended):**

```bash
# Restore from dump (prompts for confirmation)
./scripts/import_db.sh data/vectorsearch_backup_20260228_120000.sql

# Restore without confirmation (e.g. CI)
./scripts/import_db.sh data/backup.sql -y
```

**Using Docker directly:**

```bash
# Restore (WARNING: drops and replaces existing database)
# Replace backup.sql with your dump filename in data/
docker compose run --rm --no-deps -v "$(pwd)/data:/dump:ro" -e PGPASSWORD=vector postgres sh -c "
  psql -h postgres -U vector -d postgres -c 'DROP DATABASE IF EXISTS vectorsearch;'
  psql -h postgres -U vector -d postgres -c 'CREATE DATABASE vectorsearch;'
  psql -h postgres -U vector -d vectorsearch -f /dump/backup.sql
"
```

### CSV import (ingest data)

Ingest a CSV file into the vector search database. The `scripts` service mounts `./data` and `./scripts`.

```bash
# Place your CSV in data/ then run (creates embeddings via Ollama)
docker compose run scripts python scripts/ingest.py --file data/suppliers.csv --name "Suppliers"

# With options: limit rows, override filename, exclude columns
docker compose run scripts python scripts/ingest.py \
  --file data/my_export.csv \
  --name "My Dataset" \
  --filename "custom_name.csv" \
  --limit 1000 \
  --skip-columns "id,uuid"
```

### Tests

Requires postgres and ollama running. Ingests the fixture CSV once per test session.

```bash
docker compose up -d postgres ollama
# Wait for Ollama to have bge-m3, then:
docker compose run --rm scripts pytest tests/ -v
```

### Re-ingesting after search_text changes

The `build_search_text` format was updated. To benefit from improved vector search, re-ingest your data:

```bash
# Optional: clear existing data
docker compose exec postgres psql -U vector -d vectorsearch -c "TRUNCATE entities, entity_attributes, attributes, datasets CASCADE;"

# Re-generate and ingest
docker compose run scripts python scripts/generate_csv.py -n 10000 -o data/suppliers.csv
docker compose run scripts python scripts/ingest.py --file data/suppliers.csv --name "Suppliers"
```

## Project Structure

```
├── docker-compose.yml
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py
├── frontend/
│   ├── Dockerfile
│   └── src/
├── scripts/
│   ├── generate_csv.py
│   ├── ingest.py
│   └── benchmark.py
├── db/
│   └── init.sql
└── data/
```

## Environment

- `DATABASE_URL` – PostgreSQL connection string (default: `postgresql://vector:vector@postgres:5432/vectorsearch`)
- `OLLAMA_HOST` – Ollama API URL (default: `http://ollama:11434`)
