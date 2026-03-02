#!/bin/sh
# Export PostgreSQL database to a SQL dump.
# Usage: ./scripts/export_db.sh [output_file]
# Requires: docker compose up -d postgres
# Output: data/vectorsearch_backup_TIMESTAMP.sql (default) or data/<filename>

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_DIR/data"
DEFAULT_FILE="vectorsearch_backup_$(date +%Y%m%d_%H%M%S).sql"
FILE="${1:-$DEFAULT_FILE}"

# Strip data/ prefix if user passed data/backup.sql
FILE="$(echo "$FILE" | sed 's|^data/||')"
OUTPUT="$DATA_DIR/$FILE"

mkdir -p "$DATA_DIR"

echo "Exporting database to $OUTPUT ..."
docker compose -f "$PROJECT_DIR/docker-compose.yml" run --rm --no-deps \
  -v "$DATA_DIR:/data" \
  -e PGPASSWORD=vector \
  postgres pg_dump -h postgres -U vector vectorsearch -f "/data/$FILE"

echo "Done. Size: $(du -h "$OUTPUT" | cut -f1)"
