#!/bin/sh
# Import PostgreSQL database from a SQL dump.
# Run on host: ./scripts/import_db.sh [dump_file] [-y]
#
# WARNING: This drops and recreates the database. Existing data will be lost.
# Use -y to skip confirmation (for non-interactive/CI use).

set -e

SKIP_CONFIRM=false
DUMP_ARG=""
for arg in "$@"; do
  case "$arg" in
    -y|--yes) SKIP_CONFIRM=true ;;
    *) DUMP_ARG="$arg" ;;
  esac
done

if [ -z "$DUMP_ARG" ]; then
  echo "Usage: $0 <dump_file> [-y]"
  echo "Example: $0 data/vectorsearch_backup_20260228_120000.sql"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_DIR/data"

# Resolve dump file path
case "$DUMP_ARG" in
  /*) DUMP_PATH="$DUMP_ARG" ;;
  *) DUMP_PATH="$PROJECT_DIR/$DUMP_ARG" ;;
esac

# Get filename for mount (dump must be under project/data or we need full path)
DUMP_FILE="$(basename "$DUMP_PATH")"
DUMP_DIR="$(dirname "$DUMP_PATH")"

if [ ! -f "$DUMP_PATH" ]; then
  echo "Error: File not found: $DUMP_PATH"
  exit 1
fi

if [ "$SKIP_CONFIRM" = false ]; then
  echo "Importing $DUMP_ARG into database..."
  echo "WARNING: This will replace existing data."
  printf "Continue? [y/N] "
  read -r REPLY
  case "$REPLY" in
    [yY]|[yY][eE][sS]) ;;
    *) echo "Aborted."; exit 0 ;;
  esac
fi

echo "Importing..."
docker compose -f "$PROJECT_DIR/docker-compose.yml" run --rm --no-deps \
  -v "$DUMP_DIR:/dump:ro" \
  -e PGPASSWORD=vector \
  postgres sh -c "
    psql -h postgres -U vector -d postgres -c \"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'vectorsearch' AND pid <> pg_backend_pid();\" 2>/dev/null || true
    psql -h postgres -U vector -d postgres -c 'DROP DATABASE IF EXISTS vectorsearch;'
    psql -h postgres -U vector -d postgres -c 'CREATE DATABASE vectorsearch;'
    psql -h postgres -U vector -d vectorsearch -f \"/dump/$DUMP_FILE\"
  "

echo "Done."
