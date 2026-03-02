"""Pytest fixtures for search quality tests."""

import os
import subprocess
import sys
from pathlib import Path

import psycopg2
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEST_DATASET_NAME = "search_quality_test"
FIXTURE_CSV = PROJECT_ROOT / "tests" / "fixtures" / "suppliers_fixture.csv"


@pytest.fixture(scope="session")
def db_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql://vector:vector@localhost:5432/vectorsearch",
    )


@pytest.fixture(scope="session")
def ingested_dataset_id(db_url: str) -> int:
    """Ingest the fixture CSV and return the dataset ID. Requires postgres + ollama."""
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    try:
        # Remove existing test dataset (CASCADE removes entities, attributes)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM datasets WHERE name = %s", (TEST_DATASET_NAME,))

        # Run ingest subprocess
        result = subprocess.run(
            ["python", "scripts/ingest.py", "--file", str(FIXTURE_CSV), "--name", TEST_DATASET_NAME],
            cwd=PROJECT_ROOT,
            env={**os.environ, "DATABASE_URL": db_url},
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, f"Ingest failed: {result.stderr}"

        with conn.cursor() as cur:
            cur.execute("SELECT id FROM datasets WHERE name = %s", (TEST_DATASET_NAME,))
            row = cur.fetchone()
            assert row, "Dataset not found after ingest"
            return row[0]
    finally:
        conn.close()


@pytest.fixture
def db_conn(db_url: str):
    """Database connection for tests."""
    conn = psycopg2.connect(db_url)
    try:
        yield conn
    finally:
        conn.close()
