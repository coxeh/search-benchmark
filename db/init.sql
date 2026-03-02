CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- A dataset represents one CSV import
CREATE TABLE datasets (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    filename TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- An entity is one row from the CSV
CREATE TABLE entities (
    id SERIAL PRIMARY KEY,
    dataset_id INT REFERENCES datasets(id) ON DELETE CASCADE,
    source_row INT,
    search_text TEXT,
    search_text_tsvector TSVECTOR GENERATED ALWAYS AS
        (to_tsvector('english', COALESCE(search_text, ''))) STORED,
    embedding vector(1024)
);

-- Attributes are the column headers from the CSV
CREATE TABLE attributes (
    id SERIAL PRIMARY KEY,
    dataset_id INT REFERENCES datasets(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    display_name TEXT,
    column_order INT DEFAULT 0,
    UNIQUE(dataset_id, name)
);

-- The actual cell values
CREATE TABLE entity_attributes (
    entity_id INT REFERENCES entities(id) ON DELETE CASCADE,
    attribute_id INT REFERENCES attributes(id) ON DELETE CASCADE,
    value TEXT,
    PRIMARY KEY (entity_id, attribute_id)
);

-- Search indexes on the entities table
-- ef_construction=128 for better recall (trade-off: slower index build)
CREATE INDEX idx_embedding_hnsw ON entities
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128)
    WHERE embedding IS NOT NULL;
CREATE INDEX idx_trgm ON entities USING gin (search_text gin_trgm_ops)
    WHERE search_text IS NOT NULL;
CREATE INDEX idx_fts ON entities USING gin (search_text_tsvector);
CREATE INDEX idx_entity_dataset ON entities (dataset_id);
CREATE INDEX idx_ea_entity ON entity_attributes (entity_id);
CREATE INDEX idx_ea_attribute ON entity_attributes (attribute_id);
