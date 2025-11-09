CREATE TABLE IF NOT EXISTS code_dependencies (
    id BIGSERIAL PRIMARY KEY,
    repo_url TEXT,
    from_file TEXT NOT NULL,
    to_file TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    details_hash TEXT,
    created_at TIMESTAMP DEFAULT now(),
    UNIQUE (repo_url, from_file, to_file, relation_type, details_hash)
);

CREATE INDEX IF NOT EXISTS idx_dep_from ON code_dependencies(from_file);
CREATE INDEX IF NOT EXISTS idx_dep_to ON code_dependencies(to_file);
CREATE INDEX IF NOT EXISTS idx_dep_rel ON code_dependencies(relation_type);
CREATE INDEX IF NOT EXISTS idx_dep_details ON code_dependencies USING GIN (details);
