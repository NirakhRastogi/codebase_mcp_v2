-- Enable pgvector extension if not already
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS code_files (
    id BIGSERIAL PRIMARY KEY,
    repo_url TEXT NOT NULL,
    path TEXT NOT NULL,
    commit_hash TEXT,
    author_name TEXT,
    author_email TEXT,
    commit_date TIMESTAMP WITH TIME ZONE,
    size_bytes BIGINT,
    blame_summary JSONB,
    content TEXT,
    embedding VECTOR(384),        -- 384 = dimension of your embedding model (all-MiniLM-L6-v2)
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (repo_url, path)
);

CREATE INDEX IF NOT EXISTS idx_code_files_repo_path ON code_files (repo_url, path);
CREATE INDEX IF NOT EXISTS idx_code_files_commit_hash ON code_files (commit_hash);
CREATE INDEX IF NOT EXISTS idx_code_files_updated_at ON code_files (updated_at);
