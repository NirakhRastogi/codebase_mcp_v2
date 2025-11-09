-- enable pgvector extension (requires postgres extension install)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create chunks table (if not letting code create it)
-- Replace 384 with your embedding dimension if different
CREATE TABLE IF NOT EXISTS code_chunks (
  id BIGSERIAL PRIMARY KEY,
  repo_url TEXT,
  file_path TEXT NOT NULL,
  symbol_name TEXT,
  symbol_type TEXT,
  start_line INT,
  end_line INT,
  content TEXT,
  metadata JSONB DEFAULT '{}',
  commit_hash TEXT,
  embedding vector(384),
  created_at TIMESTAMP DEFAULT now(),
  UNIQUE (repo_url, file_path, start_line, end_line, symbol_name)
);

CREATE INDEX IF NOT EXISTS idx_chunks_file ON code_chunks(file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON code_chunks(symbol_name);
CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON code_chunks USING GIN (metadata);
