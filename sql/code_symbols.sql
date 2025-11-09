CREATE TABLE IF NOT EXISTS code_symbols (
    id BIGSERIAL PRIMARY KEY,
    file_path TEXT NOT NULL,
    hash TEXT NOT NULL,
    symbols JSONB,
    imports JSONB,
    exports JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    UNIQUE (file_path, hash)
);

CREATE INDEX IF NOT EXISTS idx_code_symbols_path_hash ON code_symbols (file_path, hash);
