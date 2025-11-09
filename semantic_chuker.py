#!/usr/bin/env python3
"""
Semantic Chunker with pgvector storage.

- Reads symbols from `code_symbols` (assumes you already populate it).
- For each symbol (function/class/module) creates a contextual chunk:
    - content = lines [start_context_before .. end_context_after]
    - metadata: file_path, symbol name, symbol_type, start_line, end_line, repo_url, commit_hash
- Computes embeddings (batch) using sentence-transformers model from config (or no embeddings if not available).
- Stores chunks into Postgres table `code_chunks` with a `vector` column for pgvector.
- Idempotent upsert (unique on repo_url + file_path + start_line + end_line + symbol_name).
"""

import os
import json
import time
import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import psycopg
from psycopg.rows import dict_row
from tqdm import tqdm
import yaml

# Optional embeddings
try:
    from sentence_transformers import SentenceTransformer
    _HAS_EMBED = True
except Exception:
    SentenceTransformer = None
    _HAS_EMBED = False

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("semantic_chunker")

# -----------------------
# Config loader (reuse same config.yml)
# -----------------------
def load_config(path: str = "config.yml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    expanded = os.path.expandvars(raw)
    return yaml.safe_load(expanded)

# -----------------------
# DB (psycopg) helper
# -----------------------
class PgStore:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.conn = None

    def connect(self):
        if not self.conn:
            self.conn = psycopg.connect(
                host=self.cfg.get("host"),
                port=self.cfg.get("port"),
                dbname=self.cfg.get("database"),
                user=self.cfg.get("user"),
                password=self.cfg.get("password"),
                row_factory=dict_row,
            )

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def fetch_symbols(self, repo_local_path: str) -> List[Dict[str, Any]]:
        """
        Fetch all symbol rows for a given repo path, regardless of whether it's stored
        with ./, absolute path, or relative prefix.
        """
        self.connect()
        repo_local = repo_local_path.replace("./", "").replace(".\\", "")
        repo_name = Path(repo_local).name
        like1 = f"%{repo_local}%"
        like2 = f"%{repo_name}%"
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT file_path, hash, symbols, imports, exports
                FROM code_symbols
                WHERE file_path LIKE %s
                   OR file_path LIKE %s
                """,
                (like1, like2),
            )
            rows = cur.fetchall()
        return rows

    def ensure_extension_and_table(self, embedding_dim: Optional[int] = None):
        """
        Create pgvector extension and code_chunks table (idempotent).
        IMPORTANT: pgvector vector column requires a dimension. We default to 384 if not provided.
        """
        self.connect()
        dim = embedding_dim or 384
        with self.conn.cursor() as cur:
            # enable pgvector
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # create table, embedding column typed as vector(dim)
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS code_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    repo_url TEXT,
                    file_path TEXT NOT NULL,
                    symbol_name TEXT,
                    symbol_type TEXT,
                    start_line INT,
                    end_line INT,
                    content TEXT,
                    metadata JSONB DEFAULT '{{}}',
                    commit_hash TEXT,
                    embedding vector({dim}),
                    created_at TIMESTAMP DEFAULT now(),
                    UNIQUE (repo_url, file_path, start_line, end_line, symbol_name)
                );
                """
            )
            # helpful indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file ON code_chunks(file_path);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON code_chunks(symbol_name);")
            self.conn.commit()

    def upsert_chunks(self, repo_url: str, chunks: List[Dict[str, Any]], embedding_dim: Optional[int] = None):
        """
        Upsert a batch of chunk dicts.
        Each chunk must contain: file_path, symbol_name, symbol_type, start_line, end_line, content, metadata, commit_hash, embedding (list or None)
        NOTE: embedding must be stored as string vector literal like '[0.1,0.2,...]' so we cast to ::vector in SQL.
        """
        if not chunks:
            return
        self.connect()
        with self.conn.cursor() as cur:
            # build insert
            sql = f"""
            INSERT INTO code_chunks
            (repo_url, file_path, symbol_name, symbol_type, start_line, end_line, content, metadata, commit_hash, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
            ON CONFLICT (repo_url, file_path, start_line, end_line, symbol_name)
            DO UPDATE SET
                symbol_type = EXCLUDED.symbol_type,
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                commit_hash = EXCLUDED.commit_hash,
                embedding = EXCLUDED.embedding,
                created_at = now();
            """
            args = []
            for c in chunks:
                emb = c.get("embedding")
                if emb is None:
                    emb_sql = None  # pass NULL
                else:
                    # convert to vector literal string "[0.1,0.2,...]"
                    emb_sql = "[" + ",".join(map(lambda x: f"{float(x):.6f}", emb)) + "]"
                args.append(
                    (
                        repo_url,
                        c["file_path"],
                        c.get("symbol_name"),
                        c.get("symbol_type"),
                        c.get("start_line"),
                        c.get("end_line"),
                        c.get("content"),
                        json.dumps(c.get("metadata", {})),
                        c.get("commit_hash"),
                        emb_sql,
                    )
                )
            # executemany will pass None which will be represented as NULL; but note SQL includes ::vector cast — if emb_sql is None this will be "NULL::vector" which is valid.
            cur.executemany(sql, args)
            self.conn.commit()

# -----------------------
# Git helper: get last commit hash for file
# -----------------------
def get_last_commit_hash(repo_path: str, file_path: str) -> Optional[str]:
    """
    Returns last commit sha for a file relative to repo_path or None on error.
    """
    try:
        # We run git -C repo_path log -n 1 --pretty=format:%H -- file_path
        r = subprocess.run(
            ["git", "-C", repo_path, "log", "-n", "1", "--pretty=format:%H", "--", file_path],
            capture_output=True,
            text=True,
            check=False,
        )
        out = (r.stdout or "").strip()
        return out or None
    except Exception:
        return None

# -----------------------
# SemanticChunker
# -----------------------
class SemanticChunker:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        repo_cfg = cfg["repository"]
        self.repo_local = repo_cfg["local_path"]
        self.repo_url = repo_cfg.get("url", "")
        self.db = PgStore(cfg["database"])
        graph_cfg = cfg.get("graph", {}) or {}
        self.batch_size = graph_cfg.get("chunk_batch_size", 128)
        self.context_lines = graph_cfg.get("chunk_context_lines", 3)
        self.model_name = cfg.get("embedding", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.model_device = cfg.get("embedding", {}).get("device", None)
        self.embed_batch_size = cfg.get("embedding", {}).get("batch_size", 64)
        # desired embedding dim (if set in config). If not set we'll use model.get_sentence_embedding_dimension()
        self.embedding_dim_config = cfg.get("graph", {}).get("embedding_dim", None)

        # Initialize embeddings model lazily
        self._model = None
        if _HAS_EMBED:
            try:
                log.info("Loading sentence-transformers model: %s", self.model_name)
                self._model = SentenceTransformer(self.model_name, device=self.model_device) if self.model_device else SentenceTransformer(self.model_name)
                self.model_dim = self._model.get_sentence_embedding_dimension()
                log.info("Loaded embeddings model dim=%d", self.model_dim)
            except Exception as e:
                log.exception("Failed to load sentence-transformers model: %s", e)
                self._model = None
                self.model_dim = None
        else:
            log.warning("sentence-transformers not installed; embeddings will be skipped.")
            self.model_dim = None

    def ensure_db(self):
        # ensure pgvector and table - if model_dim not known fall back to config or 384
        dim = self.model_dim or self.embedding_dim_config or 384
        self.db.ensure_extension_and_table(dim)

    def _make_chunks_from_symbol(self, file_path: str, symbol: Dict[str, Any], file_lines: List[str]) -> Dict[str, Any]:
        """
        Build a chunk dict from a single symbol entry.
        symbol expected keys: name, type, line_start, line_end, signature
        """
        start = max(1, int(symbol.get("line_start", 1)) - self.context_lines)
        end = min(len(file_lines), int(symbol.get("line_end", symbol.get("line_start", 1))) + self.context_lines)
        # join lines (1-indexed)
        content = "\n".join(file_lines[start - 1 : end])
        chunk = {
            "file_path": file_path,
            "symbol_name": symbol.get("name"),
            "symbol_type": symbol.get("type"),
            "start_line": start,
            "end_line": end,
            "content": content,
            "metadata": {
                "signature": symbol.get("signature"),
                "imports_context": None,
            },
            "commit_hash": None,
            "embedding": None,
        }
        return chunk

    def _module_level_chunk(self, file_path: str, file_lines: List[str]) -> Dict[str, Any]:
        """Create a module-level chunk for file-level docs / content when no symbols exist or to store file context."""
        # cap size to avoid enormous chunks
        max_lines = self.cfg.get("indexing", {}).get("max_chunk_lines", 1000)
        end = min(len(file_lines), max_lines)
        content = "\n".join(file_lines[:end])
        return {
            "file_path": file_path,
            "symbol_name": None,
            "symbol_type": "module",
            "start_line": 1,
            "end_line": end,
            "content": content,
            "metadata": {"module": True},
            "commit_hash": None,
            "embedding": None,
        }

    def _attach_imports_context(self, chunk: Dict[str, Any], imports: List[str]):
        # attach imports snippet if available (first few)
        if imports:
            chunk["metadata"]["imports_context"] = imports[:10]

    def _compute_embeddings(self, chunks: List[Dict[str, Any]]):
        """
        Compute embeddings in batches for chunks which don't have embeddings yet and model exists.
        Modifies chunks in-place setting 'embedding' key to List[float] or None.
        """
        if not self._model:
            for c in chunks:
                c["embedding"] = None
            return

        texts = [c["content"] for c in chunks]
        # batch
        out_embs = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch_texts = texts[i : i + self.embed_batch_size]
            arr = self._model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
            for vec in arr:
                out_embs.append(vec.tolist())
        # set
        for c, v in zip(chunks, out_embs):
            c["embedding"] = v

    def run(self, dry_run: bool = False):
        """
        Main flow:
            - ensure DB table exists
            - fetch all code_symbols rows under repo_local
            - for each file build chunks from symbols or module-level
            - compute embeddings and upsert to DB in batches
        """
        log.info("Starting semantic chunking for repo %s", self.repo_local)
        self.ensure_db()
        # fetch records
        self.db.connect()
        rows = self.db.fetch_symbols(self.repo_local)
        log.info("Found %d files in code_symbols", len(rows))

        chunks_to_upsert: List[Dict[str, Any]] = []
        for r in tqdm(rows, desc="Building chunks"):
            file_path = r["file_path"]
            # read file content safely
            try:
                text = Path(file_path).read_text(encoding="utf-8", errors="replace")
                lines = text.splitlines()
            except Exception as e:
                log.warning("Cannot read file %s: %s", file_path, e)
                continue

            symbols = r.get("symbols") or []
            imports = r.get("imports") or []
            # each symbol -> chunk
            if symbols:
                for sym in symbols:
                    # ensure numeric line fields
                    if not sym.get("line_start") or not sym.get("line_end"):
                        # skip or attempt to parse from signature fallback
                        continue
                    chunk = self._make_chunks_from_symbol(file_path, sym, lines)
                    self._attach_imports_context(chunk, imports)
                    # commit hash
                    chunk["commit_hash"] = get_last_commit_hash(self.repo_local, file_path.replace(self.repo_local + os.sep, ""))
                    chunks_to_upsert.append(chunk)
                    # flush by batch
                    if len(chunks_to_upsert) >= self.batch_size:
                        self._flush_batch(chunks_to_upsert, dry_run)
                        chunks_to_upsert = []
            else:
                # module-level chunk
                chunk = self._module_level_chunk(file_path, lines)
                self._attach_imports_context(chunk, imports)
                chunk["commit_hash"] = get_last_commit_hash(self.repo_local, file_path.replace(self.repo_local + os.sep, ""))
                chunks_to_upsert.append(chunk)
                if len(chunks_to_upsert) >= self.batch_size:
                    self._flush_batch(chunks_to_upsert, dry_run)
                    chunks_to_upsert = []

        if chunks_to_upsert:
            self._flush_batch(chunks_to_upsert, dry_run)

        log.info("Semantic chunking complete.")
        self.db.close()

    def _flush_batch(self, chunks: List[Dict[str, Any]], dry_run: bool):
        # compute embeddings
        if self._model:
            self._compute_embeddings(chunks)
            # validate embedding dim
            if self.model_dim and self.embedding_dim_config and int(self.embedding_dim_config) != int(self.model_dim):
                log.warning("Configured graph.embedding_dim=%s differs from model dim=%s", self.embedding_dim_config, self.model_dim)
        else:
            for c in chunks:
                c["embedding"] = None

        if dry_run:
            log.info("DRY RUN: prepared %d chunks (not inserting)", len(chunks))
            return

        # upsert
        try:
            self.db.upsert_chunks(self.repo_url, chunks, embedding_dim=self.model_dim or self.embedding_dim_config)
            log.info("Inserted %d chunks", len(chunks))
        except Exception as e:
            log.exception("Failed to insert chunk batch: %s", e)

# -----------------------
# CLI
# -----------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Semantic Chunker - create chunks and store embeddings in pgvector")
    parser.add_argument("--config", "-c", default="config.yml", help="Path to config.yml")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sc = SemanticChunker(cfg)
    sc.run(dry_run=args.dry_run)

if __name__ == "__main__":
    main()
