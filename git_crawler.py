#!/usr/bin/env python3
"""
Compact Git-Aware File System Crawler
- Uses git for discovery (ls-tree / ls-files)
- Extracts last-commit metadata and a small git-blame summary
- Optional embeddings (sentence-transformers)
- Upserts into Postgres (pgvector column assumed if embeddings enabled)
"""

import os
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import yaml
import psycopg
from tqdm import tqdm

# Optional embedding import
try:
    from sentence_transformers import SentenceTransformer
    _HAS_EMBED = True
except Exception as e:
    print(e)
    SentenceTransformer = None
    _HAS_EMBED = False

# -------------------------
# Class: GitRepo
# Responsibilities:
#  - ensure local clone (or fetch)
#  - list tracked files using git (respects .gitignore automatically)
#  - run git commands for metadata (log, blame)
# -------------------------
class GitRepo:
    def __init__(self, repo_url: str, local_path: str, branch: str = "main"):
        self.repo_url = repo_url
        self.local_path = os.path.expanduser(local_path)
        self.branch = branch

    def _run_git(self, args: List[str], check=False) -> subprocess.CompletedProcess:
        cmd = ["git", "-C", self.local_path] + args
        return subprocess.run(cmd, capture_output=True, text=True, check=check)

    def ensure(self):
        if not (Path(self.local_path) / ".git").exists():
            parent = Path(self.local_path).parent
            parent.mkdir(parents=True, exist_ok=True)
            print(f"Cloning {self.repo_url} -> {self.local_path} (branch={self.branch})")
            subprocess.check_call(["git", "clone", "--branch", self.branch, "--depth", "1", self.repo_url, self.local_path])
        else:
            # fetch and reset to branch tip to ensure metadata is available
            print(f"Fetching updates for {self.local_path}")
            r = self._run_git(["fetch", "origin", self.branch])
            if r.returncode != 0:
                print("git fetch failed:", r.stderr.strip())
            # hard reset to remote branch tip (safe for crawler, not for uncommitted work)
            self._run_git(["reset", "--hard", f"origin/{self.branch}"])

    def list_tracked_files(self) -> List[str]:
        # prefer ls-tree at branch tip (works even when repo not checked out)
        r = subprocess.run(["git", "-C", self.local_path, "ls-tree", "-r", "--name-only", self.branch],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"git ls-tree failed: {r.stderr}")
        return [ln for ln in r.stdout.splitlines() if ln.strip()]

    def last_commit_metadata(self, file_path: str) -> Dict[str, Optional[Any]]:
        # git log -1 --pretty=format:%H%x01%an%x01%ae%x01%aI -- <file>
        fmt = "%H%x01%an%x01%ae%x01%aI"
        r = self._run_git(["log", "-1", f"--pretty=format:{fmt}", "--", file_path])
        out = (r.stdout or "").strip()
        if not out:
            return {"commit_hash": None, "author_name": None, "author_email": None, "commit_date": None}
        parts = out.split("\x01")
        commit_hash, author_name, author_email, commit_date = (parts + [None]*4)[:4]
        commit_date_obj = None
        if commit_date:
            try:
                commit_date_obj = datetime.fromisoformat(commit_date)
            except Exception:
                commit_date_obj = None
        return {
            "commit_hash": commit_hash,
            "author_name": author_name,
            "author_email": author_email,
            "commit_date": commit_date_obj,
        }

    def blame_top_authors(self, file_path: str, top_k: int = 3) -> Dict[str, Any]:
        # lightweight blame summary using --line-porcelain
        r = self._run_git(["blame", "--line-porcelain", "HEAD", "--", file_path])
        if r.returncode != 0:
            return {}
        counts: Dict[str, int] = {}
        for line in (r.stdout or "").splitlines():
            if line.startswith("author "):
                author = line[len("author "):].strip()
                counts[author] = counts.get(author, 0) + 1
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return {"top_authors": [{"author": a, "lines": c} for a, c in top]}


# -------------------------
# Class: EmbeddingModel (optional)
# Responsibilities:
#  - load sentence-transformers model (if requested)
#  - produce embeddings as lists (serializable for psycopg)
# -------------------------
class EmbeddingModel:
    def __init__(self, model_name: Optional[str], device: Optional[str] = None):
        self.model = None
        self.dim = None
        if not model_name:
            return
        if not _HAS_EMBED:
            print("Embedding requested but sentence-transformers not installed; skipping embeddings.")
            return
        try:
            # SentenceTransformer will use CPU/GPU as configured by torch
            print(f"Loading embedding model {model_name} ...")
            self.model = SentenceTransformer(model_name)
            self.dim = self.model.get_sentence_embedding_dimension()
            print(f"Loaded model (dim={self.dim})")
        except Exception as e:
            print("Failed to load embedding model:", e)
            self.model = None

    def embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        if not self.model:
            return None
        arr = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return arr.tolist()


# -------------------------
# Class: PostgresStore
# Responsibilities:
#  - connect to Postgres
#  - batch upsert records into code_files table
#  - expects table with columns: repo_url, path, commit_hash, author_name, author_email, commit_date, size_bytes, blame_summary, content, embedding
# -------------------------
class PostgresStore:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.conn = None

    def connect(self):
        self.conn = psycopg.connect(
            host=self.cfg.get("host", "localhost"),
            port=self.cfg.get("port", 5432),
            dbname=self.cfg.get("database"),
            user=self.cfg.get("user"),
            password=self.cfg.get("password"),
        )

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def upsert_batch(self, repo_url: str, records: list[dict], dry_run: bool = False):
        if not records:
            return
        if dry_run:
            print(f"DRY RUN: would upsert {len(records)} records")
            return

        self.connect()

        with self.conn.cursor() as cur:
            for r in records:
                cur.execute(
                    """
                    INSERT INTO public.code_files (
                        repo_url, path, commit_hash, author_name, author_email,
                        commit_date, size_bytes, blame_summary, content, embedding, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (repo_url, path) DO UPDATE SET
                        commit_hash = EXCLUDED.commit_hash,
                        author_name = EXCLUDED.author_name,
                        author_email = EXCLUDED.author_email,
                        commit_date = EXCLUDED.commit_date,
                        size_bytes = EXCLUDED.size_bytes,
                        blame_summary = EXCLUDED.blame_summary,
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        updated_at = now()
                    """,
                    (
                        repo_url,
                        r["path"],
                        r.get("commit_hash"),
                        r.get("author_name"),
                        r.get("author_email"),
                        r.get("commit_date"),
                        r.get("size_bytes"),
                        json.dumps(r.get("blame_summary") or {}),
                        r.get("content"),
                        r.get("embedding"),
                    ),
                )
            self.conn.commit()


# -------------------------
# Class: Crawler
# Responsibilities:
#  - orchestrate discovery -> metadata extraction -> optional embeddings -> DB upsert
#  - configurable batch sizes and size limits
# -------------------------
class Crawler:
    def __init__(self, cfg: Dict[str, Any], dry_run: bool = False, no_embedding: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        self.no_embedding = no_embedding

        repo_cfg = cfg["repository"]
        self.repo = GitRepo(repo_cfg["url"], repo_cfg["local_path"], repo_cfg.get("branch", "main"))
        self.store = PostgresStore(cfg["database"])

        embedding_cfg = cfg.get("embedding", {})
        self.embed_model = None
        if embedding_cfg.get("model") and not no_embedding:
            self.embed_model = EmbeddingModel(embedding_cfg.get("model"), embedding_cfg.get("device"))

        self.batch_size = embedding_cfg.get("batch_size", 32)
        self.max_file_size_kb = cfg.get("indexing", {}).get("max_file_size_kb", 1024)

    def _read_file_content(self, path: str, max_bytes: int = 200 * 1024) -> str:
        full = Path(self.repo.local_path) / path
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                return f.read(max_bytes)
        except Exception:
            return ""

    def run_once(self):
        self.repo.ensure()
        files = self.repo.list_tracked_files()
        print(f"Discovered {len(files)} tracked files")
        batch: List[Dict[str, Any]] = []

        for p in tqdm(files, desc="files"):
            fullp = Path(self.repo.local_path) / p
            try:
                size = fullp.stat().st_size
            except Exception:
                size = 0
            if size > self.max_file_size_kb * 1024:
                # skip very large files
                continue

            meta = self.repo.last_commit_metadata(p)
            blame = self.repo.blame_top_authors(p)
            content = self._read_file_content(p)

            if is_binary_string(content.encode("utf-8", errors="replace")):
                print(f'Skipping binary file: {p}')
                continue

            rec = {
                "path": p,
                "commit_hash": meta.get("commit_hash"),
                "author_name": meta.get("author_name"),
                "author_email": meta.get("author_email"),
                "commit_date": meta.get("commit_date"),
                "size_bytes": size,
                "blame_summary": blame,
                "content": content,
            }
            batch.append(rec)

            if len(batch) >= self.batch_size:
                self._flush_batch(batch)
                batch = []

        if batch:
            self._flush_batch(batch)

        # close DB connection
        self.store.close()

    def _flush_batch(self, batch: List[Dict[str, Any]]):
        # compute embeddings if available
        if self.embed_model and self.embed_model.model:
            texts = [(b["path"] + "\n" + (b.get("content") or "")) for b in batch]
            embs = self.embed_model.embed(texts)
            if embs:
                for b, e in zip(batch, embs):
                    b["embedding"] = e
            else:
                for b in batch:
                    b["embedding"] = None
        else:
            for b in batch:
                b["embedding"] = None

        # upsert to DB
        self.store.upsert_batch(self.repo.repo_url, batch, dry_run=self.dry_run)

def is_binary_string(data: bytes) -> bool:
    return b"\x00" in data
