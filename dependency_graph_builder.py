#!/usr/bin/env python3
"""
Dependency Graph Builder (final production version)

✔ Uses existing config.yml (repository, database, indexing)
✔ Adds optional 'graph' section for tuning
✔ Stores dependencies in PostgreSQL (no Neo4j needed)
✔ Supports Kotlin, Java, Python, TypeScript, JavaScript, etc.
✔ Multi-threaded, batched, deduplicated, and idempotent
"""

import os
import re
import json
import yaml
import hashlib
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

import psycopg
from psycopg.rows import dict_row
from tqdm import tqdm
from tree_sitter import Language, Parser

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("depgraph")

# ---------------------------
# Config Loader
# ---------------------------
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    expanded = os.path.expandvars(raw)
    return yaml.safe_load(expanded)

# ---------------------------
# Database Layer
# ---------------------------
class GraphStore:
    """Postgres-backed graph edge store."""

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

    def ensure_table(self):
        self.connect()
        with self.conn.cursor() as cur:
            cur.execute(
                """
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
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_dep_from ON code_dependencies(from_file);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_dep_to ON code_dependencies(to_file);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_dep_rel ON code_dependencies(relation_type);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_dep_details ON code_dependencies USING GIN (details);")
            self.conn.commit()

    def upsert_batch(self, repo_url: str, edges: List[Dict[str, Any]]):
        if not edges:
            return
        self.connect()
        with self.conn.cursor() as cur:
            sql = """
                INSERT INTO code_dependencies
                (repo_url, from_file, to_file, relation_type, details, details_hash)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """
            args = []
            for e in edges:
                dh = hashlib.sha256(json.dumps(e.get("details", {}), sort_keys=True).encode()).hexdigest()
                args.append((repo_url, e["from_file"], e["to_file"], e["relation_type"], json.dumps(e.get("details", {})), dh))
            cur.executemany(sql, args)
            self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

# ---------------------------
# Parser Cache
# ---------------------------
class ParserPool:
    """Caches Tree-sitter Language objects."""

    def __init__(self, lib_path: str):
        self.lib_path = lib_path
        self.languages: Dict[str, Language] = {}

    def get_parser(self, lang: str) -> Optional[Parser]:
        if lang not in self.languages:
            try:
                self.languages[lang] = Language(self.lib_path, lang)
            except Exception as e:
                log.warning(f"⚠️ Could not load parser for '{lang}': {e}")
                return None
        parser = Parser()
        parser.set_language(self.languages[lang])
        return parser

# ---------------------------
# Dependency Analyzer
# ---------------------------
class DependencyAnalyzer:
    """Extracts import, inheritance, and call edges."""

    IMPORT_NODES = {"import_statement", "import_declaration", "import_from_statement", "import_header"}
    CLASS_NODES = {"class_declaration", "class_definition", "object_declaration"}
    CALL_NODES = {"call_expression", "method_invocation", "function_call"}

    def __init__(self, repo_files: List[str], parser_pool: ParserPool):
        self.repo_files = repo_files
        self.index = {Path(f).stem.lower(): f for f in repo_files}
        self.parsers = parser_pool

    def analyze(self, file_path: str, lang: str, content: str) -> List[Dict[str, Any]]:
        parser = self.parsers.get_parser(lang)
        if not parser:
            return []

        src = content.encode("utf-8")
        try:
            tree = parser.parse(src)
        except Exception as e:
            log.error(f"Parse failed for {file_path}: {e}")
            return []

        edges = []
        stack = [tree.root_node]

        while stack:
            node = stack.pop()
            ntype = node.type
            text = src[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

            # --- Imports ---
            if ntype in self.IMPORT_NODES:
                target = self._resolve_target(text)
                if target:
                    edges.append(self._edge(file_path, target, "import", {"raw": text.strip()}))

            # --- Inheritance ---
            if ntype in self.CLASS_NODES:
                for cname in re.findall(r"(?:extends|implements|:)\s+([A-Z][A-Za-z0-9_]*)", text):
                    target = self.index.get(cname.lower())
                    if target:
                        edges.append(self._edge(file_path, target, "inherits", {"class": cname}))

            # --- Calls ---
            if ntype in self.CALL_NODES:
                for callee in re.findall(r"([A-Z][A-Za-z0-9_]*)\s*\(", text):
                    target = self.index.get(callee.lower())
                    if target:
                        edges.append(self._edge(file_path, target, "call", {"callee": callee}))

            stack.extend(node.children)
        return edges

    def _resolve_target(self, import_text: str) -> Optional[str]:
        tokens = re.split(r"[.\s]+", import_text.strip())
        for tok in reversed(tokens):
            f = self.index.get(tok.lower())
            if f:
                return f
        return None

    def _edge(self, src: str, dst: str, rel: str, details: Dict[str, Any]):
        return {"from_file": src, "to_file": dst, "relation_type": rel, "details": details}

# ---------------------------
# Orchestrator
# ---------------------------
class DependencyGraphBuilder:
    def __init__(self, cfg: Dict[str, Any]):
        repo_cfg = cfg["repository"]
        db_cfg = cfg["database"]
        graph_cfg = cfg.get("graph", {})

        self.repo_dir = Path(repo_cfg["local_path"]).expanduser()
        self.repo_url = repo_cfg["url"]
        self.lib_path = graph_cfg.get("lib_path", "build/my-languages.so")
        self.batch_size = graph_cfg.get("batch_size", 200)
        self.workers = graph_cfg.get("workers", 6)
        self.supported_exts = cfg["indexing"]["supported_extensions"]

        self.store = GraphStore(db_cfg)
        self.store.ensure_table()

        repo_files = [str(f) for f in self.repo_dir.rglob("*") if f.suffix in self.supported_exts and f.is_file()]
        self.parser_pool = ParserPool(self.lib_path)
        self.analyzer = DependencyAnalyzer(repo_files, self.parser_pool)

    def run(self):
        files = [str(f) for f in self.repo_dir.rglob("*") if f.suffix in self.supported_exts]
        all_edges = []

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futures = {ex.submit(self._process, f): f for f in files}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Building dependency graph"):
                try:
                    edges = fut.result()
                    if edges:
                        all_edges.extend(edges)
                        if len(all_edges) >= self.batch_size:
                            self._flush(all_edges)
                except Exception as e:
                    log.error(f"Error analyzing {futures[fut]}: {e}")

        if all_edges:
            self._flush(all_edges)
        self.store.close()
        log.info("✅ Dependency graph build complete.")

    def _process(self, path: str):
        ext = Path(path).suffix
        lang = self.supported_exts.get(ext)
        if not lang:
            return []
        try:
            content = Path(path).read_text(encoding="utf-8", errors="ignore")
            return self.analyzer.analyze(path, lang, content)
        except Exception as e:
            log.warning(f"⚠️ Skipping {path}: {e}")
            return []

    def _flush(self, edges: List[Dict[str, Any]]):
        self.store.upsert_batch(self.repo_url, edges)
        edges.clear()
        log.info("💾 Flushed edges to DB")

