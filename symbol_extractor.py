import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List

import psycopg
from psycopg.rows import dict_row
from tree_sitter import Language, Parser
from tqdm import tqdm


class SymbolExtractor:
    """Extracts symbols, imports/exports, variables, and docstrings from source files using Tree-sitter."""
    LIB_PATH = "build/my-languages.so"  # Path to your compiled grammar bundle

    def __init__(self, supported_exts: Dict[str, str]):
        self.parsers: Dict[str, Parser] = {}
        self.supported_exts = supported_exts

    def _get_parser(self, lang: str) -> Parser:
        if lang not in self.parsers:
            try:
                language = Language(self.LIB_PATH, lang)
                parser = Parser()
                parser.set_language(language)
                self.parsers[lang] = parser
            except Exception as e:
                print(f"⚠️ Could not load parser for '{lang}': {e}")
                raise
        return self.parsers[lang]

    def _hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()

    def extract(self, file_path: str, content: str) -> Dict[str, Any]:
        ext = os.path.splitext(file_path)[1]
        lang = self.supported_exts.get(ext)
        if not lang:
            return {}

        parser = self._get_parser(lang)
        tree = parser.parse(bytes(content, "utf8"))
        root = tree.root_node

        result = {
            "file_path": file_path,
            "symbols": [],
            "imports": [],
            "exports": [],
            "variables": [],
            "hash": self._hash(content),
        }

        # language-specific node mappings
        function_nodes = {
            "python": ["function_definition"],
            "javascript": ["function_declaration", "method_definition"],
            "typescript": ["function_declaration", "method_definition"],
            "tsx": ["function_declaration", "method_definition"],
            "java": ["method_declaration", "constructor_declaration"],
            "kotlin": ["function_declaration"],
        }

        class_nodes = {
            "python": ["class_definition"],
            "javascript": ["class_declaration"],
            "typescript": ["class_declaration"],
            "tsx": ["class_declaration"],
            "java": ["class_declaration"],
            "kotlin": ["class_declaration", "object_declaration"],
        }

        import_nodes = {
            "python": ["import_statement", "import_from_statement"],
            "javascript": ["import_statement"],
            "typescript": ["import_statement"],
            "tsx": ["import_statement"],
            "java": ["import_declaration"],
            "kotlin": ["import_header"],
        }

        export_nodes = {
            "javascript": ["export_statement"],
            "typescript": ["export_statement"],
            "tsx": ["export_statement"],
        }

        variable_nodes = {
            "javascript": ["lexical_declaration", "variable_declaration"],
            "typescript": ["lexical_declaration", "variable_declaration"],
            "tsx": ["lexical_declaration", "variable_declaration"],
            "python": ["assignment"],
            "java": ["field_declaration", "local_variable_declaration"],
            "kotlin": ["property_declaration"],
        }

        def get_node_name(node):
            """
            Robust multi-language node name extractor.
            Tries known field names, then falls back to identifier-like children.
            """
            for field in ("name", "identifier", "declarator", "simple_identifier", "function_name"):
                n = node.child_by_field_name(field)
                if n:
                    return n.text.decode("utf-8", errors="ignore")

            # Fallback: find child node whose type suggests an identifier
            for c in node.children:
                if any(k in c.type.lower() for k in ("identifier", "name", "declarator")):
                    return c.text.decode("utf-8", errors="ignore")

            # Still nothing found
            return None

        def walk(node):
            t = node.type

            # --- Functions ---
            if t in function_nodes.get(lang, []):
                name = get_node_name(node)
                if not name:
                    # Skip true anonymous functions (like arrow/lambdas)
                    return
                result["symbols"].append({
                    "name": name,
                    "type": "function",
                    "line_start": node.start_point[0] + 1,
                    "line_end": node.end_point[0] + 1,
                    "signature": node.text.decode("utf-8", errors="ignore")[:300]
                })

            # --- Classes ---
            elif t in class_nodes.get(lang, []):
                name = get_node_name(node)
                if not name:
                    return
                result["symbols"].append({
                    "name": name,
                    "type": "class",
                    "line_start": node.start_point[0] + 1,
                    "line_end": node.end_point[0] + 1,
                    "signature": node.text.decode("utf-8", errors="ignore")[:300]
                })

            # --- Imports / Exports ---
            elif t in import_nodes.get(lang, []):
                text = node.text.decode("utf-8", errors="ignore").strip()
                if text not in result["imports"]:
                    result["imports"].append(text)
            elif t in export_nodes.get(lang, []):
                text = node.text.decode("utf-8", errors="ignore").strip()
                if text not in result["exports"]:
                    result["exports"].append(text)

            # --- Variables / Constants ---
            elif t in variable_nodes.get(lang, []):
                name = get_node_name(node)
                if not name:
                    # fallback for declarations like: const x = ..., val y = ...
                    raw = node.text.decode("utf-8", errors="ignore")
                    name = raw.split("=")[0].strip().split()[-1] if "=" in raw else raw.strip()
                result["variables"].append({
                    "name": name,
                    "type": "variable",
                    "line_start": node.start_point[0] + 1,
                    "line_end": node.end_point[0] + 1
                })

            # Recursively visit children
            for child in node.children:
                walk(child)

        walk(root)
        return result

class SymbolDB:
    """Handles writing extracted symbols into PostgreSQL."""
    def __init__(self, db_cfg: dict):
        self.db_cfg = db_cfg
        self.conn = None

    def connect(self):
        if not self.conn:
            self.conn = psycopg.connect(
                host=self.db_cfg.get("host", "localhost"),
                port=self.db_cfg.get("port", 5432),
                dbname=self.db_cfg.get("database"),
                user=self.db_cfg.get("user"),
                password=self.db_cfg.get("password"),
                row_factory=dict_row,
            )

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def upsert_symbols(self, record: Dict[str, Any]):
        self.connect()
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO code_symbols (file_path, hash, symbols, imports, exports, variables, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (file_path, hash) DO UPDATE SET
                    symbols = EXCLUDED.symbols,
                    imports = EXCLUDED.imports,
                    exports = EXCLUDED.exports,
                    variables = EXCLUDED.variables,
                    updated_at = now();
                """,
                (
                    record["file_path"],
                    record["hash"],
                    json.dumps(record.get("symbols", [])),
                    json.dumps(record.get("imports", [])),
                    json.dumps(record.get("exports", [])),
                    json.dumps(record.get("variables", [])),
                )
            )
            self.conn.commit()


class SymbolCrawler:
    """Main orchestrator: reads repo files, extracts symbols, and stores them."""
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.repo_dir = Path(config["repository"]["local_path"]).expanduser()
        self.supported_exts = config["indexing"]["supported_extensions"]
        self.extractor = SymbolExtractor(self.supported_exts)
        self.db = SymbolDB(config["database"])

    def run(self):
        print(f"🔍 Scanning repository at: {self.repo_dir}")
        files = [
            f for f in self.repo_dir.rglob("*")
            if f.suffix in self.supported_exts.keys() and f.is_file()
        ]

        for file in tqdm(files, desc="Extracting symbols"):
            try:
                content = file.read_text(encoding="utf-8", errors="ignore")
                record = self.extractor.extract(str(file), content)
                if record:
                    self.db.upsert_symbols(record)
                print(f"✅ Processed {file}")
            except Exception as e:
                print(f"⚠️ Failed to process {file}: {e}")

        self.db.close()
        print("✅ Symbol extraction complete.")
