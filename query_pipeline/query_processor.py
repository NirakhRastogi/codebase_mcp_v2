import re
import logging
import psycopg
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer
from .retrieval_strategies import (
    symbol_lookup,
    dependency_traversal,
    semantic_search,
    deep_analysis
)
from .context_builder import build_context

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class QueryProcessor:
    """Handles query understanding, routing, and execution"""

    def __init__(self, cfg):
        self.cfg = cfg
        db_cfg = cfg["database"]
        self.conn = psycopg.connect(
            host=db_cfg["host"],
            port=db_cfg.get("port", 5432),
            dbname=db_cfg["database"],
            user=db_cfg["user"],
            password=db_cfg["password"],
            row_factory=dict_row,
        )
        model_name = cfg["query_pipeline"]["default_model"]
        self.embed_model = SentenceTransformer(model_name)

    # -------------------------------------------------
    # INTENT CLASSIFIER
    # -------------------------------------------------
    def classify_intent(self, query: str) -> str:
        q = query.lower()
        if re.search(r"\b(where|definition|defined)\b", q):
            return "find_definition"
        if re.search(r"\b(usage|used|invok|reference)\b", q):
            return "find_usage"
        if re.search(r"\b(explain|describe|how|what does)\b", q):
            return "explain_function"
        if re.search(r"\b(flow|trace|path)\b", q):
            return "trace_flow"
        if re.search(r"\b(similar|like|related)\b", q):
            return "find_similar"
        if re.search(r"\b(structure|architecture|dependency)\b", q):
            return "architectural"
        return "find_similar"

    # -------------------------------------------------
    # STRATEGY ROUTER
    # -------------------------------------------------
    def process_query(self, query: str):
        intent = self.classify_intent(query)
        logger.info(f"🧭 Intent classified as: {intent}")

        if intent == "find_definition":
            results = symbol_lookup(self.conn, query, usage=False)
        elif intent == "find_usage":
            results = symbol_lookup(self.conn, query, usage=True)
        elif intent == "architectural":
            results = dependency_traversal(self.conn, query)
        elif intent == "trace_flow":
            results = dependency_traversal(self.conn, query, deep=True)
        elif intent in ("find_similar", "explain_function"):
            results = semantic_search(self.conn, query, self.embed_model, self.cfg)
        else:
            results = deep_analysis(self.conn, query, self.embed_model, self.cfg)

        context = build_context(results, self.cfg)
        return {"intent": intent, "results": results, "context": context}
