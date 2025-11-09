import logging

logger = logging.getLogger(__name__)

# -------------------------------------------------
# STRATEGY 1 — SYMBOL LOOKUP
# -------------------------------------------------
def symbol_lookup(conn, query: str, usage=False):
    keyword = query.strip().split()[-1].replace("?", "")
    logger.info(f"🔍 Symbol lookup for '{keyword}'")

    sql = """
        SELECT file_path, symbols, updated_at
        FROM public.code_symbols
        WHERE symbols::text ILIKE %s
        ORDER BY updated_at DESC
        LIMIT 10;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (f"%{keyword}%",))
        rows = cur.fetchall()
    return rows


# -------------------------------------------------
# STRATEGY 2 — DEPENDENCY GRAPH
# -------------------------------------------------
def dependency_traversal(conn, query: str, deep=False):
    keyword = query.strip().split()[-1].lower()
    depth_limit = 3 if deep else 1
    logger.info(f"🔗 Dependency traversal for '{keyword}' depth={depth_limit}")

    sql = """
        SELECT from_file, to_file, relation_type
        FROM public.code_dependencies
        WHERE from_file ILIKE %s OR to_file ILIKE %s
        LIMIT 50;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (f"%{keyword}%", f"%{keyword}%"))
        rows = cur.fetchall()
    return rows


# -------------------------------------------------
# STRATEGY 3 — SEMANTIC SEARCH (pgvector)
# -------------------------------------------------
def semantic_search(conn, query: str, embed_model, cfg):
    threshold = cfg["query_pipeline"]["semantic_threshold"]
    logger.info(f"🧠 Semantic search for '{query}' (threshold={threshold})")

    emb = embed_model.encode([query], normalize_embeddings=True)[0].tolist()

    # 🔍 Detect which column stores chunk text
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name='code_chunks';
        """)
        rows = cur.fetchall()

    # ✅ Handle both dict_row and tuple row_factory cases
    cols = [r["column_name"] if isinstance(r, dict) else r[0] for r in rows]

    # Prefer whichever exists
    text_col = next((c for c in ["chunk_text", "chunk", "content", "text"] if c in cols), None)
    if not text_col:
        raise RuntimeError("❌ No valid text column found in 'code_chunks' table.")

    sql = f"""
        SELECT 
            id, file_path, symbol_name, {text_col} AS chunk_text,
            1 - (embedding <=> %s::vector) AS similarity
        FROM code_chunks
        WHERE (1 - (embedding <=> %s::vector)) > %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (emb, emb, threshold, emb, cfg["query_pipeline"]["max_results"]))
        rows = cur.fetchall()

    logger.info(f"✅ Retrieved {len(rows)} semantic matches using '{text_col}' column")
    return rows



# -------------------------------------------------
# STRATEGY 4 — DEEP ANALYSIS (Semantic + Graph)
# -------------------------------------------------
def deep_analysis(conn, query: str, embed_model, cfg):
    logger.info("🧩 Running deep analysis pipeline...")
    chunks = semantic_search(conn, query, embed_model, cfg)
    if not chunks:
        return []

    # Collect related files
    file_paths = [c["file_path"] for c in chunks]

    sql = """
        SELECT from_file, to_file, relation_type
        FROM public.code_dependencies
        WHERE from_file = ANY(%s);
    """
    with conn.cursor() as cur:
        cur.execute(sql, (file_paths,))
        deps = cur.fetchall()

    return {"chunks": chunks, "dependencies": deps}
