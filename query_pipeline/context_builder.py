import logging

logger = logging.getLogger(__name__)

def build_context(results, cfg):
    """Smart context builder that prioritizes relevance and recency."""
    max_tokens = cfg["query_pipeline"]["max_context_tokens"]
    context, token_count = [], 0

    if isinstance(results, dict) and "chunks" in results:
        results = results["chunks"]

    for r in results:
        # Support multiple data shapes
        if "chunk_text" in r:
            text = r["chunk_text"]
        elif "symbols" in r:
            text = str(r["symbols"])
        elif "from_file" in r and "to_file" in r:
            text = f"{r['from_file']} → {r['to_file']} ({r['relation_type']})"
        else:
            continue

        token_est = len(text.split())
        if token_count + token_est < max_tokens:
            context.append(text)
            token_count += token_est
        else:
            break

    logger.info(f"🧩 Context built with {len(context)} chunks ({token_count} tokens)")
    return context
