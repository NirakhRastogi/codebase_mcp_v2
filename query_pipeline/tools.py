TOOLS = [
    {
        "name": "search_code",
        "description": "Search for code using natural language or symbols",
        "parameters": {
            "query": "authentication logic",
            "search_type": "semantic | symbol | text",
            "max_results": 5,
            "file_filter": "*.ts",
            "similarity_threshold": 0.7
        }
    },
    {
        "name": "get_definition",
        "description": "Get the definition of a symbol",
        "parameters": {"symbol_name": "handlePayment", "file_hint": "optional/path/hint.ts"}
    },
    {
        "name": "find_usages",
        "description": "Find all usages of a symbol",
        "parameters": {"symbol_name": "UserModel", "include_context": True}
    },
    {
        "name": "explain_code",
        "description": "Explain code section in detail",
        "parameters": {"file_path": "src/utils/auth.ts", "line_start": 45, "line_end": 80}
    },
    {
        "name": "trace_execution",
        "description": "Follow execution path from entry point",
        "parameters": {"entry_point": "handleRequest", "max_depth": 3}
    },
    {
        "name": "get_architecture",
        "description": "Get high-level architecture overview",
        "parameters": {"focus_area": "backend | frontend | full"}
    },
    {
        "name": "sync_repository",
        "description": "Pull latest repo changes and optionally force reindex",
        "parameters": {"force": False}
    },
    {
        "name": "get_repo_status",
        "description": "Fetch current repository sync status",
        "parameters": {}
    }
]
