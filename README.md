# Efficient Codebase Analysis MCP Server - Architecture Plan

## Overview
An MCP server that provides intelligent codebase analysis without loading everything into memory at once. The codebase is hosted on GitHub and synchronized locally with efficient git-based updates.

## Core Architecture

### 1. Repository Synchronization Layer

#### Initial Clone
```python
def initialize_repository(repo_url, local_path):
    """
    Clone repository with optimizations
    """
    # Shallow clone to save bandwidth and storage
    git.clone(
        repo_url, 
        local_path,
        depth=1,  # Only latest commit initially
        single_branch=True,  # Only main branch
        filter='blob:none'  # Blobless clone - fetch files on demand
    )
    
    # Store metadata
    store_repo_metadata({
        'url': repo_url,
        'local_path': local_path,
        'current_commit': get_current_commit_hash(),
        'last_synced': timestamp()
    })
```

#### Incremental Update Strategy
```python
def sync_repository():
    """
    Pull only changes since last sync
    """
    current_commit = get_stored_commit_hash()
    
    # Fetch changes
    git.fetch(origin='origin', depth=1)
    
    # Get diff between commits
    changed_files = git.diff(
        current_commit,
        'origin/main',
        name_only=True
    )
    
    # Categorize changes
    changes = {
        'added': [],
        'modified': [],
        'deleted': [],
        'renamed': []
    }
    
    for file_info in git.diff_tree(current_commit, 'origin/main'):
        status = file_info.status
        if status == 'A':
            changes['added'].append(file_info.path)
        elif status == 'M':
            changes['modified'].append(file_info.path)
        elif status == 'D':
            changes['deleted'].append(file_info.path)
        elif status == 'R':
            changes['renamed'].append({
                'old': file_info.old_path,
                'new': file_info.new_path
            })
    
    # Pull the actual changes
    git.pull(origin='origin', branch='main')
    
    # Update indexes incrementally
    process_changes(changes)
    
    # Update commit hash
    update_stored_commit_hash(get_current_commit_hash())
    
    return changes
```

#### Smart Sync Scheduling
- **On-Demand**: When user makes a query, check if sync needed
- **Periodic**: Background sync every N minutes (configurable)
- **Webhook-Triggered**: If GitHub webhook configured, sync on push events
- **Manual**: Expose tool for explicit sync request

### 2. Indexing Phase (One-time + Incremental Updates)

#### Git-Aware File System Crawler
- **Purpose**: Build a lightweight index of the codebase
- **Strategy**: 
  - Use git ls-tree for file discovery (faster than filesystem scan)
  - Extract metadata from git (commit hash, author, last modified)
  - Respect `.gitignore` automatically via git
  - Use git blame for tracking file history
  - Store in PostgreSQL database with pgvector extension

#### Symbol Extractor
- **Purpose**: Extract high-level code structure without full parsing
- **What to Index**:
  - Function/method signatures
  - Class definitions
  - Import/export statements
  - Comments and docstrings (first line only)
  - File-level metadata
- **Tools**: Use language-specific parsers (Tree-sitter for multi-language support)
- **Storage**: Structured index with:
  ```
  {
    file_path: string,
    symbols: [{name, type, line_start, line_end, signature}],
    imports: [string],
    exports: [string],
    hash: string  // for change detection
  }
  ```

#### Dependency Graph Builder
- **Purpose**: Map relationships between files
- **Data**:
  - Import/require relationships
  - Call graphs (lightweight, symbol-level)
  - Inheritance hierarchies
- **Storage**: Graph database or adjacency lists

#### Semantic Chunking with pgvector
- **Purpose**: Create searchable, contextual chunks
- **Strategy**:
  - Chunk by logical units (functions, classes, modules)
  - Include surrounding context (imports, related definitions)
  - Generate embeddings for semantic search
  - Store in PostgreSQL with pgvector extension
  - Store chunk metadata: file, line numbers, parent symbols, git commit hash

**Vector Storage Schema**:
```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Code chunks table with embeddings
CREATE TABLE code_chunks (
    id SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES files(id),
    chunk_text TEXT,
    chunk_type TEXT,  -- 'function', 'class', 'module', etc.
    symbol_name TEXT,
    line_start INTEGER,
    line_end INTEGER,
    embedding vector(384),  -- Dimension depends on model
    commit_hash TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create HNSW index for fast vector search
CREATE INDEX ON code_chunks 
USING hnsw (embedding vector_cosine_ops);

-- Create GiST index for metadata filtering
CREATE INDEX idx_chunks_file ON code_chunks(file_id);
CREATE INDEX idx_chunks_symbol ON code_chunks(symbol_name);
CREATE INDEX idx_chunks_type ON code_chunks(chunk_type);
```

### 2. Query Processing Pipeline

#### Query Analysis Layer
```
User Query → Intent Classification → Strategy Selection
```

**Intent Types**:
- `find_definition`: Locate where something is defined
- `find_usage`: Find all usages of a symbol
- `explain_function`: Understand specific code
- `trace_flow`: Follow execution path
- `find_similar`: Semantic code search
- `architectural`: High-level structure questions

#### Strategy Router
Based on intent, route to appropriate retrieval strategy:

**Strategy 1: Symbol-Based Lookup** (Fast)
- For: "Where is function X defined?"
- Process: Direct index lookup → return file + line number
- Load: Only the relevant file section

**Strategy 2: Dependency Traversal** (Medium)
- For: "What does this module depend on?"
- Process: Graph query → return connected nodes
- Load: Only dependency metadata

**Strategy 3: Semantic Search with pgvector** (Medium-Slow)
- For: "Find code that handles authentication"
- Process: 
  1. Embed query using same model as indexing
  2. Vector similarity search using pgvector cosine similarity
  3. Use metadata filters for better results
  4. Rank by relevance + recency
  5. Return top-k chunks with context
- SQL Query:
  ```sql
  SELECT 
      c.id,
      c.chunk_text,
      c.symbol_name,
      c.line_start,
      c.line_end,
      f.path,
      1 - (c.embedding <=> $1::vector) as similarity
  FROM code_chunks c
  JOIN files f ON c.file_id = f.id
  WHERE 
      c.chunk_type = ANY($2)  -- Optional: filter by type
      AND 1 - (c.embedding <=> $1::vector) > 0.7  -- Similarity threshold
  ORDER BY c.embedding <=> $1::vector
  LIMIT 10;
  ```
- Load: Only matching chunks + surrounding context

**Strategy 4: Deep Analysis** (Slow, on-demand)
- For: "Explain how the payment flow works"
- Process:
  1. Use semantic search to find entry points
  2. Traverse dependency graph
  3. Load only relevant files
  4. Build focused context for LLM
  5. Stream analysis incrementally

### 3. Context Assembly

#### Smart Context Builder
```python
def build_context(query, max_tokens=8000):
    relevant_symbols = find_relevant(query)
    context = []
    token_count = 0
    
    # Priority order
    for symbol in rank_by_relevance(relevant_symbols):
        chunk = load_chunk(symbol)
        if token_count + len(chunk) < max_tokens:
            context.append(chunk)
            token_count += len(chunk)
        else:
            break
    
    return context
```

**Ranking Factors**:
- Semantic similarity to query
- Recency of file modification
- Import depth (prefer higher-level modules)
- Symbol type (prioritize definitions over usages)
- User's recent context (if available)

### 4. MCP Server Tools

#### Tool Design

```typescript
// Tool 1: Search Codebase
{
  name: "search_code",
  description: "Search for code using natural language or symbols",
  parameters: {
    query: "authentication logic",
    search_type: "semantic" | "symbol" | "text",
    max_results: 5,
    file_filter: "*.ts",  // Optional glob pattern
    similarity_threshold: 0.7  // For semantic search
  }
}

// Tool 2: Get Definition
{
  name: "get_definition",
  description: "Get the definition of a symbol",
  parameters: {
    symbol_name: "handlePayment",
    file_hint: "optional/path/hint.ts"
  }
}

// Tool 3: Find Usages
{
  name: "find_usages",
  description: "Find all usages of a symbol",
  parameters: {
    symbol_name: "UserModel",
    include_context: true
  }
}

// Tool 4: Explain Code
{
  name: "explain_code",
  description: "Get detailed explanation of code section",
  parameters: {
    file_path: "src/utils/auth.ts",
    line_start: 45,
    line_end: 80
  }
}

// Tool 5: Trace Execution
{
  name: "trace_execution",
  description: "Follow execution path from entry point",
  parameters: {
    entry_point: "handleRequest",
    max_depth: 3
  }
}

// Tool 6: Get Architecture
{
  name: "get_architecture",
  description: "Get high-level architecture overview",
  parameters: {
    focus_area: "backend" | "frontend" | "full"
  }
}

// Tool 7: Sync Repository
{
  name: "sync_repository",
  description: "Pull latest changes from GitHub",
  parameters: {
    force: false  // Force full re-index if true
  }
}

// Tool 8: Get Repository Status
{
  name: "get_repo_status",
  description: "Get current repository sync status",
  parameters: {}
  // Returns: current commit, last sync time, pending changes
}
```

### 5. Caching Strategy

#### Multi-Level Cache

**L1: In-Memory Cache**
- Recently accessed chunks
- Frequent queries and their results
- Embedding cache for common queries
- Symbol lookup cache
- TTL: 5 minutes
- Size: 100MB max

**L2: Disk Cache**
- Parsed file contents (keyed by commit hash + file path)
- Intermediate analysis results
- Pre-computed dependency paths
- TTL: Until file changes in git
- Location: `~/.cache/mcp-codebase/`

**L3: PostgreSQL with pgvector**
- Persistent symbol index
- Dependency graph
- File metadata
- Vector embeddings
- Update: On git diff detection

**Cache Invalidation Strategy**:
```python
def invalidate_cache(file_path):
    """
    Invalidate caches when file changes detected via git
    """
    # L1: Clear in-memory cache for this file
    memory_cache.delete(f"file:{file_path}")
    memory_cache.delete_pattern(f"query:*:{file_path}")
    
    # L2: Remove disk cache entries
    disk_cache.delete(f"{current_commit}:{file_path}")
    
    # L3: Updated via process_file_update()
    # Embeddings are versioned by commit_hash
```

### 6. Incremental Updates (Git-Based)

#### Git Diff Processor
```python
def process_changes(changes):
    """
    Process git diff and update indexes incrementally
    """
    # Handle deletions first
    for deleted_file in changes['deleted']:
        file_id = get_file_id(deleted_file)
        if file_id:
            # Delete from database
            delete_symbols(file_id)
            delete_chunks(file_id)
            delete_dependencies(file_id)
            delete_file_record(file_id)
    
    # Handle renames
    for rename in changes['renamed']:
        update_file_path(rename['old'], rename['new'])
    
    # Handle additions and modifications
    files_to_process = changes['added'] + changes['modified']
    
    for file_path in files_to_process:
        # Check if file should be indexed (not binary, not too large)
        if should_index_file(file_path):
            process_file_update(file_path)

def process_file_update(file_path):
    """
    Update index for a single file
    """
    # Compute hash
    content = read_file(file_path)
    new_hash = compute_hash(content)
    
    file_record = get_file_record(file_path)
    
    if file_record and file_record.hash == new_hash:
        # No actual changes, skip
        return
    
    # Extract symbols
    symbols = extract_symbols(content, file_path)
    
    # Update or insert file record
    file_id = upsert_file(file_path, new_hash)
    
    # Delete old symbols and chunks
    delete_symbols(file_id)
    delete_chunks(file_id)
    
    # Insert new symbols
    for symbol in symbols:
        insert_symbol(file_id, symbol)
    
    # Generate chunks and embeddings
    chunks = create_chunks(content, symbols, file_path)
    
    # Batch embed chunks
    embeddings = generate_embeddings([chunk.text for chunk in chunks])
    
    # Insert chunks with embeddings into PostgreSQL
    for chunk, embedding in zip(chunks, embeddings):
        insert_chunk_with_embedding(
            file_id=file_id,
            chunk=chunk,
            embedding=embedding,
            commit_hash=get_current_commit_hash()
        )
    
    # Update dependencies
    update_dependencies(file_id, content)
    
    # Invalidate related caches
    invalidate_cache(file_path)

def insert_chunk_with_embedding(file_id, chunk, embedding, commit_hash):
    """
    Insert chunk with pgvector embedding
    """
    execute_sql("""
        INSERT INTO code_chunks 
        (file_id, chunk_text, chunk_type, symbol_name, line_start, line_end, embedding, commit_hash)
        VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8)
    """, [
        file_id,
        chunk.text,
        chunk.type,
        chunk.symbol_name,
        chunk.line_start,
        chunk.line_end,
        embedding.tolist(),  # Convert numpy array to list
        commit_hash
    ])
```

#### Webhook Integration (Optional)
```python
# FastAPI endpoint for GitHub webhooks
@app.post("/webhook/github")
async def github_webhook(payload: dict, signature: str):
    """
    Handle GitHub push events
    """
    # Verify webhook signature
    if not verify_signature(payload, signature):
        raise HTTPException(401)
    
    if payload['ref'] == 'refs/heads/main':
        # Trigger sync in background
        background_tasks.add_task(sync_repository)
        
    return {"status": "ok"}
```

### 7. Performance Optimizations

#### Lazy Loading
- Load file contents only when needed
- Parse syntax only for active queries
- Generate embeddings on-demand (with caching)

#### Parallel Processing
- Index multiple files concurrently
- Batch embedding generation
- Parallel dependency resolution

#### Smart Sampling
- For large files, analyze in sections
- Provide file summaries before full content
- Progressive detail loading

### 8. Implementation Stack

**Core Technologies**:
- **MCP Framework**: Model Context Protocol SDK
- **Database**: PostgreSQL 15+ with pgvector extension
- **Git Integration**: GitPython or libgit2 (pygit2)
- **Parsing**: Tree-sitter (multi-language)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2 or all-mpnet-base-v2)
- **Language**: Python 3.10+ with asyncio
- **API Framework**: FastAPI (for webhook endpoints)
- **Background Tasks**: Celery or asyncio tasks

**Database Schema**:
```sql
-- Repository metadata
CREATE TABLE repositories (
    id SERIAL PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    local_path TEXT NOT NULL,
    current_commit TEXT,
    last_synced TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Files table
CREATE TABLE files (
    id SERIAL PRIMARY KEY,
    repo_id INTEGER REFERENCES repositories(id),
    path TEXT NOT NULL,
    hash TEXT,
    language TEXT,
    size_bytes INTEGER,
    last_modified TIMESTAMP,
    commit_hash TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(repo_id, path)
);

CREATE INDEX idx_files_repo ON files(repo_id);
CREATE INDEX idx_files_path ON files(path);
CREATE INDEX idx_files_commit ON files(commit_hash);

-- Symbols table
CREATE TABLE symbols (
    id SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    type TEXT,  -- 'function', 'class', 'variable', etc.
    line_start INTEGER,
    line_end INTEGER,
    signature TEXT,
    docstring TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_symbols_file ON symbols(file_id);
CREATE INDEX idx_symbols_name ON symbols(name);
CREATE INDEX idx_symbols_type ON symbols(type);

-- Dependencies table
CREATE TABLE dependencies (
    id SERIAL PRIMARY KEY,
    from_file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    to_file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    import_type TEXT,  -- 'import', 'require', 'include'
    symbol_name TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(from_file_id, to_file_id, symbol_name)
);

CREATE INDEX idx_deps_from ON dependencies(from_file_id);
CREATE INDEX idx_deps_to ON dependencies(to_file_id);

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Code chunks table with embeddings
CREATE TABLE code_chunks (
    id SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_type TEXT,  -- 'function', 'class', 'module', 'comment'
    symbol_name TEXT,
    line_start INTEGER,
    line_end INTEGER,
    embedding vector(384),  -- Adjust dimension based on model
    commit_hash TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX idx_chunks_embedding ON code_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Additional indexes for filtering
CREATE INDEX idx_chunks_file ON code_chunks(file_id);
CREATE INDEX idx_chunks_symbol ON code_chunks(symbol_name);
CREATE INDEX idx_chunks_type ON code_chunks(chunk_type);
CREATE INDEX idx_chunks_commit ON code_chunks(commit_hash);
```

**Environment Configuration**:
```yaml
# config.yaml
repository:
  url: "https://github.com/username/repo.git"
  local_path: "./repos/repo"
  branch: "main"
  sync_interval_minutes: 15
  
database:
  host: "localhost"
  port: 5432
  database: "codebase_mcp"
  user: "mcp_user"
  password: "${DB_PASSWORD}"

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
  batch_size: 32
  device: "cuda"  # or "cpu"

indexing:
  max_file_size_kb: 1024
  supported_extensions: [".py", ".js", ".ts", ".java", ".go", ".rs"]
  chunk_size_lines: 50
  chunk_overlap_lines: 10

cache:
  memory_size_mb: 100
  disk_path: "~/.cache/mcp-codebase"

webhooks:
  enabled: true
  secret: "${GITHUB_WEBHOOK_SECRET}"
  port: 8080
```

## Example Query Flow

**Query**: "How does the authentication system work?"

1. **Sync Check**: Check if repo needs sync (last sync > 15 min ago)
   - If yes: Run `sync_repository()` in background
   - Continue with current indexed data
2. **Query Analysis**: Intent = `architectural + explain`
3. **Symbol Search**: Query PostgreSQL for "auth*" symbols → 15 matches
4. **Semantic Search**: 
   - Embed query: "authentication system" → vector(384)
   - Query pgvector:
   ```sql
   SELECT c.*, f.path, 1 - (c.embedding <=> $1::vector) as similarity
   FROM code_chunks c
   JOIN files f ON c.file_id = f.id
   WHERE 1 - (c.embedding <=> $1::vector) > 0.7
   ORDER BY c.embedding <=> $1::vector
   LIMIT 5;
   ```
   - Get top 5 relevant chunks
5. **Dependency Traversal**: Get auth module dependencies from dependency graph
6. **Context Assembly**: 
   - Load main auth module from git (2KB)
   - Load key middleware (1.5KB)
   - Load config relevant to auth (0.5KB)
   - Total: 4KB of 8KB budget
7. **Response**: Provide structured explanation with:
   - File references with GitHub URLs
   - Line numbers
   - Commit hash for version tracking
   - Links to definitions

## Git-Specific Optimizations

### 1. Blobless Clone
```bash
git clone --filter=blob:none --depth=1 <repo-url>
```
- Downloads only tree and commit objects initially
- Fetches file contents on-demand
- Saves ~70% initial clone time for large repos

### 2. Sparse Checkout (Optional)
For monorepos, only checkout relevant directories:
```python
def setup_sparse_checkout(paths):
    """
    Only checkout specified paths
    """
    git.config('core.sparseCheckout', 'true')
    with open('.git/info/sparse-checkout', 'w') as f:
        for path in paths:
            f.write(f"{path}\n")
    git.read_tree('-mu', 'HEAD')
```

### 3. Git Object Database Queries
Instead of reading files from disk, query git directly:
```python
def get_file_at_commit(file_path, commit_hash):
    """
    Get file content without checkout
    """
    return git.show(f"{commit_hash}:{file_path}")
```

### 4. Efficient Diff Processing
```python
def get_changed_files_efficient(from_commit, to_commit):
    """
    Get changed files without full checkout
    """
    return git.diff_tree(
        '--no-commit-id',
        '--name-status',
        '-r',
        from_commit,
        to_commit
    )
```

## Scalability Considerations

- **Small codebases (<1000 files)**: 
  - Full embedding index
  - Sync every 5 minutes
  - Fast queries (<500ms)
  
- **Medium (1000-10000 files)**: 
  - Selective embedding (prioritize frequently accessed files)
  - Sync every 15 minutes
  - Use pgvector's HNSW index for fast searches
  - Parallel processing during indexing
  
- **Large (>10000 files)**: 
  - Hierarchical indexing (module-level summaries first)
  - Smart sync (only active branches)
  - Batch embedding generation with GPU
  - Consider multiple worker processes
  - Use PostgreSQL partitioning by repository/module
  - Incremental git fetch with shallow depth

**PostgreSQL Performance Tuning**:
```sql
-- Tune HNSW index parameters for your dataset
CREATE INDEX idx_chunks_embedding ON code_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 16,              -- Increase for better recall (16-64)
    ef_construction = 64  -- Increase for better index quality (64-200)
);

-- Query-time tuning
SET hnsw.ef_search = 100;  -- Higher = better accuracy, slower

-- Connection pooling
-- Use pgBouncer or SQLAlchemy pool for concurrent requests
```

**Embedding Generation Optimization**:
```python
def batch_embed_chunks(chunks, batch_size=32):
    """
    Generate embeddings in batches for efficiency
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        embeddings = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        all_embeddings.extend(embeddings)
    
    return all_embeddings
```

## Success Metrics

- Query response time: <2s for 90% of queries (symbol lookup <200ms)
- Initial clone + index time: <5 min for 10k files
- Incremental sync time: <30s for typical PR changes (10-50 files)
- Index size: <10% of codebase size
- Accuracy: >85% relevance for top-3 semantic search results
- PostgreSQL storage: ~2-3x codebase size (includes embeddings)
- Sync frequency: Every 15 minutes (configurable)
- Memory usage: <500MB resident (without GPU)

## Deployment Architecture

```
┌─────────────────────┐
│   GitHub Repo       │
│   (Remote)          │
└──────────┬──────────┘
           │
           │ git fetch/pull
           │ (incremental)
           ▼
┌─────────────────────┐      ┌──────────────────┐
│   Local Git Clone   │      │  PostgreSQL +    │
│   (Shallow/Blobless)│◄────►│  pgvector        │
└──────────┬──────────┘      └────────┬─────────┘
           │                          │
           │                          │
           ▼                          ▼
┌─────────────────────┐      ┌──────────────────┐
│  MCP Server         │◄────►│  Embedding Model │
│  (FastAPI/Python)   │      │  (sentence-trans)│
└──────────┬──────────┘      └──────────────────┘
           │
           │ MCP Protocol
           ▼
┌─────────────────────┐
│   Claude Desktop    │
│   or API Client     │
└─────────────────────┘
```