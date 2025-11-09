import os

from sentence_transformers.util import semantic_search

from config_loader import load_default_config
from dependency_graph_builder import DependencyGraphBuilder
from git_crawler import Crawler
from github_actions import clone_repo, pull_repo_diff
from semantic_chuker import SemanticChunker
from symbol_extractor import SymbolCrawler

if __name__ == "__main__":
    # Load configuration
    config = load_default_config()
    repo_cfg = config["repository"]

    repo_url = repo_cfg["url"]
    clone_dir = repo_cfg["local_path"]
    branch = repo_cfg.get("branch")

    # Ensure parent directories exist
    os.makedirs(os.path.dirname(clone_dir), exist_ok=True)

    # Clone or reuse existing repo
    repo = clone_repo(repo_url, clone_dir, branch)

    # Pull latest changes
    pull_repo_diff(repo, branch)
    crawler = Crawler(config, dry_run=False, no_embedding=False)
    crawler.run_once()

    symbol_crawler = SymbolCrawler(config)
    symbol_crawler.run()

    dependency_graph_builder = DependencyGraphBuilder(config)
    dependency_graph_builder.run()

    semantic_chunker = SemanticChunker(config)
    semantic_chunker.run()
