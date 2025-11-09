import os
import yaml
import requests
import logging
from git import Repo, GitCommandError

os.environ["PYTHONIOENCODING"] = "utf-8"

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# -------------------------------
# 🧩 Load configuration
# -------------------------------
def load_config(config_path: str = "config.yml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------------
# 🚀 Git repo utilities
# -------------------------------
def clone_repo(repo_url: str, clone_dir: str, branch: str) -> Repo:
    """Clone a Git repository if not already cloned."""
    if not os.path.exists(clone_dir):
        logger.info(f"🚀 Cloning {repo_url} into {clone_dir} (branch: {branch})...")
        try:
            repo = Repo.clone_from(repo_url, clone_dir, branch=branch)
            logger.info("✅ Repository cloned successfully.")
            return repo
        except GitCommandError as e:
            logger.error(f"❌ Failed to clone repository: {e}")
            raise
    else:
        logger.warning(f"⚠️ Repository already exists at {clone_dir}.")
        return Repo(clone_dir)


def pull_repo_diff(repo: Repo, branch: str) -> None:
    """Fetch and pull latest changes, printing diffs."""
    try:
        origin = repo.remotes.origin
        logger.info("🔄 Fetching latest changes...")
        origin.fetch()

        current_branch = repo.active_branch.name
        if current_branch != branch:
            logger.info(f"🔁 Switching from {current_branch} to {branch}...")
            repo.git.checkout(branch)

        diff = repo.git.diff(f"HEAD..origin/{branch}")
        if diff:
            logger.info("🧩 Changes to be pulled:\n%s", diff)
        else:
            logger.info("✅ Already up-to-date.")

        logger.info("⬇️ Pulling latest changes...")
        origin.pull()
        logger.info("✅ Repository updated successfully.")
    except GitCommandError as e:
        logger.error(f"❌ Git command failed: {e}")
    except Exception as e:
        logger.error(f"❌ Error: {e}")


# -------------------------------
# ☁️ Cloud agent call
# -------------------------------
def call_cloud_agent(prompt: str, config: dict) -> str:
    cloud_conf = config.get("cloud", {})
    api_url = cloud_conf.get("api_url")
    api_key = cloud_conf.get("api_key")
    model_id = cloud_conf.get("model_id")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_id,
        "prompt": prompt,
        "temperature": cloud_conf.get("temperature", 0.7),
        "max_tokens": cloud_conf.get("max_tokens", 1024)
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        # adapt to expected JSON structure
        return data.get("completion") or data.get("output") or str(data)
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Cloud agent call failed: {e}")
        return "⚠️ Cloud agent error."


# -------------------------------
# 🧠 Ollama local call
# -------------------------------
def call_local_ollama(prompt: str, config: dict) -> str:
    import ollama
    model = config.get("ollama", {}).get("model", "llama3")
    try:
        result = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return result["message"]["content"]
    except Exception as e:
        logger.error(f"❌ Ollama call failed: {e}")
        return "⚠️ Ollama error."


# -------------------------------
# 🧩 Query Agent
# -------------------------------
class QueryAgent:
    def __init__(self, config_path: str = "config.yml"):
        self.config = load_config(config_path)
        self.mode = self.config.get("mode", "ollama").lower()

    def run_query(self, prompt: str) -> str:
        logger.info(f"🤖 Running query in {self.mode.upper()} mode...")

        if self.mode == "cloud":
            return call_cloud_agent(prompt, self.config)
        elif self.mode == "ollama":
            return call_local_ollama(prompt, self.config)
        else:
            raise ValueError(f"❌ Unsupported mode in config: {self.mode}")


# -------------------------------
# 🔧 Example usage
# -------------------------------
if __name__ == "__main__":
    agent = QueryAgent("config.yml")
    response = agent.run_query("Explain the difference between Git fetch and Git pull.")
    print("\n🧠 Agent Response:\n", response)
