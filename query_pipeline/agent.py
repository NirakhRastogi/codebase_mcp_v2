import json
import logging
import time
import re
import threading

import ollama
import requests
from typing import List, Dict, Any

from config_loader import load_config
from query_pipeline.query_processor import QueryProcessor

# --- Logger Setup ---
logger = logging.getLogger("QueryAgent")
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

# --- Colors for better CLI readability ---
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"


class QueryAgent:
    """
    🧠 Interactive, reasoning Query Agent for codebase insights.
    Features:
      - Contextual conversation (follows up based on previous responses)
      - Uses QueryProcessor + configurable LLM backend (Ollama or Cloud)
      - Iterative reasoning loop (multi-round)
    """

    def __init__(self, config_path: str):
        logger.info(f"{CYAN}🚀 Initializing Query Agent...{RESET}")
        self.config = load_config(config_path)
        self.processor = QueryProcessor(self.config)

        agent_conf = self.config.get("agent", {})
        self.mode = agent_conf.get("mode", "ollama").lower()
        self.llm_model = agent_conf.get("llm_model", "llama3")
        self.max_rounds = agent_conf.get("max_rounds", 3)
        self.history = []  # 🧠 Conversation memory

        if self.mode == "cloud":
            cloud_conf = agent_conf.get("cloud", {})
            self.api_url = cloud_conf.get("api_url")
            self.api_key = cloud_conf.get("api_key")
            self.model_id = cloud_conf.get("model_id")
            self.temperature = cloud_conf.get("temperature", 0.7)
            self.max_tokens = cloud_conf.get("max_tokens", 1024)
            logger.info(f"{CYAN}☁️ Using Cloud LLM backend ({self.model_id}) via {self.api_url}{RESET}")
        else:
            import ollama
            self.ollama = ollama
            ollama.pull('llama3')
            logger.info(f"{CYAN}🤖 Using Local Ollama backend ({self.llm_model}){RESET}")

    # --- LLM Helper ---
    def _call_llm(self, prompt: str, max_output: int = 512, timeout: int = 60) -> str:
        """
        Thread-safe, timeout-based LLM call for both Ollama and Cloud.
        """
        result = {"text": None, "error": None}

        def worker():
            try:
                if self.mode == "cloud":
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": self.model_id,
                        "prompt": prompt,
                        "temperature": self.temperature,
                        "max_tokens": max_output
                    }
                    response = requests.post(self.api_url, headers=headers, json=payload, timeout=timeout)
                    response.raise_for_status()
                    data = response.json()
                    result["text"] = (
                        data.get("completion")
                        or data.get("output")
                        or data.get("response")
                        or str(data)
                    ).strip()
                else:
                    response = self.ollama.chat(
                        model=self.llm_model,
                        messages=[{"role": "user", "content": prompt}],
                        options={"temperature": 0.2, "num_predict": max_output},
                    )
                    result["text"] = response["message"]["content"].strip()

            except Exception as e:
                result["error"] = str(e)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            logger.warning(f"{YELLOW}⚠️ LLM call timed out after {timeout}s.{RESET}")
            return "Timeout during model response."

        if result["error"]:
            logger.warning(f"{YELLOW}⚠️ Model error: {result['error']}{RESET}")
            return "Model error occurred."

        return result["text"] or "Empty model response."

    # --- Summarization of QueryProcessor Results ---
    def _summarize_results(self, results: Any) -> str:
        if not results:
            return "No relevant code or symbols found."

        if isinstance(results, dict):
            if "results" in results:
                results = results["results"]
            elif "data" in results:
                results = results["data"]
            else:
                return json.dumps(results, indent=2)

        if not isinstance(results, (list, tuple)):
            results = [results]

        summary_lines = []
        for r in results[:5]:
            if isinstance(r, dict):
                file_path = r.get("file_path") or r.get("path") or "unknown"
                symbol = r.get("symbol_name") or "N/A"
                snippet = (r.get("chunk_text") or r.get("content") or str(r)).replace("\n", " ")
                summary_lines.append(f"📄 {file_path} | 🔹 {symbol}: {snippet[:200]}")
            else:
                summary_lines.append(str(r))
        return "\n".join(summary_lines)

    # --- Decide Next Action ---
    def _extract_json(self, text: str) -> dict:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {"action": "finalize", "rationale": "no valid json"}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {"action": "finalize", "rationale": "invalid json"}

    def _decide_next_action(self, query: str, context_summary: str, round_num: int) -> dict:
        prompt = f"""
You are a reasoning code assistant analyzing retrieved code data.
Round: {round_num}

<QUERY>
{query}
</QUERY>

<RETRIEVED_DATA>
{context_summary[:1500]}
</RETRIEVED_DATA>

Respond ONLY in valid JSON:
{{
  "action": "refine_query" | "finalize",
  "next_query": "string (if refine_query)",
  "rationale": "short reasoning (max 15 words)"
}}

If unsure, finalize.
"""
        response = self._call_llm(prompt)
        return self._extract_json(response)

    # --- Main Reasoning Loop ---
    def run_query(self, user_query: str) -> str:
        logger.info(f"{CYAN}🧠 Understanding query:{RESET} {user_query}")
        query = user_query
        collected_contexts = []

        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"{GREEN}🔍 [Round {round_num}] Searching codebase...{RESET}")
            results = self.processor.process_query(query)
            summary = self._summarize_results(results)
            collected_contexts.append(summary)

            if not results:
                logger.info(f"{YELLOW}⚠️ No results found; finalizing early.{RESET}")
                break

            logger.info(f"{CYAN}💭 Thinking about context...{RESET}")
            decision = self._decide_next_action(query, summary, round_num)
            action = decision.get("action", "finalize")
            rationale = decision.get("rationale", "no rationale")
            logger.info(f"{GREEN}🤔 Decision: {action} ({rationale}){RESET}")

            if action == "finalize" or round_num == self.max_rounds:
                logger.info(f"{CYAN}🧩 Synthesizing final answer...{RESET}")
                combined_context = "\n".join(collected_contexts[-2:])
                final_prompt = f"""
You are an expert software analyst.

<USER_QUERY>
{user_query}
</USER_QUERY>

<CODE_CONTEXT>
{combined_context[:4000]}
</CODE_CONTEXT>

Answer based ONLY on this context (90%) and minimal reasoning (10%).
Mention file names, functions, or code references if relevant.
"""
                final_answer = self._call_llm(final_prompt, max_output=768)
                logger.info(f"{GREEN}✅ Finalized response generated.{RESET}")
                return final_answer

            if action == "refine_query":
                next_query = decision.get("next_query", "").strip()
                if next_query and next_query.lower() != query.lower():
                    logger.info(f"{CYAN}🔁 Refining query → {next_query}{RESET}")
                    query = next_query
                    time.sleep(1)
                else:
                    logger.info(f"{YELLOW}⚠️ No meaningful refinement. Stopping.{RESET}")
                    break

        return "No final answer could be synthesized."

    # --- Interactive CLI ---
    def chat(self):
        print(f"\n{CYAN}💬 Interactive Codebase Agent (type 'exit' to quit){RESET}\n")
        while True:
            user_query = input(f"{YELLOW}🔍 Ask your codebase:{RESET} ").strip()
            if not user_query or user_query.lower() in {"exit", "quit"}:
                print(f"{CYAN}👋 Goodbye!{RESET}")
                break

            # Add conversation history
            context_snippet = "\n".join([f"Q: {q}\nA: {a}" for q, a in self.history[-2:]])
            if context_snippet:
                user_query = (
                    f"The user previously asked:\n{context_snippet}\n\nNow they ask: {user_query}"
                )

            print(f"\n{CYAN}🧠 Thinking...\n{RESET}")
            answer = self.run_query(user_query)
            print(f"\n{GREEN}✅ Answer:\n{RESET}{answer}\n")

            self.history.append((user_query, answer))


if __name__ == "__main__":
    agent = QueryAgent("E://Projects//codebase_mcp_v2//config.yaml")
    agent.chat()
