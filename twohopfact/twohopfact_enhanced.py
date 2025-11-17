import argparse
import ast
import csv
import logging
import random
import re
import threading
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset


LOGGER = logging.getLogger("twohopfact_enhanced")
_THREAD_LOCAL = threading.local()


class OllamaClient:
    """Thread-confined client for interacting with the Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:1b") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session = self._build_session()

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=4, max_retries=2)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def generate(self, prompt: str, timeout: int = 45) -> Optional[str]:
        """Generate a response from Ollama using the chat or generate endpoints."""
        payload_chat = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 64},
        }
        payload_generate = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 64},
        }

        try:
            response = self.session.post(f"{self.base_url}/api/chat", json=payload_chat, timeout=timeout)
            if response.status_code == 404:
                response = self.session.post(f"{self.base_url}/api/generate", json=payload_generate, timeout=timeout)

            response.raise_for_status()
            result = response.json()

            if isinstance(result, dict):
                if "message" in result and isinstance(result["message"], dict):
                    content = result["message"].get("content")
                    if isinstance(content, str):
                        return content.strip()
                if "response" in result and isinstance(result["response"], str):
                    return result["response"].strip()

            LOGGER.error("Unexpected Ollama response format: %s", result)
            return None
        except requests.exceptions.Timeout:
            LOGGER.warning("Timeout while querying Ollama")
            return None
        except requests.exceptions.RequestException as exc:
            LOGGER.error("Request failure: %s", exc)
            return None


def _get_thread_client(model: str, base_url: str) -> OllamaClient:
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None or client.model != model or client.base_url != base_url.rstrip("/"):
        client = OllamaClient(base_url=base_url, model=model)
        _THREAD_LOCAL.client = client
    return client


def create_master_prompt(question: str) -> str:
    return (
        "Answer the following question with ONLY the answer. Do not include any explanation, reasoning, or additional text.\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def _flatten_aliases(values: Iterable) -> Iterable[str]:
    for value in values:
        if isinstance(value, (list, tuple, set)):
            yield from _flatten_aliases(value)
        elif isinstance(value, str):
            alias = value.strip()
            if alias:
                yield alias


def parse_aliases(alias_repr: str) -> List[str]:
    """Robustly parse alias strings stored in the dataset."""
    if not alias_repr:
        return []

    try:
        parsed = ast.literal_eval(alias_repr)
    except (ValueError, SyntaxError):
        parsed = alias_repr

    if isinstance(parsed, str):
        aliases: Sequence[str] = [parsed]
    elif isinstance(parsed, (list, tuple, set)):
        aliases = list(_flatten_aliases(parsed))
    else:
        aliases = []

    # Deduplicate while preserving order
    seen = set()
    result = []
    for alias in aliases:
        key = alias.casefold()
        if key not in seen:
            seen.add(key)
            result.append(alias)
    return result


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def check_answer_in_response(response: Optional[str], aliases: Sequence[str]) -> Tuple[bool, Optional[str]]:
    if not response:
        return False, None

    normalized_response = _normalize(response)
    padded_response = f" {normalized_response} "

    for alias in aliases:
        normalized_alias = _normalize(alias)
        if not normalized_alias:
            continue
        if f" {normalized_alias} " in padded_response:
            return True, alias
        # Fallback: allow substring match when alias is a reasonably long single token
        compact_alias = normalized_alias.replace(" ", "")
        if len(compact_alias) >= 4 and " " not in normalized_alias and compact_alias in normalized_response.replace(" ", ""):
            return True, alias
    return False, None


def process_single_row(row: Dict, *, model: str, base_url: str, delay_seconds: float) -> Dict:
    uid = row.get("uid", "<missing>")
    client = _get_thread_client(model=model, base_url=base_url)

    r1_prompt = row.get("r1(e1).prompt")
    r2_prompt = row.get("r2(e2).prompt")

    e2_aliases = parse_aliases(row.get("e2.aliases", ""))
    e3_aliases = parse_aliases(row.get("e3.aliases", ""))

    if not r1_prompt or not r2_prompt:
        return {"status": "failed", "uid": uid, "reason": "missing_prompts"}
    if not e2_aliases or not e3_aliases:
        return {"status": "failed", "uid": uid, "reason": "missing_aliases"}

    r1_prompt_full = create_master_prompt(r1_prompt)
    r1_response = client.generate(r1_prompt_full)

    if r1_response is None:
        return {"status": "failed", "uid": uid, "reason": "r1_no_response"}

    r1_ok, r1_alias = check_answer_in_response(r1_response, e2_aliases)
    if not r1_ok:
        return {
            "status": "failed",
            "uid": uid,
            "reason": "r1_alias_not_found",
            "response_excerpt": r1_response[:200],
        }

    time.sleep(delay_seconds)

    r2_prompt_full = create_master_prompt(r2_prompt)
    r2_response = client.generate(r2_prompt_full)

    if r2_response is None:
        return {"status": "failed", "uid": uid, "reason": "r2_no_response"}

    r2_ok, r2_alias = check_answer_in_response(r2_response, e3_aliases)
    if not r2_ok:
        return {
            "status": "failed",
            "uid": uid,
            "reason": "r2_alias_not_found",
            "response_excerpt": r2_response[:200],
        }

    return {
        "status": "success",
        "uid": uid,
        "e1_value": row.get("e1.value"),
        "e2_value": row.get("e2.value"),
        "e3_value": row.get("e3.value"),
        "r1_prompt": r1_prompt,
        "r1_expected_aliases": "|".join(e2_aliases),
        "r1_response": r1_response,
        "r1_matched_alias": r1_alias,
        "r2_prompt": r2_prompt,
        "r2_expected_aliases": "|".join(e3_aliases),
        "r2_response": r2_response,
        "r2_matched_alias": r2_alias,
        "fact_comp_type": row.get("fact_comp_type"),
        "category": row.get("category"),
    }


def evaluate_dataset(
    *,
    sample_size: Optional[int],
    max_workers: int,
    model: str,
    base_url: str,
    delay_seconds: float,
    seed: Optional[int],
    shuffle: bool,
) -> Tuple[List[Dict], List[Dict]]:
    LOGGER.info("Loading dataset ...")
    dataset = load_dataset("soheeyang/TwoHopFact", split="train")
    total_rows = len(dataset)
    LOGGER.info("Dataset size: %s rows", total_rows)

    if sample_size is not None:
        sample_size = min(sample_size, total_rows)
        if shuffle:
            if seed is not None:
                rng = random.Random(seed)
            else:
                rng = random.Random()
            indices = list(range(total_rows))
            rng.shuffle(indices)
            selected = indices[:sample_size]
            dataset = dataset.select(selected)
        else:
            dataset = dataset.select(range(sample_size))
        LOGGER.info("Using sample size: %s (%sshuffled)", sample_size, "" if shuffle else "not ")
    else:
        LOGGER.info("Processing the full dataset")

    work_items = list(dataset)
    successes: List[Dict] = []
    failures: List[Dict] = []

    LOGGER.info("Dispatching %s rows with up to %s workers", len(work_items), max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_row,
                row,
                model=model,
                base_url=base_url,
                delay_seconds=delay_seconds,
            ): row.get("uid", f"row_{idx}")
            for idx, row in enumerate(work_items)
        }

        for future in as_completed(futures):
            uid = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Exception while processing %s: %s", uid, exc)
                failures.append({"status": "failed", "uid": uid, "reason": "exception", "details": str(exc)})
                continue

            if result.get("status") == "success":
                successes.append(result)
            else:
                failures.append(result)

            processed = len(successes) + len(failures)
            if processed % 50 == 0:
                LOGGER.info(
                    "Progress: %s processed | %s successes | %s failures",
                    processed,
                    len(successes),
                    len(failures),
                )

    LOGGER.info("Processing complete: %s successes, %s failures", len(successes), len(failures))
    return successes, failures


def configure_logging(level: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"twohopfact_enhanced_{timestamp}.log"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )
    LOGGER.info("Logging initialized. Writing to %s", log_file)


def save_results(successes: List[Dict], failures: List[Dict], output_dir: Path) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    if successes:
        success_path = output_dir / f"twohopfact_success_{timestamp}.csv"
        fieldnames = [
            "uid",
            "e1_value",
            "e2_value",
            "e3_value",
            "r1_prompt",
            "r1_expected_aliases",
            "r1_response",
            "r1_matched_alias",
            "r2_prompt",
            "r2_expected_aliases",
            "r2_response",
            "r2_matched_alias",
            "fact_comp_type",
            "category",
        ]
        cleaned_successes = [
            {key: value for key, value in success.items() if key != "status"}
            for success in successes
        ]

        with success_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cleaned_successes)
        LOGGER.info("Saved %s successful rows to %s", len(successes), success_path)
    else:
        LOGGER.info("No successful rows to save.")

    if failures:
        failure_path = output_dir / f"twohopfact_failures_{timestamp}.csv"
        with failure_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = sorted({key for failure in failures for key in failure.keys()})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(failures)
        LOGGER.info("Saved %s failure rows to %s", len(failures), failure_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TwoHopFact with improved matching.")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of rows to process (default: all)")
    parser.add_argument("--max-workers", type=int, default=3, help="Number of concurrent workers")
    parser.add_argument("--model", type=str, default="llama3.2:1b", help="Ollama model name")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.2,
        help="Delay between R1 and R2 calls per sample to avoid server overload",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (ignored unless --shuffle)")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle dataset before sampling (default: False, use first N rows)",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where CSV outputs and logs will be stored",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level, args.output_dir)

    client = OllamaClient(base_url=args.base_url, model=args.model)
    LOGGER.info("Testing Ollama connection with model %s", args.model)
    test_prompt = create_master_prompt("What is 2+2?")
    if client.generate(test_prompt) is None:
        LOGGER.error("Failed to communicate with Ollama at %s", args.base_url)
        return
    LOGGER.info("Ollama connection looks good. Starting evaluation.")

    successes, failures = evaluate_dataset(
        sample_size=args.sample_size,
        max_workers=args.max_workers,
        model=args.model,
        base_url=args.base_url,
        delay_seconds=args.delay_seconds,
        seed=args.seed,
        shuffle=args.shuffle,
    )

    save_results(successes, failures, args.output_dir)

    total = len(successes) + len(failures)
    success_rate = (len(successes) / total * 100) if total else 0.0
    LOGGER.info("Summary: %s processed | %s successes | %s failures | %.2f%% success rate", total, len(successes), len(failures), success_rate)


if __name__ == "__main__":
    main()
