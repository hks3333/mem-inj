"""MuSiQue evaluation pipeline aligned with TwoHopFact workflow."""

import argparse
import csv
import json
import logging
import random
import re
import string
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from datasets import load_dataset

# ---------------- CONFIGURATION ----------------
DEFAULT_SAMPLE_SIZE = 0  # 0 -> full dataset
DEFAULT_MAX_WORKERS = 3
DEFAULT_MODEL = "llama3.2:1b"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 60
DEFAULT_DELAY = 0.1
DEFAULT_SHUFFLE = False
DEFAULT_SEED = 42
DEFAULT_MAX_HOPS = 2
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_OUTPUT_DIR = Path("results")
OUTPUT_BASENAME = "musique_eval"
SUCCESS_FILENAME_TEMPLATE = "musique_success_{timestamp}.csv"
FAILURE_FILENAME_TEMPLATE = "musique_failures_{timestamp}.csv"
F1_THRESHOLD = 0.8
# -----------------------------------------------

REFUSAL_PATTERNS = [
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "i cannot",
    "i can't",
    "unable to",
    "no information",
    "i have no",
    "not enough information",
    "can't find",
    "i'm sorry",
    "sorry, i",
    "cannot find",
    "no data",
    "what is your question",
]

REFUSAL_REGEXES = [
    re.compile(r"\bunknown\b", re.IGNORECASE),
    re.compile(r"\bnot\s+enough\s+(?:information|context)\b", re.IGNORECASE),
    re.compile(r"\b(?:cannot|can't)\s+(?:answer|help|find)\b", re.IGNORECASE),
    re.compile(r"\b(?:insufficient|missing)\s+(?:information|context)\b", re.IGNORECASE),
    re.compile(r"\b(?:no|without)\s+(?:knowledge|information)\b", re.IGNORECASE),
]

LOGGER = logging.getLogger(__name__)
_THREAD_LOCAL = threading.local()


class OllamaClient:
    """Connection-pooled Ollama client supporting chat fallback."""

    def __init__(self, *, base_url: str, model: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = self._build_session()

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=6, pool_maxsize=12, max_retries=2)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def generate(self, prompt: str) -> Optional[str]:
        payload_chat = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 128},
        }
        payload_generate = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 128},
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload_chat,
                timeout=self.timeout,
            )
            if response.status_code == 404:
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json=payload_generate,
                    timeout=self.timeout,
                )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ConnectionError(f"Ollama API error: {exc}") from exc

        result = response.json()

        if isinstance(result, dict):
            message = result.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
            for key in ("response", "text"):
                value = result.get(key)
                if isinstance(value, str):
                    return value.strip()

        LOGGER.error("Unexpected Ollama response format: %s", result)
        return None


def _get_thread_client(base_url: str, model: str, timeout: float) -> OllamaClient:
    client: Optional[OllamaClient] = getattr(_THREAD_LOCAL, "client", None)
    base = base_url.rstrip("/")
    if (
        client is None
        or client.base_url != base
        or client.model != model
        or client.timeout != timeout
    ):
        client = OllamaClient(base_url=base, model=model, timeout=timeout)
        _THREAD_LOCAL.client = client
    return client


def configure_logging(level: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"{OUTPUT_BASENAME}_{timestamp}.log"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    LOGGER.info("Logging initialized -> %s", log_file)


def create_prompt(question: str) -> str:
    return (
        "Answer the following question with ONLY the answer. Do not include any explanation, "
        "reasoning, or additional text.\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def substitute_placeholders(text: str, answers: Sequence[str]) -> str:
    def repl(match: re.Match[str]) -> str:
        idx = int(match.group(1)) - 1
        return answers[idx] if 0 <= idx < len(answers) else match.group(0)

    return re.sub(r"#(\d+)", repl, text)


def is_refusal_or_generic(text: Optional[str]) -> bool:
    if not text:
        return True
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    if len(normalized) <= 2:
        return True
    if any(pattern in normalized for pattern in REFUSAL_PATTERNS):
        return True
    return any(regex.search(normalized) for regex in REFUSAL_REGEXES)


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans({ch: " " for ch in string.punctuation}))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compare_answers(model_answer: str, ground_truth_answer: Any) -> bool:
    if is_refusal_or_generic(model_answer):
        return False

    ground_truths = ground_truth_answer if isinstance(ground_truth_answer, list) else [ground_truth_answer]
    normalized_model = normalize_answer(model_answer)

    for gt in ground_truths:
        if gt is None:
            continue
        gt_text = gt if isinstance(gt, str) else json.dumps(gt, ensure_ascii=False)
        normalized_gt = normalize_answer(gt_text)

        if not normalized_gt:
            continue

        if normalized_model == normalized_gt:
            return True
        if normalized_gt in normalized_model:
            return True
        if compute_f1(model_answer, gt_text) >= F1_THRESHOLD:
            return True

    return False


def extract_subquestions(row: Dict[str, Any], max_hops: int) -> List[Tuple[str, str]]:
    decompositions: List[Tuple[str, str]] = []
    raw = row.get("question_decomposition")

    if isinstance(raw, list):
        for item in raw[:max_hops]:
            if isinstance(item, dict):
                question = item.get("question", "").strip()
                answer = item.get("answer", "").strip()
                if question and answer:
                    decompositions.append((question, answer))
    return decompositions


def load_and_sample_dataset(
    sample_size: int,
    shuffle: bool,
    seed: int,
) -> Any:
    dataset = load_dataset("dgslibisey/MuSiQue", split="train")
    total_rows = len(dataset)
    if sample_size <= 0 or sample_size > total_rows:
        sample_size = total_rows

    indices = list(range(total_rows))
    if shuffle:
        random.Random(seed).shuffle(indices)

    LOGGER.info("Dataset size: %s rows", total_rows)
    LOGGER.info("Sampling %s rows (shuffle=%s)", sample_size, shuffle)
    return dataset.select(indices[:sample_size])


def process_single_row(
    row: Dict[str, Any],
    *,
    model: str,
    base_url: str,
    timeout: float,
    delay: float,
    max_hops: int,
) -> Dict[str, Any]:
    row_id = row.get("id") or row.get("question_id") or row.get("uid") or "<unknown>"
    LOGGER.info("Processing row %s", row_id)

    decompositions = extract_subquestions(row, max_hops)
    if not decompositions:
        return {"status": "failed", "row_id": row_id, "reason": "no_decomposition"}

    step_records: List[Dict[str, Any]] = []
    substitution_answers: List[str] = []

    for idx, (question, ground_truth) in enumerate(decompositions, start=1):
        substituted_question = substitute_placeholders(question, substitution_answers)
        prompt = create_prompt(substituted_question)
        client = _get_thread_client(base_url=base_url, model=model, timeout=timeout)
        response = client.generate(prompt)

        if response is None:
            return {
                "status": "failed",
                "row_id": row_id,
                "reason": "no_response",
                "step": idx,
                "question": substituted_question,
            }

        if not compare_answers(response, ground_truth):
            return {
                "status": "failed",
                "row_id": row_id,
                "reason": "incorrect_step",
                "step": idx,
                "question": substituted_question,
                "expected": ground_truth,
                "response": response,
            }

        step_records.append(
            {
                "step": idx,
                "question": substituted_question,
                "ground_truth": ground_truth,
                "response": response,
            }
        )
        substitution_answers.append(ground_truth)
        time.sleep(max(delay, 0))

    return {
        "status": "success",
        "row_id": row_id,
        "full_question": row.get("question", ""),
        "final_ground_truth": row.get("answer", ""),
        "steps": step_records,
    }


def evaluate_dataset(
    sample_size: int,
    max_workers: int,
    model: str,
    base_url: str,
    timeout: float,
    delay: float,
    shuffle: bool,
    seed: int,
    max_hops: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dataset = load_and_sample_dataset(sample_size, shuffle, seed)
    successes: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    LOGGER.info("Dispatching %s rows with %s workers", len(dataset), max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_row,
                row,
                model=model,
                base_url=base_url,
                timeout=timeout,
                delay=delay,
                max_hops=max_hops,
            ): row.get("id", idx)
            for idx, row in enumerate(dataset)
        }

        for future in as_completed(futures):
            row_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Row %s raised exception: %s", row_id, exc)
                failures.append({"status": "failed", "row_id": row_id, "reason": "exception", "details": str(exc)})
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


def save_results(successes: List[Dict[str, Any]], failures: List[Dict[str, Any]], output_dir: Path) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    if successes:
        success_path = output_dir / SUCCESS_FILENAME_TEMPLATE.format(timestamp=timestamp)
        fieldnames = [
            "row_id",
            "full_question",
            "final_ground_truth",
        ]

        max_steps = max((len(item["steps"]) for item in successes), default=0)
        for idx in range(1, max_steps + 1):
            fieldnames.extend(
                [
                    f"step{idx}_question",
                    f"step{idx}_ground_truth",
                    f"step{idx}_response",
                ]
            )

        with success_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for item in successes:
                row_dict = {
                    "row_id": item["row_id"],
                    "full_question": item.get("full_question"),
                    "final_ground_truth": item.get("final_ground_truth"),
                }
                for idx, step in enumerate(item["steps"], start=1):
                    row_dict[f"step{idx}_question"] = step["question"]
                    row_dict[f"step{idx}_ground_truth"] = step["ground_truth"]
                    row_dict[f"step{idx}_response"] = step["response"]
                writer.writerow(row_dict)
        LOGGER.info("Saved %s successes -> %s", len(successes), success_path)
    else:
        LOGGER.info("No successful rows to save")

    if failures:
        failure_path = output_dir / FAILURE_FILENAME_TEMPLATE.format(timestamp=timestamp)
        fieldnames = sorted({key for failure in failures for key in failure.keys()})
        with failure_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(failures)
        LOGGER.info("Saved %s failures -> %s", len(failures), failure_path)
    else:
        LOGGER.info("No failures to save")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MuSiQue decompositions with a small LLM.")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Number of rows to process (0 -> all)")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Thread pool size")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="Ollama base URL")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Request timeout (seconds)")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Delay between per-row hops (seconds)")
    parser.add_argument("--shuffle", action="store_true", default=DEFAULT_SHUFFLE, help="Shuffle dataset before sampling")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Shuffle seed")
    parser.add_argument("--max-hops", type=int, default=DEFAULT_MAX_HOPS, help="Maximum decomposition steps per row")
    parser.add_argument("--log-level", type=str, default=DEFAULT_LOG_LEVEL, help="Logging level")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for CSV outputs/logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level, args.output_dir)

    LOGGER.info("Testing Ollama connectivity (model=%s)", args.model)
    test_client = _get_thread_client(base_url=args.base_url, model=args.model, timeout=args.timeout)
    if test_client.generate(create_prompt("What is 2+2?")) is None:
        LOGGER.error("Failed to communicate with Ollama. Ensure the server is running and the model is available.")
        return

    successes, failures = evaluate_dataset(
        sample_size=args.sample_size,
        max_workers=args.max_workers,
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
        delay=args.delay,
        shuffle=args.shuffle,
        seed=args.seed,
        max_hops=args.max_hops,
    )

    save_results(successes, failures, args.output_dir)

    total = len(successes) + len(failures)
    success_rate = (len(successes) / total * 100) if total else 0.0
    LOGGER.info(
        "Summary: %s processed | %s successes | %s failures | %.2f%% success rate",
        total,
        len(successes),
        len(failures),
        success_rate,
    )


if __name__ == "__main__":
    main()