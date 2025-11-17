"""Checkpoint 3 baseline reproduction: evaluate composite questions with HF model.

Example usage::

    python checkpoint3_baseline.py \
        --dataset checkpoint2_artifacts/combined_success.csv \
        --model-id meta-llama/Llama-3.2-1B \
        --sample-size 200 \
        --output-dir checkpoint3_artifacts

Outputs:
    * baseline_results.csv – per-row outcomes and failure taxonomy labels.
    * baseline_summary.json – aggregate counts for success and failure classes.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger("checkpoint3")

REFUSAL_PATTERNS = [
    r"i'm sorry",
    r"i cannot",
    r"i can't",
    r"cannot comply",
    r"as an ai",
    r"no information",
    r"i do not know",
    r"unknown",
]


@dataclass
class BaselineConfig:
    model_id: str
    revision: Optional[str]
    max_new_tokens: int
    temperature: float
    top_p: float
    sample_size: Optional[int]
    seed: int
    output_dir: Path


def configure_logging(verbose: bool) -> None:
    handler = logging.StreamHandler()
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)


def preferred_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 7:
            return torch.float16
        return torch.float32
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def load_hf_model(model_id: str, revision: Optional[str]) -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    dtype = preferred_dtype()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else ("mps" if torch.backends.mps.is_available() else "cpu"))

    LOGGER.info("Loading tokenizer for %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading model %s with dtype %s", model_id, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    return tokenizer, model, device


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans({ch: " " for ch in string.punctuation}))
    text = re.sub(r"\s+", " ", text)
    return text


def split_aliases(alias_blob: Optional[str]) -> List[str]:
    if alias_blob is None:
        return []
    alias_blob = str(alias_blob).strip()
    if not alias_blob:
        return []
    # Try JSON first
    try:
        decoded = json.loads(alias_blob)
        if isinstance(decoded, list):
            aliases = []
            for item in decoded:
                aliases.extend(split_aliases(item))
            return aliases
    except json.JSONDecodeError:
        pass
    if "|" in alias_blob:
        return [part.strip() for part in alias_blob.split("|") if part.strip()]
    return [alias_blob]


def contains_any(text: str, candidates: Sequence[str]) -> bool:
    if not text:
        return False
    norm_text = normalize_text(text)
    for cand in candidates:
        norm_cand = normalize_text(cand)
        if norm_cand and norm_cand in norm_text:
            return True
    return False


def is_refusal(text: str) -> bool:
    norm_text = text.lower()
    return any(re.search(pattern, norm_text) for pattern in REFUSAL_PATTERNS)


def load_dataset(path: Path) -> pd.DataFrame:
    LOGGER.info("Loading combined dataset from %s", path)
    df = pd.read_csv(path)
    required = {"dataset", "row_id", "full_question", "final_ground_truth", "hop_questions", "hop_ground_truths"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    def parse_json_list(value: str) -> List[str]:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            LOGGER.debug("Failed to json.loads hop field: %s", value[:80])
        return []

    df["hop_questions"] = df["hop_questions"].apply(parse_json_list)
    df["hop_ground_truths"] = df["hop_ground_truths"].apply(parse_json_list)
    df["hop_responses"] = df.get("hop_responses", pd.Series([[]] * len(df))).apply(parse_json_list)
    return df


def sample_dataset(df: pd.DataFrame, sample_size: Optional[int], seed: int) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(df):
        return df
    LOGGER.info("Sampling %d rows (seed=%d)", sample_size, seed)
    return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def final_aliases(raw_final: Optional[str], hop_ground_truths: Sequence[str]) -> List[str]:
    aliases = split_aliases(raw_final)
    if not aliases and hop_ground_truths:
        aliases = split_aliases(hop_ground_truths[-1])
    return aliases


def classify_failure(response: str, hop_aliases: List[List[str]], final_aliases_: List[str]) -> str:
    if is_refusal(response):
        return "F1_refusal"
    contains_intermediate = any(contains_any(response, aliases) for aliases in hop_aliases[:-1]) if hop_aliases else False
    contains_final = contains_any(response, final_aliases_)
    if contains_intermediate and not contains_final:
        return "F2_partial"
    return "F3_other"


def generate_answer(prompt: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, device: torch.device, cfg: BaselineConfig) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen_kwargs = dict(
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        do_sample=False,
    )
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()


def evaluate_rows(df: pd.DataFrame, cfg: BaselineConfig, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, device: torch.device) -> Tuple[pd.DataFrame, Dict[str, int]]:
    results: List[Dict[str, object]] = []
    counts = {"success": 0, "F1_refusal": 0, "F2_partial": 0, "F3_other": 0}

    for _, row in df.iterrows():
        dataset = row["dataset"]
        row_id = row["row_id"]
        prompt = row.get("full_question")
        if not isinstance(prompt, str) or not prompt.strip():
            LOGGER.debug("Skipping row %s: missing full_question", row_id)
            continue

        hop_gt_raw = row.get("hop_ground_truths") or []
        hop_aliases = [split_aliases(item) for item in hop_gt_raw]
        final_alias = final_aliases(row.get("final_ground_truth"), hop_gt_raw)

        model_answer = generate_answer(prompt, tokenizer, model, device, cfg)
        success = contains_any(model_answer, final_alias)
        if success:
            counts["success"] += 1
            failure_type = "success"
        else:
            failure_type = classify_failure(model_answer, hop_aliases, final_alias)
            counts[failure_type] += 1

        results.append(
            {
                "dataset": dataset,
                "row_id": row_id,
                "prompt": prompt,
                "final_ground_truth": row.get("final_ground_truth"),
                "model_answer": model_answer,
                "status": "success" if success else "failure",
                "failure_type": failure_type if not success else None,
            }
        )

    results_df = pd.DataFrame(results)
    return results_df, counts


def write_summary(summary: Dict[str, int], out_path: Path) -> None:
    total = sum(summary.values()) or 1
    summary_with_rates = {
        key: {
            "count": value,
            "rate": value / total,
        }
        for key, value in summary.items()
    }
    out_path.write_text(json.dumps(summary_with_rates, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpoint 3 baseline reproduction")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to combined_success.csv from checkpoint 2")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B", help="Hugging Face model identifier")
    parser.add_argument("--revision", default=None, help="Optional HF revision/commit")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional limit on number of rows to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoint3_artifacts"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BaselineConfig(
        model_id=args.model_id,
        revision=args.revision,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        sample_size=args.sample_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    configure_logging(args.verbose)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, device = load_hf_model(cfg.model_id, cfg.revision)
    df = load_dataset(args.dataset)
    df = sample_dataset(df, cfg.sample_size, cfg.seed)

    results_df, summary_counts = evaluate_rows(df, cfg, tokenizer, model, device)

    results_path = cfg.output_dir / "baseline_results.csv"
    summary_path = cfg.output_dir / "baseline_summary.json"

    results_df.to_csv(results_path, index=False)
    write_summary(summary_counts, summary_path)

    LOGGER.info("Saved results to %s", results_path)
    LOGGER.info("Saved summary to %s", summary_path)
    LOGGER.info("Successes: %d | F1_refusal: %d | F2_partial: %d | F3_other: %d", summary_counts.get("success", 0), summary_counts.get("F1_refusal", 0), summary_counts.get("F2_partial", 0), summary_counts.get("F3_other", 0))


if __name__ == "__main__":
    main()
