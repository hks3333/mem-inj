"""
Robust Checkpoint 3 Runner.
- Keeps Batching (Speed)
- Fixes Groq 400 Error (Safety)
- Handles missing Environment Variables
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import string
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Imports & Setup ---
try:
    from groq import Groq, BadRequestError
except ImportError:
    Groq = None
    BadRequestError = Exception

LOGGER = logging.getLogger("checkpoint3")

def configure_logging(verbose: bool) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    # Silence standard libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

def _load_env() -> None:
    """Robustly load .env file."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Manual fallback if python-dotenv isn't installed
    if not os.getenv("GROQ_API_KEY"):
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip("'").strip('"'))

_load_env()

# --- Helper Functions ---

def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = text.translate(str.maketrans({ch: " " for ch in string.punctuation}))
    return re.sub(r"\s+", " ", text)

def parse_alias_field(value: Any) -> List[str]:
    """Parse a single alias field (string or JSON list)."""
    if pd.isna(value) or value is None:
        return []
    value = str(value).strip()
    if not value:
        return []
    try:
        decoded = json.loads(value)
        if isinstance(decoded, list):
            flat = []
            for item in decoded:
                flat.extend(parse_alias_field(item))
            return flat
    except json.JSONDecodeError:
        pass
    if "|" in value:
        return [part.strip() for part in value.split("|") if part.strip()]
    return [value]

def parse_hop_ground_truths(value: Any) -> List[List[str]]:
    """Parse hop_ground_truths: JSON array of strings, each parsed as aliases."""
    if pd.isna(value) or value is None:
        return []
    value = str(value).strip()
    if not value:
        return []
    try:
        decoded = json.loads(value)
        if isinstance(decoded, list):
            return [parse_alias_field(item) for item in decoded]
    except json.JSONDecodeError:
        pass
    return []

def contains_any(text: str, aliases: Sequence[str]) -> bool:
    norm_text = normalize_text(text)
    for alias in aliases:
        norm_alias = normalize_text(alias)
        if norm_alias and norm_alias in norm_text:
            return True
    return False

def is_refusal(text: str) -> bool:
    patterns = [
        r"i'm sorry", r"i cannot", r"i can't", r"not able",
        r"as an ai", r"do not know", r"no information",
        r"cannot answer"
    ]
    return any(re.search(p, text.lower()) for p in patterns)

def classify_answer(answer: str, final_aliases: Sequence[str], hop_aliases: List[List[str]]) -> str:
    """
    Classify answer into: success, F1_refusal, F2_partial, F3_other.
    F2_partial: contains intermediate hop answers but NOT the final answer.
    """
    if contains_any(answer, final_aliases):
        return "success"
    if is_refusal(answer):
        return "F1_refusal"
    # F2: contains intermediate hops (all but last) but not final answer
    if hop_aliases and len(hop_aliases) > 1:
        intermediate_hops = hop_aliases[:-1]  # All hops except the last
        if any(contains_any(answer, hop) for hop in intermediate_hops):
            return "F2_partial"
    return "F3_other"

# --- Model Loading & Batch Generation ---

def load_model(model_id: str) -> tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    # Smart Device Selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    LOGGER.info(f"Loading {model_id} on {device} ({dtype})...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"  # Required for batch generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None
    )
    if device.type != "cuda":
        model.to(device)
    
    model.eval()
    return tokenizer, model, device

def generate_batch(
    prompts: List[str], 
    tokenizer: AutoTokenizer, 
    model: AutoModelForCausalLM, 
    device: torch.device, 
    max_tokens: int
) -> List[str]:
    if not prompts:
        return []
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # Greedy decoding (deterministic)
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Slice strictly the new tokens
    input_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[:, input_len:]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    
    return [d.strip() for d in decoded]

# --- Groq Judge Logic (Fixed) ---

def call_groq_judge(
    client: Groq,
    question: str,
    expected: str,
    intermediate_facts: Sequence[str],
    final_fact: str,
    answer: str
) -> Dict[str, str]:
    
    system_prompt = (
        "You are a multi-hop QA evaluator. Classify model answers using ONLY these labels:\n\n"
        "- success: Model answer matches or semantically equals the FINAL ground truth.\n"
        "- F1_refusal: Model refuses to answer (e.g., 'I don't know', 'I cannot answer').\n"
        "- F2_partial: Model contains intermediate hop facts but FAILS on the final answer.\n"
        "- F3_other: Model answer is wrong, irrelevant, or nonsensical.\n\n"
        "Output ONLY valid JSON: {\"final_label\": \"<label>\", \"reason\": \"<brief explanation>\"}"
    )
    
    user_prompt = (
        f"Question: {question}\n\n"
        f"Intermediate Hop Facts (should appear in answer for F2): {list(intermediate_facts)}\n"
        f"Final Ground Truth (required for success): {final_fact}\n\n"
        f"Model Answer: {answer}\n\n"
        "Classify this answer."
    )

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)

    except BadRequestError as e:
        # This catches the 400 "json_validate_failed" error
        LOGGER.warning(f"Groq 400 Error (Likely Safety/Refusal): {e}")
        return {"final_label": "F3_other", "reason": "Groq API Refusal/Format Error"}
    except Exception as e:
        LOGGER.error(f"Groq System Error: {e}")
        return {"final_label": "UNKNOWN", "reason": str(e)}

# --- Main Pipeline ---

def run_evaluation(args):
    configure_logging(args.verbose)
    
    # 1. Load Data
    df = pd.read_csv(args.dataset)
    if args.sample_size:
        df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
    
    # 2. Load Model
    tokenizer, model, device = load_model(args.model_id)
    
    # 3. Setup Groq
    groq_key = os.getenv("GROQ_API_KEY")
    groq_client = None
    if groq_key and Groq:
        groq_client = Groq(api_key=groq_key)
        LOGGER.info("Groq Judge: ENABLED")
    else:
        LOGGER.warning("Groq Judge: DISABLED (Check API Key)")

    results = []
    stats = {"total": 0, "success": 0, "F1_refusal": 0, "F2_partial": 0, "F3_other": 0, "groq_errors": 0}

    # 4. Run Batch Inference
    batch_size = args.batch_size
    for i in tqdm(range(0, len(df), batch_size), desc="Evaluating Batches"):
        batch_df = df.iloc[i : i + batch_size]
        prompts = batch_df["full_question"].astype(str).tolist()
        
        # A. Batch Generate (Local GPU)
        answers = generate_batch(prompts, tokenizer, model, device, args.max_tokens)
        
        # B. Process Results (Sequential Logic)
        for idx, model_ans in enumerate(answers):
            row = batch_df.iloc[idx]
            final_gt = parse_alias_field(row.get("final_ground_truth"))
            hop_gt = parse_hop_ground_truths(row.get("hop_ground_truths"))
            
            # Step 1: Local Classification (using hop ground truths)
            label = classify_answer(model_ans, final_gt, hop_gt)
            reason = "Local classification"

            # Step 2: Groq Judge (Only if local classification is failure)
            if label != "success" and groq_client:
                # Separate intermediate hops from final answer
                intermediate_hops = []
                if hop_gt and len(hop_gt) > 1:
                    intermediate_hops = [alias for hop in hop_gt[:-1] for alias in hop]
                final_answer = str(final_gt) if final_gt else ""
                
                judge_res = call_groq_judge(
                    groq_client, 
                    prompts[idx], 
                    final_answer,
                    intermediate_hops,
                    final_answer,
                    model_ans
                )
                judge_label = judge_res.get("final_label", "F3_other")
                if judge_label in {"success", "F1_refusal", "F2_partial", "F3_other"}:
                    label = judge_label
                    reason = judge_res.get("reason", "Judge decision")
            
            # Update Stats
            stats["total"] += 1
            if label in stats: 
                stats[label] += 1
            
            results.append({
                "row_id": row.get("row_id"),
                "question": prompts[idx],
                "answer": model_ans,
                "label": label,
                "reason": reason
            })

    # 5. Save Results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output_dir / "baseline_results.csv", index=False)
    
    with open(args.output_dir / "baseline_summary.json", "w") as f:
        json.dump(stats, f, indent=2)
        
    LOGGER.info(f"Finished. Summary: {stats}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoint3_artifacts"))
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    run_evaluation(args)