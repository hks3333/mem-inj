"""
Checkpoint 6: Back-Patching Diagnostic

This module integrates the manual hook-based back-patching prototype from
`backshots.py` into the project pipeline. It identifies late-emerging
intermediate answers (via a lightweight logit lens pass) and attempts to repair
failures by copying the late-layer residual stream into earlier layers before
final decoding. Results are summarised per (source_layer, target_layer) pair.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger("checkpoint6")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CaseSample:
    row_id: str
    label: str
    question: str
    intermediate_answer: str
    final_answers: List[str]
    baseline_answer: str
    baseline_success: bool
    intermediate_token_ids: List[int]
    intermediate_token_texts: List[str]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def configure_logging(verbose: bool) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def preferred_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        major = torch.cuda.get_device_capability(0)[0]
        return torch.float16 if major >= 7 else torch.float32
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def load_base_model(model_id: str, revision: Optional[str]) -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    dtype = preferred_dtype()
    LOGGER.info("Loading tokenizer for %s ...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        LOGGER.info("Tokenizer lacks pad_token -> setting pad_token to eos_token (%s)", tokenizer.eos_token)
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading model %s (dtype=%s)...", model_id, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        target_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(target_device)
    model.eval()
    device = next(model.parameters()).device
    LOGGER.info("Model loaded on %s (%.2fM params)", device, sum(p.numel() for p in model.parameters()) / 1e6)
    return tokenizer, model, device


def parse_jsonish_list(value: Optional[str]) -> List:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except json.JSONDecodeError:
        return []


def _split_alias_string(text: str) -> List[str]:
    parts = [part.strip() for part in text.split("|")]
    return [part for part in parts if part]


def flatten_aliases(seq: Iterable) -> List[str]:
    flat: List[str] = []
    for item in seq:
        if isinstance(item, (list, tuple, set)):
            flat.extend(flatten_aliases(item))
        elif isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                if "|" in cleaned:
                    flat.extend(_split_alias_string(cleaned))
                else:
                    flat.append(cleaned)
    return flat


def select_intermediate_answer(hop_ground_truths: List) -> Optional[str]:
    if not hop_ground_truths:
        return None
    first_hop = hop_ground_truths[0]
    if isinstance(first_hop, str):
        if "|" in first_hop:
            aliases = _split_alias_string(first_hop)
            return aliases[0] if aliases else None
        candidate = first_hop.strip()
        return candidate or None
    if isinstance(first_hop, (list, tuple, set)):
        flat = flatten_aliases(first_hop)
        return flat[0] if flat else None
    return None


def encode_intermediate_tokens(tokenizer: AutoTokenizer, answer_text: str) -> Optional[Tuple[List[int], List[str]]]:
    token_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    if not token_ids:
        return None
    token_texts = [tokenizer.decode([tok_id]).strip() for tok_id in token_ids]
    return token_ids, token_texts


def parse_final_answers(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return flatten_aliases(raw)
    flattened = flatten_aliases(parse_jsonish_list(raw))
    if flattened:
        return flattened
    text = str(raw).strip()
    if not text:
        return []
    if "|" in text:
        return _split_alias_string(text)
    return [text]


def normalize_answer(text: str) -> str:
    import re
    import string

    text = text.lower().strip()
    text = text.translate(str.maketrans({ch: " " for ch in string.punctuation}))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def answer_matches(prediction: str, expected_aliases: Sequence[str]) -> bool:
    if not prediction:
        return False
    normalized_pred = normalize_answer(prediction)
    if not normalized_pred:
        return False
    for alias in expected_aliases:
        if not alias:
            continue
        normalized_alias = normalize_answer(alias)
        if not normalized_alias:
            continue
        if normalized_alias == normalized_pred:
            return True
        if normalized_alias in normalized_pred:
            return True
    return False


def compute_rank(logits: torch.Tensor, token_id: int) -> int:
    target_logit = logits[token_id]
    rank = int((logits > target_logit).sum().item()) + 1
    return rank


def apply_final_norm(model: AutoModelForCausalLM, hidden_vec: torch.Tensor) -> torch.Tensor:
    core_model = getattr(model, "model", model)
    norm_layer = getattr(core_model, "norm", None)
    if norm_layer is None:
        return hidden_vec
    vec = hidden_vec.to(device=norm_layer.weight.device, dtype=norm_layer.weight.dtype)
    if vec.dim() == 1:
        return norm_layer(vec.unsqueeze(0)).squeeze(0)
    return norm_layer(vec)


def load_failure_cases(
    dataset_path: Path,
    baseline_path: Path,
    target_labels: List[str],
    sample_size: Optional[int],
    seed: int,
) -> pd.DataFrame:
    data_df = pd.read_csv(dataset_path)
    baseline_df = pd.read_csv(baseline_path)

    data_df["row_id"] = data_df["row_id"].astype(str)
    baseline_df["row_id"] = baseline_df["row_id"].astype(str)

    baseline_df = baseline_df.rename(columns={
        "question": "baseline_question",
        "answer": "baseline_answer",
    })
    data_df = data_df.rename(columns={
        "full_question": "composite_question",
        "final_ground_truth": "final_ground_truth_raw",
        "hop_ground_truths": "hop_ground_truths_raw",
    })

    merged = baseline_df.merge(data_df, on="row_id", how="inner", suffixes=("_baseline", ""))
    filtered = merged[merged["label"].isin(target_labels)].copy()

    if filtered.empty:
        raise RuntimeError("No matching cases found for requested labels")

    if sample_size is not None and sample_size < len(filtered):
        filtered = filtered.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    else:
        filtered = filtered.reset_index(drop=True)
    LOGGER.info("Prepared %d rows for back-patching analysis", len(filtered))
    return filtered


def prepare_cases(df: pd.DataFrame, tokenizer: AutoTokenizer) -> List[CaseSample]:
    samples: List[CaseSample] = []
    for _, row in df.iterrows():
        hop_truths = parse_jsonish_list(row.get("hop_ground_truths_raw"))
        intermediate = select_intermediate_answer(hop_truths)
        if not intermediate:
            LOGGER.debug("Skipping row %s: no intermediate answer", row["row_id"])
            continue

        encoded = encode_intermediate_tokens(tokenizer, intermediate)
        if not encoded:
            LOGGER.debug("Skipping row %s: intermediate answer tokenisation failed", row["row_id"])
            continue
        token_ids, token_texts = encoded

        final_answers = parse_final_answers(row.get("final_ground_truth_raw"))
        if not final_answers:
            LOGGER.debug("Skipping row %s: final answers missing", row["row_id"])
            continue

        baseline_answer = row.get("baseline_answer", "")
        if isinstance(baseline_answer, float):
            baseline_answer = ""
        baseline_answer = str(baseline_answer)

        baseline_success = answer_matches(baseline_answer, final_answers)

        samples.append(
            CaseSample(
                row_id=row["row_id"],
                label=row["label"],
                question=row["composite_question"],
                intermediate_answer=intermediate,
                final_answers=final_answers,
                baseline_answer=baseline_answer,
                baseline_success=baseline_success,
                intermediate_token_ids=token_ids,
                intermediate_token_texts=token_texts,
            )
        )
    if not samples:
        raise RuntimeError("No usable samples after preprocessing")
    LOGGER.info("Prepared %d case samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# Diagnostic computations (logit lens style)
# ---------------------------------------------------------------------------


def compute_layer_diagnostics(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    inputs: Dict[str, torch.Tensor],
    intermediate_token_ids: Sequence[int],
) -> Tuple[List[int], List[float], List[int], List[torch.Tensor]]:
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden_states = [tensor.detach() for tensor in outputs.hidden_states]

    lm_head = model.lm_head
    best_ranks: List[int] = []
    best_logits: List[float] = []
    best_indices: List[int] = []

    for layer_idx in range(1, len(hidden_states)):
        layer_hidden = hidden_states[layer_idx][0, -1, :]
        layer_hidden = apply_final_norm(model, layer_hidden)
        logits = torch.matmul(layer_hidden.to(lm_head.weight.device), lm_head.weight.T)
        token_logits = logits[intermediate_token_ids]
        token_ranks = [compute_rank(logits, tok_id) for tok_id in intermediate_token_ids]
        best_idx = min(range(len(token_ranks)), key=lambda i: token_ranks[i])
        best_ranks.append(token_ranks[best_idx])
        best_logits.append(token_logits[best_idx].item())
        best_indices.append(best_idx)

    return best_ranks, best_logits, best_indices, hidden_states


# ---------------------------------------------------------------------------
# Back-patching execution (manual hook approach)
# ---------------------------------------------------------------------------


def run_backpatch_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    inputs: Dict[str, torch.Tensor],
    source_memory: torch.Tensor,
    target_layer_idx: int,
    max_new_tokens: int,
) -> str:
    """Inject the late-layer residual into an earlier layer during generation."""

    handles = []
    source_last = source_memory[:, -1:, :].detach()

    def set_activation_hook(module, module_input, module_output):  # noqa: ANN001
        if isinstance(module_output, tuple):
            hidden = module_output[0]
        else:
            hidden = module_output
        replacement = hidden.clone()
        source_slice = source_last.to(device=hidden.device, dtype=hidden.dtype)
        if replacement.dim() == 3:
            replacement[:, -1:, :] = source_slice
        elif replacement.dim() == 2:
            replacement[:, :] = source_slice.squeeze(1)
        else:
            replacement = source_slice.squeeze()
        if isinstance(module_output, tuple):
            return (replacement,) + module_output[1:]
        return replacement

    hook = model.model.layers[target_layer_idx].register_forward_hook(set_activation_hook)
    handles.append(hook)

    try:
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        completion_ids = generated_ids[0, prompt_len:]
        patched_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        return patched_text
    finally:
        for handle in handles:
            handle.remove()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def parse_target_layers(arg_value: str) -> List[int]:
    if not arg_value:
        return []
    try:
        layers = [int(item.strip()) for item in arg_value.split(",") if item.strip()]
        return sorted(set(layer for layer in layers if layer >= 0))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid target layer list: {arg_value}") from exc


def run_backpatch_analysis(args: argparse.Namespace) -> None:
    configure_logging(args.verbose)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_labels = [label.strip() for label in args.target_labels.split(",") if label.strip()]
    if not target_labels:
        raise ValueError("No target labels provided")

    target_layers = parse_target_layers(args.target_layers)
    if not target_layers:
        raise ValueError("No --target-layers specified (comma-separated list required)")

    df = load_failure_cases(
        dataset_path=args.dataset,
        baseline_path=args.baseline_results,
        target_labels=target_labels,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    tokenizer, model, device = load_base_model(args.model_id, args.revision)

    cases = prepare_cases(df, tokenizer)
    result_rows: List[Dict[str, object]] = []

    for case in tqdm(cases, desc="Back-patching cases"):
        tokenised = tokenizer(case.question, return_tensors="pt")
        tokenised = {k: v.to(device) for k, v in tokenised.items()}

        best_ranks, best_logits, best_indices, hidden_states = compute_layer_diagnostics(
            model,
            tokenizer,
            tokenised,
            case.intermediate_token_ids,
        )

        source_layer = min(range(len(best_ranks)), key=lambda idx: best_ranks[idx])
        source_rank = best_ranks[source_layer]
        source_token_choice = best_indices[source_layer]
        source_token_text = case.intermediate_token_texts[source_token_choice]
        source_memory = hidden_states[source_layer + 1]  # +1 to skip embedding layer

        for target_layer in target_layers:
            if source_layer <= target_layer:
                continue

            gen_inputs = {k: v.clone() for k, v in tokenised.items()}
            patched_text = run_backpatch_generation(
                model,
                tokenizer,
                gen_inputs,
                source_memory,
                target_layer,
                args.max_new_tokens,
            )
            patched_success = answer_matches(patched_text, case.final_answers)
            improvement = int(patched_success and not case.baseline_success)

            result_rows.append({
                "row_id": case.row_id,
                "label": case.label,
                "baseline_answer": case.baseline_answer,
                "baseline_success": case.baseline_success,
                "intermediate_answer": case.intermediate_answer,
                "final_answers": case.final_answers,
                "intermediate_token_ids": case.intermediate_token_ids,
                "intermediate_token_texts": case.intermediate_token_texts,
                "source_layer": source_layer,
                "source_rank": source_rank,
                "source_token_text": source_token_text,
                "target_layer": target_layer,
                "patched_answer": patched_text,
                "patched_success": patched_success,
                "improvement": improvement,
                "best_ranks": best_ranks,
                "best_logits": best_logits,
                "best_indices": best_indices,
            })

    if not result_rows:
        LOGGER.error("No back-patching runs completed")
        return

    results_df = pd.DataFrame(result_rows)
    results_df["final_answers_json"] = results_df["final_answers"].apply(json.dumps)
    results_df["intermediate_token_ids_json"] = results_df["intermediate_token_ids"].apply(json.dumps)
    results_df["intermediate_token_texts_json"] = results_df["intermediate_token_texts"].apply(json.dumps)
    results_df["best_ranks_json"] = results_df["best_ranks"].apply(json.dumps)
    results_df["best_logits_json"] = results_df["best_logits"].apply(json.dumps)
    results_df["best_indices_json"] = results_df["best_indices"].apply(json.dumps)

    export_columns = [
        "row_id",
        "label",
        "baseline_answer",
        "baseline_success",
        "intermediate_answer",
        "final_answers_json",
        "source_layer",
        "source_rank",
        "source_token_text",
        "target_layer",
        "patched_answer",
        "patched_success",
        "improvement",
        "intermediate_token_ids_json",
        "intermediate_token_texts_json",
        "best_ranks_json",
        "best_logits_json",
        "best_indices_json",
    ]

    results_path = args.output_dir / "backpatch_results.csv"
    results_df.to_csv(results_path, index=False, columns=export_columns)
    LOGGER.info("Saved detailed results -> %s", results_path)

    overall = {
        "total_cases": len(cases),
        "baseline_successes": int(sum(case.baseline_success for case in cases)),
        "patched_successes": int(results_df["patched_success"].sum()),
        "new_success_cases": int(results_df.groupby("row_id")["patched_success"].max().sum()),
    }

    per_target = {}
    for target_layer, group in results_df.groupby("target_layer"):
        per_target[int(target_layer)] = {
            "runs": len(group),
            "patched_successes": int(group["patched_success"].sum()),
            "improvements": int(group["improvement"].sum()),
        }

    summary = {
        "overall": overall,
        "per_target_layer": per_target,
        "target_layers": target_layers,
        "target_labels": target_labels,
        "model_id": args.model_id,
    }

    summary_path = args.output_dir / "backpatch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    LOGGER.info("Saved summary -> %s", summary_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpoint 6: Back-Patching Diagnostic")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B", help="Hugging Face model ID")
    parser.add_argument("--revision", default=None, help="Optional model revision or commit hash")
    parser.add_argument("--dataset", type=Path, default=Path("checkpoint2_artifacts/combined_success.csv"))
    parser.add_argument("--baseline-results", type=Path, default=Path("checkpoint3_artifacts/baseline_results.csv"))
    parser.add_argument("--target-labels", default="F2_partial,F3_other", help="Comma separated baseline labels to analyse")
    parser.add_argument("--target-layers", default="2,4,6,8", help="Comma separated list of early layers to inject into")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of failure cases to analyse (None => all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoint6_artifacts"))
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Tokens to decode after back-patching")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_backpatch_analysis(args)


if __name__ == "__main__":
    main()
