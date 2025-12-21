"""
Checkpoint 7: Memory Injection Mitigation

This script builds on the memory injection prototype in `mem_clean.py`, adapting
it to the project pipeline. For each multi-hop failure case, we derive a
normalized representation of the intermediate ("bridge") answer at a chosen
source layer, and inject that residual into earlier layers while generating the
composite answer. We compare baseline outputs with injected outputs, track
probability changes for the correct final answers, and report improvement
statistics across layer/strength sweeps.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger("checkpoint7")


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
    target_token_ids: List[int]
    baseline_answer: str
    baseline_label: str


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


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


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


def collect_target_token_ids(tokenizer: AutoTokenizer, answers: Sequence[str]) -> List[int]:
    token_ids: List[int] = []
    for alias in answers:
        if not alias:
            continue
        ids = tokenizer.encode(alias, add_special_tokens=False)
        if ids:
            token_ids.append(ids[0])
    return sorted(set(token_ids))


def normalize_answer(text: str) -> str:
    import re
    import string

    normalized = text.lower().strip()
    normalized = normalized.translate(str.maketrans({ch: " " for ch in string.punctuation}))
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


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
    LOGGER.info("Prepared %d rows for memory injection analysis", len(filtered))
    return filtered


def prepare_cases(df: pd.DataFrame, tokenizer: AutoTokenizer) -> List[CaseSample]:
    samples: List[CaseSample] = []
    for _, row in df.iterrows():
        hop_truths = parse_jsonish_list(row.get("hop_ground_truths_raw"))
        intermediate = select_intermediate_answer(hop_truths)
        if not intermediate:
            LOGGER.debug("Skipping row %s: no intermediate answer", row["row_id"])
            continue

        final_answers = parse_final_answers(row.get("final_ground_truth_raw"))
        if not final_answers:
            LOGGER.debug("Skipping row %s: final answers missing", row["row_id"])
            continue

        target_token_ids = collect_target_token_ids(tokenizer, final_answers)
        if not target_token_ids:
            LOGGER.debug("Skipping row %s: no target tokens for final answers", row["row_id"])
            continue

        baseline_answer = row.get("baseline_answer", "")
        if isinstance(baseline_answer, float):
            baseline_answer = ""
        baseline_answer = str(baseline_answer)

        samples.append(
            CaseSample(
                row_id=row["row_id"],
                label=row["label"],
                question=row["composite_question"],
                intermediate_answer=intermediate,
                final_answers=final_answers,
                target_token_ids=target_token_ids,
                baseline_answer=baseline_answer,
                baseline_label=row["label"],
            )
        )
    if not samples:
        raise RuntimeError("No usable samples after preprocessing")
    LOGGER.info("Prepared %d case samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# Memory injection helpers
# ---------------------------------------------------------------------------


def get_concept_vector(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    concept_text: str,
    layer_idx: int,
    device: torch.device,
) -> torch.Tensor:
    inputs = tokenizer(concept_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    target_index = layer_idx + 1  # account for embedding hidden state at index 0
    if target_index >= len(hidden_states):
        raise ValueError(f"Layer index {layer_idx} out of range for concept vector extraction")
    vec = hidden_states[target_index][:, -1, :]
    norm = vec.norm(dim=-1, keepdim=True)
    vec = vec / torch.clamp(norm, min=1e-8)
    return vec.squeeze(0)


def register_residual_injection(
    model: AutoModelForCausalLM,
    concept_vec: torch.Tensor,
    layer_idx: int,
    strength: float,
):
    if strength == 0.0:
        return None

    layer = model.model.layers[layer_idx]

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            tail = output[1:]
        else:
            hidden = output
            tail = ()

        hidden = hidden.clone()
        injection_vec = concept_vec.to(hidden.device, hidden.dtype)
        scale = hidden.norm(dim=-1, keepdim=True).mean()
        scaled_injection = injection_vec * scale * strength

        if hidden.dim() == 3:
            hidden[:, -1, :] = hidden[:, -1, :] + scaled_injection.unsqueeze(0)
        elif hidden.dim() == 2:
            hidden[:, :] = hidden[:, :] + scaled_injection
        else:
            hidden = hidden + scaled_injection

        if tail:
            return (hidden,) + tail
        return hidden

    return layer.register_forward_hook(hook)


def evaluate_target_probability(
    model: AutoModelForCausalLM,
    inputs: Dict[str, torch.Tensor],
    target_token_ids: Sequence[int],
    concept_vec: Optional[torch.Tensor] = None,
    layer_idx: Optional[int] = None,
    strength: float = 0.0,
) -> Tuple[float, float]:
    handle = None
    if concept_vec is not None and layer_idx is not None and strength != 0.0:
        handle = register_residual_injection(model, concept_vec, layer_idx, strength)
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        if not target_token_ids:
            return 0.0, 0.0
        selected = probs[list(target_token_ids)]
        max_prob = float(selected.max().item())
        sum_prob = float(selected.sum().item())
        return max_prob, sum_prob
    finally:
        if handle is not None:
            handle.remove()


def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    inputs: Dict[str, torch.Tensor],
    max_new_tokens: int,
    concept_vec: Optional[torch.Tensor] = None,
    layer_idx: Optional[int] = None,
    strength: float = 0.0,
) -> str:
    handle = None
    if concept_vec is not None and layer_idx is not None and strength != 0.0:
        handle = register_residual_injection(model, concept_vec, layer_idx, strength)
    try:
        gen_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "num_beams": 1,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        # Remove None attention_mask to avoid HF warnings
        if gen_kwargs["attention_mask"] is None:
            gen_kwargs.pop("attention_mask")

        with torch.no_grad():
            generated_ids = model.generate(**gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        completion_ids = generated_ids[0, prompt_len:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        return text
    finally:
        if handle is not None:
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


def parse_strengths(arg_value: str) -> List[float]:
    if not arg_value:
        return []
    try:
        strengths = [float(item.strip()) for item in arg_value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid strengths list: {arg_value}") from exc
    filtered = [value for value in strengths if value != 0.0]
    if not filtered:
        raise argparse.ArgumentTypeError("Strength list must contain at least one non-zero value")
    return sorted(filtered)


def run_memory_injection_analysis(args: argparse.Namespace) -> None:
    configure_logging(args.verbose)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_labels = [label.strip() for label in args.target_labels.split(",") if label.strip()]
    if not target_labels:
        raise ValueError("No target labels provided")

    target_layers = parse_target_layers(args.target_layers)
    strengths = parse_strengths(args.strengths)

    df = load_failure_cases(
        dataset_path=args.dataset,
        baseline_path=args.baseline_results,
        target_labels=target_labels,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    tokenizer, model, device = load_base_model(args.model_id, args.revision)

    num_layers = model.config.num_hidden_layers
    if not (0 <= args.source_layer < num_layers):
        raise ValueError(f"source_layer {args.source_layer} out of range (0-{num_layers - 1})")
    for layer in target_layers:
        if not (0 <= layer < num_layers):
            raise ValueError(f"target layer {layer} out of range (0-{num_layers - 1})")

    cases = prepare_cases(df, tokenizer)

    results: List[Dict[str, object]] = []
    baseline_success_count = 0
    recorded_baseline_success_count = 0

    combo_stats: Dict[Tuple[int, float], Dict[str, object]] = {}
    for layer in target_layers:
        for strength in strengths:
            combo_stats[(layer, strength)] = {
                "runs": 0,
                "injection_successes": 0,
                "improvements": 0,
                "prob_gain_pct": [],
                "prob_gain_abs": [],
            }

    for case in tqdm(cases, desc="Memory injection cases"):
        tokenised = tokenizer(case.question, return_tensors="pt")
        tokenised = {k: v.to(device) for k, v in tokenised.items()}

        baseline_text = generate_answer(
            model,
            tokenizer,
            tokenised,
            max_new_tokens=args.max_new_tokens,
        )
        baseline_success = answer_matches(baseline_text, case.final_answers)
        if baseline_success:
            baseline_success_count += 1

        recorded_baseline_success = answer_matches(case.baseline_answer, case.final_answers)
        if recorded_baseline_success:
            recorded_baseline_success_count += 1

        base_prob_max, base_prob_sum = evaluate_target_probability(
            model,
            tokenised,
            case.target_token_ids,
        )

        try:
            concept_vec = get_concept_vector(
                model,
                tokenizer,
                case.intermediate_answer,
                layer_idx=args.source_layer,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to compute concept vector for row %s: %s", case.row_id, exc)
            concept_vec = None

        for target_layer in target_layers:
            for strength in strengths:
                combo_key = (target_layer, strength)
                combo_stats[combo_key]["runs"] += 1

                if concept_vec is None:
                    inj_prob_max, inj_prob_sum = (float("nan"), float("nan"))
                    injected_text = ""
                    injected_success = False
                else:
                    inj_prob_max, inj_prob_sum = evaluate_target_probability(
                        model,
                        tokenised,
                        case.target_token_ids,
                        concept_vec=concept_vec,
                        layer_idx=target_layer,
                        strength=strength,
                    )
                    injected_text = generate_answer(
                        model,
                        tokenizer,
                        tokenised,
                        max_new_tokens=args.max_new_tokens,
                        concept_vec=concept_vec,
                        layer_idx=target_layer,
                        strength=strength,
                    )
                    injected_success = answer_matches(injected_text, case.final_answers)

                if injected_success:
                    combo_stats[combo_key]["injection_successes"] += 1
                improvement = injected_success and not baseline_success
                if improvement:
                    combo_stats[combo_key]["improvements"] += 1

                if not math.isnan(base_prob_max) and base_prob_max > 0 and not math.isnan(inj_prob_max):
                    gain_pct = ((inj_prob_max - base_prob_max) / base_prob_max) * 100.0
                    combo_stats[combo_key]["prob_gain_pct"].append(gain_pct)
                    combo_stats[combo_key]["prob_gain_abs"].append(inj_prob_max - base_prob_max)
                else:
                    gain_pct = float("nan")

                results.append({
                    "row_id": case.row_id,
                    "label": case.label,
                    "baseline_label": case.baseline_label,
                    "baseline_recorded_answer": case.baseline_answer,
                    "baseline_generated_answer": baseline_text,
                    "baseline_success": baseline_success,
                    "baseline_prob_max": base_prob_max,
                    "baseline_prob_sum": base_prob_sum,
                    "injected_answer": injected_text,
                    "injected_success": injected_success,
                    "injected_prob_max": inj_prob_max,
                    "injected_prob_sum": inj_prob_sum,
                    "prob_gain_pct": gain_pct,
                    "prob_gain_abs": (inj_prob_max - base_prob_max) if not math.isnan(inj_prob_max) else float("nan"),
                    "intermediate_answer": case.intermediate_answer,
                    "final_answers": case.final_answers,
                    "target_layer": target_layer,
                    "strength": strength,
                    "source_layer": args.source_layer,
                    "improvement": improvement,
                })

    if not results:
        LOGGER.error("No analyses completed successfully")
        return

    results_df = pd.DataFrame(results)
    results_df["final_answers_json"] = results_df["final_answers"].apply(json.dumps)

    export_columns = [
        "row_id",
        "label",
        "baseline_label",
        "baseline_recorded_answer",
        "baseline_generated_answer",
        "baseline_success",
        "baseline_prob_max",
        "baseline_prob_sum",
        "injected_answer",
        "injected_success",
        "injected_prob_max",
        "injected_prob_sum",
        "prob_gain_pct",
        "prob_gain_abs",
        "intermediate_answer",
        "final_answers_json",
        "source_layer",
        "target_layer",
        "strength",
        "improvement",
    ]

    results_path = args.output_dir / "memory_injection_results.csv"
    results_df.to_csv(results_path, index=False, columns=export_columns)
    LOGGER.info("Saved detailed results -> %s", results_path)

    overall = {
        "total_cases": len(cases),
        "baseline_successes_generated": baseline_success_count,
        "baseline_success_rate_generated": baseline_success_count / len(cases),
        "baseline_successes_recorded": recorded_baseline_success_count,
        "baseline_success_rate_recorded": recorded_baseline_success_count / len(cases),
    }

    per_combo = {}
    for (layer, strength), stats in combo_stats.items():
        runs = stats["runs"]
        success_count = stats["injection_successes"]
        improvement_count = stats["improvements"]

        gain_pct_values = stats["prob_gain_pct"]
        gain_abs_values = stats["prob_gain_abs"]

        avg_gain_pct = statistics.mean(gain_pct_values) if gain_pct_values else None
        median_gain_pct = statistics.median(gain_pct_values) if gain_pct_values else None
        avg_gain_abs = statistics.mean(gain_abs_values) if gain_abs_values else None

        per_combo[str((layer, strength))] = {
            "target_layer": layer,
            "strength": strength,
            "runs": runs,
            "injected_successes": success_count,
            "injected_success_rate": success_count / runs if runs else 0.0,
            "improvements": improvement_count,
            "improvement_rate": improvement_count / runs if runs else 0.0,
            "avg_prob_gain_pct": avg_gain_pct,
            "median_prob_gain_pct": median_gain_pct,
            "avg_prob_gain_abs": avg_gain_abs,
        }

    summary = {
        "overall": overall,
        "per_combo": per_combo,
        "source_layer": args.source_layer,
        "target_layers": target_layers,
        "strengths": strengths,
        "target_labels": target_labels,
        "model_id": args.model_id,
    }

    summary_path = args.output_dir / "memory_injection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    LOGGER.info("Saved summary -> %s", summary_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpoint 7: Memory Injection Mitigation")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B", help="Hugging Face model ID")
    parser.add_argument("--revision", default=None, help="Optional model revision or commit hash")
    parser.add_argument("--dataset", type=Path, default=Path("checkpoint2_artifacts/combined_success.csv"))
    parser.add_argument("--baseline-results", type=Path, default=Path("checkpoint3_artifacts/baseline_results.csv"))
    parser.add_argument("--target-labels", default="F2_partial,F3_other", help="Comma separated baseline labels to analyse")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of failure cases to analyse (None => all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoint7_artifacts"))
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--source-layer", type=int, default=10, help="Layer index to derive concept vectors")
    parser.add_argument("--target-layers", default="6,8", help="Comma separated list of layers to inject into")
    parser.add_argument("--strengths", default="0.3,0.5", help="Comma separated list of injection strengths")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_memory_injection_analysis(args)


if __name__ == "__main__":
    main()
