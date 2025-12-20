"""
Checkpoint 5: Logit Lens Diagnostic

This script collects per-layer hidden states (via pyvene CollectIntervention), projects
those states through the model's unembedding matrix, and tracks the rank of the
intermediate-hop answer token across transformer layers. The goal is to visualize
"fade-out" behaviour for multi-hop failures where the model knew the hop answers
but failed on the composite question.

Typical usage:
    python checkpoint5_logit_lens.py \
        --dataset checkpoint2_artifacts/combined_success.csv \
        --baseline-results checkpoint3_artifacts/baseline_results.csv \
        --output-dir checkpoint5_artifacts \
        --sample-size 10 \
        --generate-plots
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
import pyvene as pv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger("checkpoint5")


# ---------------------------------------------------------------------------
# Utility dataclasses & helpers
# ---------------------------------------------------------------------------

@dataclass
class CaseSample:
    row_id: str
    label: str
    question: str
    intermediate_answer: str
    target_token_id: int
    target_token_text: str
    target_token_ids: List[int]
    target_token_texts: List[str]
    baseline_answer: str


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


def load_base_model(model_id: str, revision: Optional[str]) -> tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    dtype = preferred_dtype()
    LOGGER.info("Loading tokenizer for %s ...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        LOGGER.info("Tokenizer lacks pad_token -> setting to eos_token (%s)", tokenizer.eos_token)
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading model %s (dtype=%s)...", model_id, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        # auto device map unavailable (CPU / MPS)
        target_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(target_device)
    model.eval()

    device = next(model.parameters()).device
    LOGGER.info("Model loaded on %s (%.2fM params)", device, sum(p.numel() for p in model.parameters()) / 1e6)
    return tokenizer, model, device


def create_collect_config(model: AutoModelForCausalLM) -> pv.IntervenableConfig:
    num_layers = model.config.num_hidden_layers
    representations = []
    for layer_idx in range(num_layers):
        representations.append(
            pv.RepresentationConfig(
                layer=layer_idx,
                component="block_output",  # Residual stream after each transformer block
                unit="pos",
                intervention_type=pv.CollectIntervention,
            )
        )
    LOGGER.info("Configured CollectIntervention for %d layers", num_layers)
    return pv.IntervenableConfig(
        representations=representations,
        intervention_types=pv.CollectIntervention,
        mode="parallel",
    )


def wrap_with_collect_model(model: AutoModelForCausalLM, config: pv.IntervenableConfig) -> pv.IntervenableModel:
    pv_model = pv.IntervenableModel(config, model=model)
    pv_model.eval()
    return pv_model


def parse_jsonish_list(value: Optional[str]) -> List:
    if value is None:
        return []
    if isinstance(value, list):  # Already parsed
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        LOGGER.debug("Failed to json-decode '%s'", text)
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


def apply_final_norm(model: AutoModelForCausalLM, hidden_vec: torch.Tensor) -> torch.Tensor:
    """Apply the model's final RMSNorm (if present) to a hidden vector."""
    core_model = getattr(model, "model", model)
    norm_layer = getattr(core_model, "norm", None)
    if norm_layer is None:
        return hidden_vec

    target_device = norm_layer.weight.device
    target_dtype = norm_layer.weight.dtype
    vec = hidden_vec.to(device=target_device, dtype=target_dtype)
    if vec.dim() == 1:
        return norm_layer(vec.unsqueeze(0)).squeeze(0)
    return norm_layer(vec)


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


def encode_target_token(tokenizer: AutoTokenizer, answer_text: str) -> Optional[Tuple[List[int], List[str]]]:
    token_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    if not token_ids:
        return None
    token_texts = [tokenizer.decode([tok_id]).strip() for tok_id in token_ids]
    return token_ids, token_texts


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
    LOGGER.info("Merged %d baseline rows with dataset", len(merged))

    filtered = merged[merged["label"].isin(target_labels)].copy()
    LOGGER.info("Filtered to %d rows with labels %s", len(filtered), target_labels)

    if filtered.empty:
        raise RuntimeError("No matching cases found for requested labels")

    if sample_size is not None and sample_size < len(filtered):
        filtered = filtered.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    else:
        filtered = filtered.reset_index(drop=True)
    return filtered


def prepare_cases(df: pd.DataFrame, tokenizer: AutoTokenizer) -> List[CaseSample]:
    samples: List[CaseSample] = []
    for _, row in df.iterrows():
        hop_truths = parse_jsonish_list(row.get("hop_ground_truths_raw"))
        intermediate = select_intermediate_answer(hop_truths)

        if not intermediate:
            LOGGER.debug("Skipping row %s: no intermediate answer detected", row["row_id"])
            continue

        encoded = encode_target_token(tokenizer, intermediate)
        if not encoded:
            LOGGER.debug("Skipping row %s: tokenization produced empty ids", row["row_id"])
            continue

        token_ids, token_texts = encoded
        samples.append(
            CaseSample(
                row_id=row["row_id"],
                label=row["label"],
                question=row["composite_question"],
                intermediate_answer=intermediate,
                target_token_id=token_ids[0],
                target_token_text=token_texts[0],
                target_token_ids=token_ids,
                target_token_texts=token_texts,
                baseline_answer=row.get("baseline_answer", ""),
            )
        )
    if not samples:
        raise RuntimeError("No usable samples after processing hop ground truths")
    LOGGER.info("Prepared %d case samples", len(samples))
    return samples


def build_unit_locations(pv_model: pv.IntervenableModel, last_token_index: int) -> Dict[str, tuple[Optional[List], List[List[List[int]]]]]:
    num_interventions = len(pv_model.interventions)
    return {
        "sources->base": (
            None,
            [
                [[last_token_index]]
                for _ in range(num_interventions)
            ],
        )
    }


def extract_hidden_vector(tensor: torch.Tensor) -> torch.Tensor:
    # Expected shapes: [batch, positions, hidden] or [batch, hidden]
    if tensor.dim() == 3:
        return tensor[0, -1, :]
    if tensor.dim() == 2:
        return tensor[0, :]
    if tensor.dim() == 1:
        return tensor
    raise ValueError(f"Unexpected activation shape: {tuple(tensor.shape)}")


def compute_rank(logits: torch.Tensor, token_id: int) -> int:
    target_logit = logits[token_id]
    rank = int((logits > target_logit).sum().item()) + 1
    return rank


def run_logit_lens(
    tokenizer: AutoTokenizer,
    pv_model: pv.IntervenableModel,
    device: torch.device,
    sample: CaseSample,
    unembed_weight: torch.Tensor,
) -> Dict:
    batch = tokenizer(sample.question, return_tensors="pt")
    base_inputs = {k: v.to(device) for k, v in batch.items()}
    last_token_index = base_inputs["input_ids"].shape[1] - 1
    unit_locations = build_unit_locations(pv_model, last_token_index)

    with torch.no_grad():
        outputs = pv_model(
            base=base_inputs,
            unit_locations=unit_locations,
            return_dict=True,
        )

    activations = outputs.collected_activations
    if activations is None:
        raise RuntimeError("CollectIntervention did not return activations")

    if isinstance(activations, dict):
        layer_tensors = [activations[key] for key in pv_model.sorted_keys]
    else:
        layer_tensors = list(activations)

    best_ranks: List[int] = []
    best_logits: List[float] = []
    best_token_indices: List[int] = []
    best_token_texts: List[str] = []

    for idx, tensor in enumerate(layer_tensors):
        if tensor is None:
            best_ranks.append(-1)
            best_logits.append(float("nan"))
            best_token_indices.append(0)
            best_token_texts.append("")
            continue
        if isinstance(tensor, (list, tuple)):
            tensor = next((t for t in tensor if isinstance(t, torch.Tensor)), None)
        if tensor is None:
            best_ranks.append(-1)
            best_logits.append(float("nan"))
            best_token_indices.append(0)
            best_token_texts.append("")
            continue
        hidden_vec = extract_hidden_vector(tensor.to(device))
        hidden_vec = apply_final_norm(pv_model.model, hidden_vec)
        hidden_for_logits = hidden_vec.to(unembed_weight.device)
        logits = torch.matmul(hidden_for_logits, unembed_weight.T)

        token_ranks: List[int] = []
        token_logits = logits[sample.target_token_ids]
        for tok_id in sample.target_token_ids:
            token_ranks.append(compute_rank(logits, tok_id))

        best_idx = min(range(len(token_ranks)), key=lambda i: token_ranks[i])
        best_ranks.append(token_ranks[best_idx])
        best_logits.append(token_logits[best_idx].item())
        best_token_indices.append(best_idx)
        best_token_texts.append(sample.target_token_texts[best_idx])

    # Pull final forward logits for the last token (model prediction)
    final_logits = None
    if getattr(outputs.intervened_outputs, "logits", None) is not None:
        final_logits = outputs.intervened_outputs.logits[0, last_token_index, :].detach().cpu()

    return {
        "best_ranks": best_ranks,
        "best_logits": best_logits,
        "best_token_indices": best_token_indices,
        "best_token_texts": best_token_texts,
        "final_logits": final_logits,
    }


def maybe_generate_plot(
    ranks: List[int],
    sample: CaseSample,
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting optional
        LOGGER.warning("matplotlib not available; skipping plot for %s", sample.row_id)
        return

    layers = list(range(len(ranks)))
    filtered_layers = [l for l, r in zip(layers, ranks) if r > 0]
    filtered_ranks = [r for r in ranks if r > 0]
    if not filtered_layers:
        LOGGER.warning("No positive ranks to plot for %s", sample.row_id)
        return

    plt.figure(figsize=(6, 4))
    plt.plot(filtered_layers, filtered_ranks, marker="o")
    plt.yscale("log")
    plt.xlabel("Layer")
    plt.ylabel("Rank of intermediate token (log scale)")
    plt.title(f"Row {sample.row_id} | label={sample.label}")
    plt.grid(True, which="both", ls="--", alpha=0.4)

    plot_path = output_dir / f"logit_lens_rank_{sample.row_id}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpoint 5: Logit Lens Diagnostic")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B", help="Hugging Face model ID")
    parser.add_argument("--revision", default=None, help="Optional model revision or commit hash")
    parser.add_argument("--dataset", type=Path, default=Path("checkpoint2_artifacts/combined_success.csv"))
    parser.add_argument("--baseline-results", type=Path, default=Path("checkpoint3_artifacts/baseline_results.csv"))
    parser.add_argument(
        "--target-labels",
        default="F2_partial,F3_other",
        help="Comma-separated baseline labels to analyse",
    )
    parser.add_argument("--sample-size", type=int, default=10, help="Number of cases to analyse (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoint5_artifacts"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--generate-plots", action="store_true", help="Generate layer-rank plots per case")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_labels = [label.strip() for label in args.target_labels.split(",") if label.strip()]
    if not target_labels:
        raise ValueError("No target labels provided")

    LOGGER.info("Loading data and selecting cases ...")
    merged_df = load_failure_cases(
        dataset_path=args.dataset,
        baseline_path=args.baseline_results,
        target_labels=target_labels,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    LOGGER.info("Loading base model ...")
    tokenizer, base_model, device = load_base_model(args.model_id, args.revision)

    collect_config = create_collect_config(base_model)
    pv_model = wrap_with_collect_model(base_model, collect_config)
    unembed_weight = pv_model.model.get_output_embeddings().weight

    samples = prepare_cases(merged_df, tokenizer)

    analysis_rows = []

    for sample in tqdm(samples, desc="Running Logit Lens"):
        try:
            result = run_logit_lens(tokenizer, pv_model, device, sample, unembed_weight)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed on row %s: %s", sample.row_id, exc)
            continue

        analysis_rows.append({
            "row_id": sample.row_id,
            "label": sample.label,
            "question": sample.question,
            "baseline_answer": sample.baseline_answer,
            "intermediate_answer": sample.intermediate_answer,
            "target_token_id": sample.target_token_id,
            "target_token_text": sample.target_token_text,
            "target_token_ids": sample.target_token_ids,
            "target_token_texts": sample.target_token_texts,
            "layer_best_ranks": result["best_ranks"],
            "layer_best_logits": result["best_logits"],
            "layer_best_token_indices": result["best_token_indices"],
            "layer_best_token_texts": result["best_token_texts"],
        })

        if args.generate_plots:
            maybe_generate_plot(result["best_ranks"], sample, args.output_dir)

    if not analysis_rows:
        LOGGER.error("No analyses completed successfully")
        return

    # Save tabular summary (ranks encoded as JSON strings)
    summary_df = pd.DataFrame(analysis_rows)
    summary_df["target_token_ids_json"] = summary_df["target_token_ids"].apply(json.dumps)
    summary_df["target_token_texts_json"] = summary_df["target_token_texts"].apply(json.dumps)
    summary_df["layer_ranks_json"] = summary_df["layer_best_ranks"].apply(json.dumps)
    summary_df["target_logit_per_layer_json"] = summary_df["layer_best_logits"].apply(json.dumps)
    summary_df["layer_best_token_indices_json"] = summary_df["layer_best_token_indices"].apply(json.dumps)
    summary_df["layer_best_token_texts_json"] = summary_df["layer_best_token_texts"].apply(json.dumps)

    summary_csv_path = args.output_dir / "logit_lens_summary.csv"
    columns_to_export = [
        "row_id",
        "label",
        "baseline_answer",
        "intermediate_answer",
        "target_token_text",
        "target_token_id",
        "target_token_ids_json",
        "target_token_texts_json",
        "layer_ranks_json",
        "target_logit_per_layer_json",
        "layer_best_token_indices_json",
        "layer_best_token_texts_json",
    ]
    summary_df.to_csv(summary_csv_path, index=False, columns=columns_to_export)
    LOGGER.info("Saved summary CSV -> %s", summary_csv_path)

    # Save metadata / stats JSON
    label_counts = Counter(row["label"] for row in analysis_rows)
    metadata = {
        "total_cases": len(analysis_rows),
        "labels": dict(label_counts),
        "target_labels": target_labels,
        "model_id": args.model_id,
        "dataset": str(args.dataset),
        "baseline_results": str(args.baseline_results),
    }
    metadata_path = args.output_dir / "logit_lens_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    LOGGER.info("Saved metadata -> %s", metadata_path)


if __name__ == "__main__":
    main()
