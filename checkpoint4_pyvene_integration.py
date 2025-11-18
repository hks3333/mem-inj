"""
Checkpoint 4: pyvene Integration & Generation Parity Verification

This checkpoint wraps the Llama 1B model with pyvene's IntervenableModel,
ensuring that generation outputs are identical to the base model (no interventions).

Key Tasks:
1. Load the base model (from Checkpoint 1)
2. Create an IntervenableConfig targeting key Llama decoder components
3. Wrap with pv.IntervenableModel
4. Run parity tests: identical prompts through both models should yield identical tokens
5. Verify no runtime errors when calling pv_model.generate() without interventions

Run:
    python checkpoint4_pyvene_integration.py \
        --model-id meta-llama/Llama-3.2-1B \
        --num-test-prompts 10 \
        --output-dir checkpoint4_artifacts
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import pyvene as pv
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger("checkpoint4")


def configure_logging(verbose: bool) -> None:
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOGGER.addHandler(handler)
    logging.getLogger("transformers").setLevel(logging.WARNING)


def preferred_dtype() -> torch.dtype:
    """Select optimal dtype based on available hardware."""
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        capabilities = torch.cuda.get_device_capability(0)
        if capabilities[0] >= 7:
            return torch.float16
        return torch.float32
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def load_base_model(
    model_id: str, revision: Optional[str] = None
) -> tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    """Load the base Llama model and tokenizer."""
    dtype = preferred_dtype()
    LOGGER.info("Loading tokenizer for %s ...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        LOGGER.info("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading model %s (dtype=%s) ...", model_id, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    device = next(model.parameters()).device
    total_params = sum(p.numel() for p in model.parameters())
    LOGGER.info("Model loaded: %.2f M parameters on %s", total_params / 1e6, device)

    return tokenizer, model, device


def create_intervenable_config(model: AutoModelForCausalLM) -> pv.IntervenableConfig:
    """
    Create an IntervenableConfig targeting MLP outputs in all Llama decoder layers.

    This config allows intervention on the output of every MLP block, which is
    essential for "memory injection" experiments in later checkpoints.
    """
    num_layers = model.config.num_hidden_layers
    LOGGER.info("Creating IntervenableConfig for %d layers", num_layers)

    representations = []
    for layer_idx in range(num_layers):
        representations.append(
            pv.RepresentationConfig(
                layer=layer_idx,
                component="mlp_output",
                unit="pos",
                intervention_type=pv.VanillaIntervention,
            )
        )

    config = pv.IntervenableConfig(
        representations=representations,
        intervention_types=pv.VanillaIntervention,
        mode="parallel",
    )
    LOGGER.info(
        "IntervenableConfig created with %d intervention points", len(representations)
    )
    return config


def wrap_model_with_pyvene(
    model: AutoModelForCausalLM, config: pv.IntervenableConfig
) -> pv.IntervenableModel:
    """Wrap the base model with pyvene's IntervenableModel."""
    LOGGER.info("Wrapping model with pyvene.IntervenableModel ...")
    pv_model = pv.IntervenableModel(config, model=model)
    pv_model.eval()
    LOGGER.info("Model wrapped successfully")
    return pv_model


def generate_with_base_model(
    prompts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    max_new_tokens: int = 20,
) -> list[str]:
    """Generate text using the base (non-intervened) model."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Extract only the new tokens
    input_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[:, input_len:]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [d.strip() for d in decoded]


def generate_with_pyvene_model(
    prompts: list[str],
    tokenizer: AutoTokenizer,
    pv_model: pv.IntervenableModel,
    device: torch.device,
    max_new_tokens: int = 20,
) -> list[str]:
    """Generate text using the pyvene-wrapped model (no interventions)."""
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    base_inputs = {k: v.to(device) for k, v in encoded.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }

    with torch.no_grad():
        base_outputs, generated_ids = pv_model.generate(
            base={k: v.clone() for k, v in base_inputs.items()},
            **gen_kwargs,
        )

    if isinstance(generated_ids, tuple):
        # pyvene may return (outputs, sequences); we care about sequences
        generated_ids = generated_ids[1]

    if base_outputs is not None:
        # For completeness free CUDA memory (not required but tidy)
        del base_outputs

    input_len = base_inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, input_len:]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [d.strip() for d in decoded]


def test_generation_parity(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    pv_model: pv.IntervenableModel,
    device: torch.device,
    num_prompts: int = 10,
    max_new_tokens: int = 20,
) -> dict:
    """
    Test that base model and pyvene-wrapped model produce identical outputs.

    Returns a dict with:
    - total_tests: number of prompts tested
    - passed: number of exact matches
    - failed: number of mismatches
    - examples: list of test cases (prompt, base_output, pv_output, match)
    """
    test_prompts = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is 2 + 2?",
        "Explain quantum computing in one sentence.",
        "What is the largest planet in our solar system?",
        "Who was the first President of the United States?",
        "What is the chemical symbol for gold?",
        "How many continents are there?",
        "What is the speed of light?",
        "Name a famous mathematician.",
    ][:num_prompts]

    LOGGER.info("Running parity tests on %d prompts ...", len(test_prompts))

    base_outputs = generate_with_base_model(test_prompts, tokenizer, model, device, max_new_tokens)
    pv_outputs = generate_with_pyvene_model(test_prompts, tokenizer, pv_model, device, max_new_tokens)

    results = {
        "total_tests": len(test_prompts),
        "passed": 0,
        "failed": 0,
        "examples": [],
    }

    for prompt, base_out, pv_out in zip(test_prompts, base_outputs, pv_outputs):
        match = base_out == pv_out
        if match:
            results["passed"] += 1
        else:
            results["failed"] += 1

        results["examples"].append(
            {
                "prompt": prompt,
                "base_output": base_out,
                "pyvene_output": pv_out,
                "match": match,
            }
        )

        status = "✓" if match else "✗"
        LOGGER.info(
            "%s Prompt: %s | Base: %s | PyVene: %s",
            status,
            prompt[:50],
            base_out[:50],
            pv_out[:50],
        )

    LOGGER.info(
        "Parity test results: %d/%d passed (%.1f%%)",
        results["passed"],
        results["total_tests"],
        100 * results["passed"] / results["total_tests"],
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpoint 4: pyvene Integration & Parity Verification")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B", help="Hugging Face model repo id")
    parser.add_argument("--revision", default=None, help="Optional model revision or commit hash")
    parser.add_argument("--num-test-prompts", type=int, default=10, help="Number of prompts for parity testing")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="Max tokens to generate per prompt")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoint4_artifacts"), help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    configure_logging(args.verbose)
    LOGGER.info("Starting Checkpoint 4: pyvene Integration")

    # 1. Load base model
    tokenizer, model, device = load_base_model(args.model_id, args.revision)

    # 2. Create IntervenableConfig
    config = create_intervenable_config(model)

    # 3. Wrap with pyvene
    pv_model = wrap_model_with_pyvene(model, config)

    # 4. Run parity tests
    parity_results = test_generation_parity(
        tokenizer,
        model,
        pv_model,
        device,
        num_prompts=args.num_test_prompts,
        max_new_tokens=args.max_new_tokens,
    )

    # 5. Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "parity_test_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(parity_results, f, indent=2, ensure_ascii=False)
    LOGGER.info("Parity test results saved to %s", results_path)

    # 6. Summary
    if parity_results["failed"] == 0:
        LOGGER.info("✓ Checkpoint 4 PASSED: All parity tests passed. Model is ready for interventions.")
    else:
        LOGGER.warning(
            "⚠ Checkpoint 4 WARNING: %d parity test(s) failed. Review mismatches before proceeding.",
            parity_results["failed"],
        )


if __name__ == "__main__":
    main()
