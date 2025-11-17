"""Checkpoint 1 helper: load and inspect the Llama 1B base model.

Run:
    python checkpoint1_load_model.py --model-id meta-llama/Llama-3.2-1B

The script prints:
  * Torch & device diagnostics
  * Tokenizer padding configuration
  * Parameter counts (total / trainable)
  * Model dtype and inferred device map
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger("checkpoint1")


def configure_logging() -> None:
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(handler)


def preferred_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        capabilities = torch.cuda.get_device_capability(0)
        # Many recent GPUs support bfloat16; fall back to float16 otherwise
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if capabilities[0] >= 7:  # Volta/Turing+ handle float16 well
            return torch.float16
        return torch.float32
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def load_model(model_id: str, revision: Optional[str]) -> None:
    LOGGER.info("Torch version: %s", torch.__version__)
    if torch.cuda.is_available():
        LOGGER.info("CUDA available: %s GPU(s)", torch.cuda.device_count())
        for idx in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(idx)
            capability = torch.cuda.get_device_capability(idx)
            LOGGER.info("  GPU %d -> %s (compute capability %s)", idx, name, capability)
    else:
        LOGGER.info("CUDA not available; falling back to CPU/MPS")
    if torch.backends.mps.is_available():
        LOGGER.info("Apple MPS backend detected")

    dtype = preferred_dtype()
    LOGGER.info("Preferred dtype: %s", dtype)

    LOGGER.info("Loading tokenizer for %s ...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        LOGGER.info("Tokenizer lacks pad_token -> setting to eos_token (%s)", tokenizer.eos_token)
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading model %s ...", model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info("Model dtype: %s", next(model.parameters()).dtype)
    LOGGER.info("Total parameters: %.2f M", total_params / 1e6)
    LOGGER.info("Trainable parameters: %.2f M", trainable_params / 1e6)

    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        LOGGER.info("Device map:")
        for module, device in device_map.items():
            LOGGER.info("  %s -> %s", module, device)
    else:
        LOGGER.info("No device map exposed; model resides on %s", next(model.parameters()).device)

    sample_ids = tokenizer("Test", return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**sample_ids, output_hidden_states=False)
    LOGGER.info("Forward pass ok: logits shape %s", outputs.logits.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpoint 1: load base Llama model")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B", help="Hugging Face model repo id")
    parser.add_argument("--revision", default=None, help="Optional model revision or commit hash")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    LOGGER.info("Starting checkpoint 1 loader")
    load_model(model_id=args.model_id, revision=args.revision)
    LOGGER.info("Checkpoint 1 completed successfully")


if __name__ == "__main__":
    main()
