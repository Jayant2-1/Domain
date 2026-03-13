"""
Merge a LoRA adapter into the base model (optional).

This produces a standalone model that no longer needs PEFT at inference
time.  Useful for final deployment or for starting a new round of
fine-tuning from a merged checkpoint.

Usage:
    python -m finetune.merge_adapters \
        --adapter adapters/v_20260224_120000 \
        --output models/merged_20260224
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "models/Mistral-7B-Instruct-v0.3"


def merge(
    adapter_path: str,
    output_dir: str,
    model_id: str = DEFAULT_MODEL_ID,
) -> str:
    """
    Load the base model in 4-bit, apply the adapter, merge, and save
    the merged model in FP16.

    Returns the output directory path.
    """
    path = Path(adapter_path)
    if not path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base model %s …", model_id)
    # Load in FP16 (not 4-bit) so merge produces full-precision weights
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading adapter from %s …", adapter_path)
    model = PeftModel.from_pretrained(base_model, adapter_path)

    logger.info("Merging adapter into base model …")
    merged = model.merge_and_unload()

    merged.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    logger.info("Merged model saved to %s", out)

    return str(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter dir")
    parser.add_argument("--output", required=True, help="Output directory for merged model")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = parser.parse_args()
    merge(args.adapter, args.output, args.model_id)


if __name__ == "__main__":
    main()
