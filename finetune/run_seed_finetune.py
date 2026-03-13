"""
Quick-start fine-tuning with seed DSA data.

Usage:
    python -m finetune.run_seed_finetune

This uses the pre-made seed_dsa_training.jsonl (12 high-quality DSA QA pairs)
to create an initial LoRA adapter. Good for testing the pipeline before
accumulating real user interactions.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

def main() -> None:
    seed_data = Path("finetune/data/seed_dsa_training.jsonl")
    if not seed_data.exists():
        print(f"ERROR: Seed data not found at {seed_data}")
        sys.exit(1)

    # Count records
    with open(seed_data) as f:
        count = sum(1 for line in f if line.strip())
    print(f"Found {count} seed training records")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"adapters/seed_v_{ts}"

    print(f"Output adapter: {output_dir}")
    print("Starting LoRA fine-tuning on seed data...")
    print("=" * 60)

    from finetune.train_lora import train

    train(
        data_path=str(seed_data),
        output_dir=output_dir,
        model_id="models/Mistral-7B-Instruct-v0.3",
        max_seq_len=512,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        batch_size=1,
        grad_accum=4,
        learning_rate=2e-4,
        max_steps=100,  # fewer steps for seed data
    )

    print("=" * 60)
    print(f"Seed adapter saved to: {output_dir}")
    print(f"To use: set MLML_ADAPTER_DIR={output_dir} and restart the API")


if __name__ == "__main__":
    main()
