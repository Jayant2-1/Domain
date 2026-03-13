"""
End-to-end training runner for DSA Tutor fine-tuning.

Usage:
    python -m finetune.run_full_training

This script:
1. Checks that training data exists
2. Merges all available data sources (master + seed + optional downloaded supplement)
3. Shows training config and asks for confirmation
4. Runs multi-round LoRA training (3 rounds with LR decay)
5. Runs post-training evaluation on 10 held-out questions
6. Updates .env with the final adapter path
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

# -- Paths ---------------------------------------------------------------------

DATA_DIR = Path("finetune/data")
MASTER_DATA = DATA_DIR / "dsa_master_training.jsonl"
SEED_DATA = DATA_DIR / "seed_dsa_training.jsonl"
SUPPLEMENT_DATA = DATA_DIR / "downloaded_supplement.jsonl"
COMBINED_DATA = DATA_DIR / "combined_training.jsonl"
ENV_FILE = Path(".env")


# -- Helpers -------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, str]]:
    """Load JSONL records from a file."""
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def merge_and_deduplicate(
    *sources: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Merge multiple record lists, deduplicating by instruction text."""
    seen: Set[str] = set()
    merged: List[Dict[str, str]] = []
    for records in sources:
        for rec in records:
            key = rec.get("instruction", "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged.append(rec)
    return merged


def estimate_training_time(total_records: int) -> str:
    """Rough estimate based on measured 4-bit training speeds on RTX 2060."""
    # Round 1: 400, Round 2: 150, Round 3: 75 = 625 total steps
    total_steps = 625
    # Measured: ~45-90 seconds per step (batch=1, grad_accum=2, seq_len=512)
    low_mins = (total_steps * 45) // 60
    high_mins = (total_steps * 90) // 60
    return f"{low_mins // 60}h{low_mins % 60}m - {high_mins // 60}h{high_mins % 60}m"


def update_env_adapter(adapter_path: str) -> None:
    """Update MLML_ADAPTER_DIR in .env file."""
    env_lines = []
    found = False

    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("MLML_ADAPTER_DIR="):
                    env_lines.append(f"MLML_ADAPTER_DIR={adapter_path}\n")
                    found = True
                else:
                    env_lines.append(line)

    if not found:
        env_lines.append(f"MLML_ADAPTER_DIR={adapter_path}\n")

    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.writelines(env_lines)


def print_config(total_records: int, adapter_dir: str) -> None:
    """Print the full training configuration (no heavy imports)."""
    print()
    print("+" + "=" * 58 + "+")
    print("|  TRAINING CONFIGURATION                                  |")
    print("+" + "=" * 58 + "+")
    print()
    print(f"  Model:            models/Mistral-7B-Instruct-v0.3")
    print(f"  Training records: {total_records}")
    print(f"  Output dir:       {adapter_dir}")
    print()
    print(f"  LoRA r:           64")
    print(f"  LoRA alpha:       128")
    print(f"  LoRA dropout:     0.05")
    print(f"  Target modules:   ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']")
    print(f"  Max seq length:   512")
    print(f"  Batch size:       1")
    print(f"  Grad accumulation:2")
    print()
    print("  Training Rounds:")
    print("    Single round: lr=2e-4, 625 steps, cosine decay, warmup=50")
    print()

    # Hardware info
    # Get GPU info via nvidia-smi to avoid initialising CUDA context
    # (premature CUDA init uses ~4 GB on Windows WDDM, breaking model loading).
    try:
        import subprocess as _sp
        _nv = _sp.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if _nv.returncode == 0 and _nv.stdout.strip():
            print(f"  GPU:              {_nv.stdout.strip()}")
        else:
            print("  GPU:              (unable to query)")
    except Exception:
        print("  GPU:              (nvidia-smi not found)")

    est = estimate_training_time(total_records)
    print(f"  Estimated time:   ~{est}")
    print()


# -- Main ----------------------------------------------------------------------

def main() -> None:
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    _args = _p.parse_args()

    print()
    print("=" * 60)
    print("  DSA TUTOR - FULL TRAINING PIPELINE")
    print("=" * 60)
    print()

    # Step 1: Check master data exists
    if not MASTER_DATA.exists():
        print(f"ERROR: Master training data not found at:")
        print(f"  {MASTER_DATA.resolve()}")
        print()
        print("Run the conversion script first:")
        print("  python -m finetune.convert_leetcode")
        print()
        sys.exit(1)

    # Step 2: Load and count all data sources
    print("Loading training data sources...")
    master_records = load_jsonl(MASTER_DATA)
    print(f"  Master data:      {len(master_records):>6} records  ({MASTER_DATA})")

    seed_records = load_jsonl(SEED_DATA)
    if seed_records:
        print(f"  Seed data:        {len(seed_records):>6} records  ({SEED_DATA})")
    else:
        print(f"  Seed data:        (not found, skipping)")

    supplement_records = load_jsonl(SUPPLEMENT_DATA)
    if supplement_records:
        print(f"  Supplement data:  {len(supplement_records):>6} records  ({SUPPLEMENT_DATA})")
    else:
        print(f"  Supplement data:  (not found, skipping)")

    # Step 3: Merge and deduplicate
    print()
    print("Merging and deduplicating...")
    combined = merge_and_deduplicate(master_records, seed_records, supplement_records)
    print(f"  Combined unique:  {len(combined):>6} records")

    # Write combined file
    with open(COMBINED_DATA, "w", encoding="utf-8") as f:
        for rec in combined:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved to:         {COMBINED_DATA}")

    # Step 4: Show config
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    adapter_dir = f"adapters/v_{ts}"
    print_config(len(combined), adapter_dir)

    # Step 5: Confirm
    est = estimate_training_time(len(combined))
    if not _args.yes:
        response = input(
            f"Ready to train? This will take approximately {est} on your GPU. [y/N] "
        ).strip().lower()

        if response not in ("y", "yes"):
            print("Training cancelled.")
            sys.exit(0)

    print()
    print("Starting training...")
    print()

    # Step 6: Train (single round to avoid CUDA context reload issues on WDDM)
    from finetune.train_lora import train, evaluate

    start_time = time.time()

    adapter_path, start_loss, end_loss = train(
        data_path=str(COMBINED_DATA),
        output_dir=adapter_dir,
        max_steps=625,
        learning_rate=2e-4,
        warmup_steps=50,
        lr_scheduler_type="cosine",
    )

    elapsed = time.time() - start_time
    elapsed_mins = int(elapsed // 60)
    elapsed_secs = int(elapsed % 60)

    print()
    print(f"  Total training time: {elapsed_mins}m {elapsed_secs}s")

    # Step 7: Post-training evaluation (run in subprocess for clean CUDA context)
    print()
    print("Running post-training evaluation...")
    import subprocess
    eval_result = subprocess.run(
        [sys.executable, "-c",
         f"from finetune.train_lora import evaluate; evaluate('{adapter_path}')"],
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    if eval_result.returncode != 0:
        print("  (Evaluation failed — you can re-run manually later)")
        print(f"    python -c \"from finetune.train_lora import evaluate; evaluate('{adapter_path}')\"")


    # Step 8: Update .env
    update_env_adapter(adapter_path)

    # Final summary
    print()
    print("+" + "=" * 58 + "+")
    print("|  TRAINING COMPLETE                                       |")
    print(f"|  Adapter saved to: {adapter_path:<39} |")
    print("|  To activate: restart the API server                     |")
    print("|  The adapter is already set in .env                      |")
    print("+" + "=" * 58 + "+")
    print()


if __name__ == "__main__":
    main()
