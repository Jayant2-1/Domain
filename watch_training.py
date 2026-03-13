"""
Live training watcher — polls progress every 30 seconds.
Ctrl+C stops watching. Training keeps running.

Progress sources (in priority order):
  1. progress.json written by LiveProgressCallback every 10 steps
  2. trainer_state.json inside each checkpoint-* dir
  3. Latest checkpoint number as lower-bound estimate
"""
import time
import glob
import os
import sys
import json
import subprocess

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "adapters", "v_20260310_hf_multi")

ROUNDS = [
    ("round_1", 400,  "Primary Learning"),
    ("round_2", 150,  "Refinement"),
    ("round_3", 75,   "Final Polish"),
]

seen_ckpts: set = set()


def _gpu_stats():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=4
        ).decode().strip()
        util, used, total, temp = [x.strip() for x in out.split(",")]
        return f"GPU {util}%  VRAM {used}/{total} MiB  {temp}°C"
    except Exception:
        return "GPU stats unavailable"


def _read_step(rnd_path):
    """Return (step, loss, accuracy) from the best available source."""
    # 1. progress.json (written every 10 steps by LiveProgressCallback)
    pf = os.path.join(rnd_path, "progress.json")
    if os.path.exists(pf):
        try:
            with open(pf) as f:
                d = json.load(f)
            return d.get("step", 0), d.get("loss"), d.get("accuracy")
        except Exception:
            pass

    # 2. trainer_state.json in output dir (written at each checkpoint)
    sf = os.path.join(rnd_path, "trainer_state.json")
    if os.path.exists(sf):
        try:
            with open(sf) as f:
                s = json.load(f)
            step = int(s.get("global_step", 0))
            loss = None
            if s.get("log_history"):
                last = s["log_history"][-1]
                loss = last.get("loss")
            return step, loss, None
        except Exception:
            pass

    # 3. Latest checkpoint number
    ckpts = sorted(glob.glob(os.path.join(rnd_path, "checkpoint-*")))
    if ckpts:
        try:
            step = max(int(c.split("checkpoint-")[-1].rstrip("/\\")) for c in ckpts)
            # Try reading trainer_state inside the checkpoint
            for ckpt in reversed(ckpts):
                sf2 = os.path.join(ckpt, "trainer_state.json")
                if os.path.exists(sf2):
                    try:
                        with open(sf2) as f:
                            s = json.load(f)
                        step = int(s.get("global_step", step))
                        loss = s["log_history"][-1].get("loss") if s.get("log_history") else None
                        return step, loss, None
                    except Exception:
                        pass
            return step, None, None
        except Exception:
            pass

    return 0, None, None


print()
print("=" * 70)
print("  TRAINING WATCHER  —  Ctrl+C to stop watching")
print("  Training process continues regardless")
print("  NOTE: step counter updates every 10 steps (progress.json)")
print("        or at checkpoint saves (every 100 steps) if no progress.json")
print("=" * 70)

while True:
    # Announce new checkpoints
    all_ckpts = sorted(glob.glob(os.path.join(OUTPUT_DIR, "**", "checkpoint-*"), recursive=True))
    for c in all_ckpts:
        if c not in seen_ckpts:
            step = c.split("checkpoint-")[-1].rstrip("/\\")
            rnd = next((r[0] for r in ROUNDS if r[0] in c), "?")
            print(f"\n  *** CHECKPOINT SAVED  [{rnd}]  step {step}  →  {os.path.basename(c)}")
    seen_ckpts = set(all_ckpts)

    now = time.strftime("%H:%M:%S")
    gpu = _gpu_stats()
    print(f"\n  {now}  |  {gpu}")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    for rnd_dir, rnd_total, rnd_label in ROUNDS:
        rnd_path = os.path.join(OUTPUT_DIR, rnd_dir)
        if not os.path.isdir(rnd_path):
            print(f"  [{rnd_label:<20}]  not started yet")
            continue

        step, loss, acc = _read_step(rnd_path)
        pct    = (step / rnd_total) * 100
        filled = int(pct / 2)
        bar    = "█" * filled + "░" * (50 - filled)

        if step >= rnd_total:
            status = "DONE ✓"
        elif step == 0:
            status = "running — waiting for first log (step 10)..."
        else:
            extra = ""
            if loss is not None:
                extra += f"  loss={loss:.4f}"
            if acc is not None:
                extra += f"  acc={acc:.1%}"
            status = f"step {step}/{rnd_total} ({pct:.0f}%){extra}"

        print(f"  [{rnd_label:<20}]  |{bar}|  {status}")

    print()
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        print("\n  Watcher stopped. Training continues in background.\n")
        sys.exit(0)
