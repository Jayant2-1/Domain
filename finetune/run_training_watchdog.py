"""
Watchdog for LoRA training – auto-resumes from the latest checkpoint if the
training process exits unexpectedly (OOM, SIGKILL, power blip, etc.).

Usage:
    python -m finetune.run_training_watchdog [same args as train_lora.py]

It forwards all arguments to train_lora.py, adding --resume-from-checkpoint=auto
on every retry so the Trainer picks up the last saved checkpoint.
"""
import argparse
import glob
import logging
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WATCHDOG] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("watchdog")

MAX_RETRIES = 20          # stop after this many consecutive crashes
RETRY_DELAY_SEC = 15      # seconds to wait before each retry


def _latest_checkpoint(output_dir: str) -> str | None:
    """Return the path of the highest-numbered checkpoint under output_dir."""
    pattern = str(Path(output_dir) / "**" / "checkpoint-*")
    candidates = sorted(
        glob.glob(pattern, recursive=True),
        key=lambda p: int(p.split("checkpoint-")[-1].rstrip("/\\")),
    )
    return candidates[-1] if candidates else None


def _run(cmd: list[str]) -> int:
    """Run a subprocess, streaming its stdout/stderr, return exit code."""
    log.info("Running: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    proc.wait()
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Crash-resilient wrapper around train_lora.py",
        add_help=False,
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--retry-delay", type=int, default=RETRY_DELAY_SEC)
    known, passthrough = parser.parse_known_args()

    # Derive output dir the same way train_lora.main() does
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = known.output or f"adapters/v_{ts}"

    # Inject --output so train_lora.py uses the same dir across retries
    if "--output" not in passthrough and "-o" not in passthrough:
        passthrough += ["--output", output_dir]

    base_cmd = [
        sys.executable, "-m", "finetune.train_lora",
    ] + passthrough

    attempt = 0
    while attempt < known.max_retries:
        attempt += 1
        log.info("=" * 60)
        log.info("Attempt %d / %d", attempt, known.max_retries)

        cmd = list(base_cmd)

        # On retries always try to resume – --resume-from-checkpoint=auto
        # lets train_lora.py scan for the latest checkpoint itself.
        if attempt > 1:
            # Remove any existing --resume-from-checkpoint so we don't dupe it
            cmd = [a for a in cmd if not a.startswith("--resume-from-checkpoint")]
            cmd += ["--resume-from-checkpoint", "auto"]
            ckpt = _latest_checkpoint(output_dir)
            if ckpt:
                log.info("Latest checkpoint found: %s", ckpt)
            else:
                log.warning("No checkpoint found – will restart from scratch.")
            log.info("Waiting %d s before retry …", known.retry_delay)
            time.sleep(known.retry_delay)

        rc = _run(cmd)

        if rc == 0:
            log.info("Training completed successfully (exit 0).")
            return

        log.error(
            "Training process exited with code %d (attempt %d/%d).",
            rc, attempt, known.max_retries,
        )

    log.error("Exceeded max retries (%d). Giving up.", known.max_retries)
    sys.exit(1)


if __name__ == "__main__":
    main()
