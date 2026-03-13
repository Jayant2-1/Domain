"""Resume training from Round 3 (after Round 2 completes)."""
import sys
import os

log_file = open("training_round3.log", "w", buffering=1)

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

print("=== Resuming training from Round 3 ===")
print(f"Python: {sys.version}")
print(f"CWD: {os.getcwd()}")

try:
    from finetune.train_lora import train_rounds
    print("Import OK")

    adapter_path = train_rounds(
        data_path="finetune/data/combined_v2_training.jsonl",
        base_output_dir="adapters/v_20260310_hf_multi",
        start_round=3,
    )
    print(f"\n=== ALL TRAINING COMPLETE ===")
    print(f"Final adapter: {adapter_path}")
except Exception as e:
    print(f"\n=== ERROR ===\n{e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
finally:
    log_file.close()
