"""
LoRA fine-tuning script for Mistral-7B-Instruct-v0.3.

Trains a LoRA adapter on top of the 4-bit quantised base model using
instruction-tuning data.  Designed for RTX 2060 (6 GB VRAM) on Windows.

Usage:
    python -m finetune.train_lora \
        --data finetune/data/dsa_master_training.jsonl \
        --output adapters/my_adapter \
        --max-steps 400
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Python 3.14 + PyTorch 2.10 compatibility patch
# ---------------------------------------------------------------------------
# transformers ≥ 4.51 uses safe_open(...).get_slice(name)[...] which triggers
# a Rust→Python ABI bug in safetensors 0.7.0 + PyTorch 2.10 on Python 3.14:
#   UntypedStorage.__getitem__ receives an invalid (null) self → access violation
# Fix: replace get_slice() with get_tensor() inside modeling_utils so that
# actual torch.Tensor indexing is used (which is stable on all Python versions).

def _apply_py314_loading_patch() -> bool:
    """Patch transformers.modeling_utils.safe_open for Python 3.14 compatibility.
    Returns True if the patch was applied, False if not needed."""
    import sys
    if sys.version_info < (3, 14):
        return False

    try:
        import transformers.modeling_utils as _mu
        from safetensors import safe_open as _real_safe_open
        import contextlib

        if getattr(_mu, "_py314_patch_applied", False):
            return True  # already patched

        from safetensors.torch import load_file as _safetensors_load_file

        class _SafeOpenWrapper:
            """Drop-in for safe_open.
            Preloads the shard once via load_file() (the only Py3.14-safe path),
            holds all tensors for the lifetime of THIS object only, then releases
            them when the object is GC'd (so only ONE shard is in RAM at a time).
            """
            def __init__(self, path: str, framework: str, device: str = "cpu"):
                self._path = str(path)
                self._framework = framework
                # Open real fp for metadata/keys only (no tensor data)
                self._meta_fp = _real_safe_open(self._path, framework=framework, device="cpu")
                self._tensors: Optional[dict] = None   # loaded on first tensor access

            def __enter__(self):
                return self

            def __exit__(self, *args):
                # Release shard tensors as soon as the 'with' block exits
                self._tensors = None

            def _ensure_loaded(self) -> dict:
                if self._tensors is None:
                    logger.debug("Loading shard via load_file: %s",
                                 self._path.split("\\")[-1])
                    self._tensors = _safetensors_load_file(self._path, device="cpu")
                return self._tensors

            def get_slice(self, name: str):
                return _TensorSliceProxy(self, name)

            def get_tensor(self, name: str):
                return self._ensure_loaded()[name]

            def keys(self):
                return self._meta_fp.keys()

            def metadata(self):
                return self._meta_fp.metadata()

            def __getattr__(self, attr: str):
                return getattr(self._meta_fp, attr)

            def __del__(self):
                self._tensors = None   # explicit freeing on GC

        class _TensorSliceProxy:
            """Tensor slice served from the shard's preloaded dict.
            • get_shape / get_dtype use the real safetensors Slice (reads file
              header, no tensor creation, no crash).
            • __getitem__ triggers full shard preload via load_file and indexes
              the preloaded torch.Tensor (avoids all storage.__getitem__ paths).
            """
            def __init__(self, wrapper: "_SafeOpenWrapper", name: str):
                self._wrapper = wrapper
                self._name = name
                # Lazy-loaded real Slice – only for metadata (dtype/shape)
                # Note: _real_slice holds a Rust ref that keeps the fp alive
                self._real_slice = None

            def _meta_slice(self):
                """Return real safetensors Slice for metadata queries only."""
                if self._real_slice is None:
                    self._real_slice = _real_safe_open(
                        self._wrapper._path,
                        framework=self._wrapper._framework,
                        device="cpu"
                    ).get_slice(self._name)
                return self._real_slice

            def __getitem__(self, idx):
                # Full shard load — only path that was previously crashing
                return self._wrapper.get_tensor(self._name)[idx]

            def get_shape(self):
                return self._meta_slice().get_shape()

            def get_dtype(self):
                return self._meta_slice().get_dtype()

            def __getattr__(self, attr: str):
                return getattr(self._meta_slice(), attr)

        _mu.safe_open = _SafeOpenWrapper
        _mu._py314_patch_applied = True
        logger.info(
            "Applied Python 3.14 safetensors patch (get_slice→get_tensor) "
            "to transformers.modeling_utils"
        )
        return True

    except Exception as exc:
        logger.warning("Could not apply Py3.14 loading patch: %s", exc)
        return False


_apply_py314_loading_patch()




# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "models/Mistral-7B-Instruct-v0.3"
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_LORA_R = 64
DEFAULT_LORA_ALPHA = 128
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 2
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_MAX_STEPS = 400
DEFAULT_WARMUP_STEPS = 40
DEFAULT_LR_SCHEDULER = "cosine"
DEFAULT_LOGGING_STEPS = 10
DEFAULT_SAVE_STEPS = 100
DEFAULT_SAVE_TOTAL_LIMIT = 3


# ---------------------------------------------------------------------------
# Live Progress Callback
# ---------------------------------------------------------------------------

class LiveProgressCallback(TrainerCallback):
    """ASCII progress bar printed at every logging step."""

    def __init__(self, total_steps: int, bar_width: int = 50):
        self.total_steps = total_steps
        self.bar_width = bar_width
        self.loss_history: List[float] = []
        self.start_time: Optional[float] = None

    def on_train_begin(self, args, state, control, **kw):
        self.start_time = time.time()
        print()
        print("=" * 60)
        print(f"  TRAINING STARTED -- {self.total_steps} steps")
        print("=" * 60)

    def on_log(self, args, state, control, logs=None, **kw):
        try:
            if logs is None:
                return
            loss = logs.get("loss")
            if loss is None:
                return
            self.loss_history.append(loss)
            step = state.global_step
            pct = step / self.total_steps if self.total_steps else 0
            filled = int(self.bar_width * pct)
            bar = "=" * filled + ">" + "." * (self.bar_width - filled - 1)

            if len(self.loss_history) >= 2:
                trend = "v" if self.loss_history[-1] < self.loss_history[-2] else "^"
            else:
                trend = ">"

            elapsed = time.time() - (self.start_time or time.time())
            if step > 0:
                eta_secs = int(elapsed / step * (self.total_steps - step))
                eta_m, eta_s = divmod(eta_secs, 60)
                eta = f"{eta_m}m{eta_s}s"
            else:
                eta = "..."

            print(
                f"  [{bar}] {step:>5}/{self.total_steps}"
                f"  loss={loss:.4f}{trend}  eta={eta}"
            )

            # Write progress.json so watch_training.py can show live step
            try:
                acc = logs.get("mean_token_accuracy")
                progress = {
                    "step": step,
                    "total_steps": self.total_steps,
                    "loss": round(loss, 6),
                    "accuracy": round(acc, 6) if acc is not None else None,
                    "eta": eta,
                    "timestamp": time.strftime("%H:%M:%S"),
                }
                pf = Path(args.output_dir) / "progress.json"
                pf.parent.mkdir(parents=True, exist_ok=True)
                pf.write_text(json.dumps(progress))
            except Exception:
                pass
        except Exception:
            pass

    def on_train_end(self, args, state, control, **kw):
        try:
            elapsed = time.time() - (self.start_time or time.time())
            m, s = divmod(int(elapsed), 60)
            print()
            print(f"  Training finished in {m}m {s}s")
            if self.loss_history:
                print(f"  Final loss: {self.loss_history[-1]:.4f}")
            print()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, str]]:
    """Load and validate JSONL training data."""
    records = []
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    with open(path_obj, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d", i)
                continue
            if "instruction" not in rec or "response" not in rec:
                logger.warning("Skipping record at line %d (missing keys)", i)
                continue
            if not rec["instruction"].strip() or not rec["response"].strip():
                logger.warning("Skipping empty record at line %d", i)
                continue
            records.append(rec)

    logger.info("Loaded %d records from %s", len(records), path)
    return records


def format_for_sft(records: List[Dict[str, str]], tokenizer=None) -> Dataset:
    """Convert instruction records to chat-formatted text for SFTTrainer."""
    texts = []
    for rec in records:
        instruction = rec["instruction"]
        context = rec.get("context", "")
        response = rec["response"]

        user_part = instruction
        if context:
            user_part = f"[Reference Material]\n{context}\n\n[Question]\n{instruction}"

        system_prompt = (
            "You are an expert Data Structures and Algorithms (DSA) tutor.\n"
            "Instructions:\n"
            "- Structure your response with <think>...</think> tags for your internal analysis first, "
            "then provide the student-facing answer.\n"
            "- Give concrete code examples in Python when helpful.\n"
            "- Include time and space complexity analysis.\n"
            "- Be accurate. If you are unsure about something, say so.\n"
            "- Keep the response focused and well-structured."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_part},
            {"role": "assistant", "content": response},
        ]

        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            text = (
                f"[INST] {system_prompt}\n\n"
                f"{user_part} [/INST]{response}</s>"
            )
        texts.append(text)

    return Dataset.from_dict({"text": texts})


# ---------------------------------------------------------------------------
# Model loading (handles Windows WDDM quirks)
# ---------------------------------------------------------------------------

def _cleanup_gpu():
    """Force-free GPU memory between training rounds."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        freed_mb = torch.cuda.memory_reserved() / (1024**2)
        alloc_mb = torch.cuda.memory_allocated() / (1024**2)
        logger.info("GPU cleanup: allocated=%.0f MiB, reserved=%.0f MiB", alloc_mb, freed_mb)


def load_model_4bit(model_id: str):
    """Load model in 4-bit NF4 quantisation.

    IMPORTANT: Do NOT call torch.cuda.empty_cache() or synchronize()
    before this!  On Windows WDDM, those calls initialise the CUDA
    context (~4 GB overhead), causing the device-map planner to think
    the GPU is too small and offload layers to CPU.

    Note: The Python 3.14 safetensors patch is applied at module import time
    so that safe_open.get_slice() uses get_tensor() internally, avoiding the
    UntypedStorage access violation that occurred with transformers ≥ 4.51.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # If CUDA context is already initialised (i.e. between rounds),
    # the device-map planner underestimates free VRAM on Windows WDDM.
    # Pass explicit max_memory so it puts all layers on the GPU.
    max_memory = None
    if torch.cuda.is_available() and torch.cuda.memory_stats().get("num_alloc_retries", -1) != -1:
        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        max_memory = {0: f"{int(total_mb * 0.90)}MiB", "cpu": "16GiB"}
        logger.info("CUDA context active — explicit max_memory: %s", max_memory)

    logger.info("Loading %s in 4-bit NF4 ...", model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )

    gpu_mb = torch.cuda.memory_allocated() / (1024**2)
    logger.info("Model loaded. GPU memory used: %.0f MiB", gpu_mb)
    return model


# ---------------------------------------------------------------------------
# Single-round training
# ---------------------------------------------------------------------------

def train(
    data_path: str,
    output_dir: str,
    model_id: str = DEFAULT_MODEL_ID,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_steps: int = DEFAULT_MAX_STEPS,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    lr_scheduler_type: str = DEFAULT_LR_SCHEDULER,
    logging_steps: int = DEFAULT_LOGGING_STEPS,
    save_steps: int = DEFAULT_SAVE_STEPS,
    save_total_limit: int = DEFAULT_SAVE_TOTAL_LIMIT,
    target_modules: Optional[List[str]] = None,
    resume_from_adapter: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> tuple:
    """Run one round of LoRA fine-tuning.
    Returns (adapter_path, start_loss, end_loss)."""
    if target_modules is None:
        target_modules = list(DEFAULT_TARGET_MODULES)

    records = load_jsonl(data_path)
    if not records:
        raise ValueError(f"No valid training data in {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = format_for_sft(records, tokenizer)
    model = load_model_4bit(model_id)

    if resume_from_adapter and Path(resume_from_adapter).exists():
        logger.info("Resuming from adapter %s ...", resume_from_adapter)
        model = PeftModel.from_pretrained(model, resume_from_adapter, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        logger.info(
            "LoRA config: r=%d, alpha=%d, dropout=%.2f, targets=%s",
            lora_r, lora_alpha, lora_dropout, target_modules,
        )
        # Apply LoRA BEFORE SFTTrainer so that adapter weights are
        # created in fp32 (PEFT default).  Passing peft_config to
        # SFTTrainer instead would initialise them in bf16, and on
        # the RTX 2060 (no native bf16) this cascades into nan grads.
        model = get_peft_model(model, lora_config)

    # On resumed adapters the trainable LoRA params may come back as bf16.
    # Force them to fp32 for stability on RTX 2060 (no native bf16).
    n_lora_cast = 0
    for param in model.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
            n_lora_cast += 1
    if n_lora_cast:
        logger.info("Cast %d trainable params to fp32", n_lora_cast)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # No autocast / GradScaler.  Quantised layers already compute in
    # fp16 (bnb_4bit_compute_dtype).  LoRA params stay fp32 for stable
    # gradients; frozen bf16 params are cast to fp16 for Tensor-Core
    # speed on Turing.
    training_args = SFTConfig(
        output_dir=str(out),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        max_steps=max_steps,
        max_length=max_seq_len,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to="none",
        dataset_text_field="text",
        warmup_steps=warmup_steps,
        lr_scheduler_type=lr_scheduler_type,
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
    )

    progress_cb = LiveProgressCallback(total_steps=max_steps)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # peft_config NOT passed – LoRA already applied above
        processing_class=tokenizer,
        callbacks=[progress_cb],
    )

    # Trainer/accelerate wrapping can recast trainable LoRA params.
    # Enforce fp32 again here so gradients avoid bf16-only AMP paths.
    n_trainable_fp32 = 0
    for name, p in trainer.model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.to(torch.float32)
            n_trainable_fp32 += 1
    if n_trainable_fp32:
        logger.info("Re-cast %d wrapped trainable params to fp32", n_trainable_fp32)

    # Cast frozen bf16 → fp16 for speed (Tensor Cores on Turing).
    # Trainable LoRA params are fp32 – leave them for gradient stability.
    n_cast = 0
    for param in trainer.model.parameters():
        if param.dtype == torch.bfloat16 and not param.requires_grad:
            param.data = param.data.to(torch.float16)
            n_cast += 1
    if n_cast:
        logger.info("Cast %d frozen bf16 params to fp16", n_cast)

    # Log LoRA param dtypes for confirmation
    lora_dtypes = set()
    for name, p in trainer.model.named_parameters():
        if p.requires_grad:
            lora_dtypes.add(str(p.dtype))
    logger.info("Trainable param dtypes: %s", lora_dtypes)

    logger.info("Starting LoRA training (max_steps=%d) ...", max_steps)
    if resume_from_checkpoint:
        logger.info("Resuming from checkpoint: %s", resume_from_checkpoint)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint or None)

    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    logger.info("Adapter saved to %s", out)

    start_loss = progress_cb.loss_history[0] if progress_cb.loss_history else None
    end_loss = progress_cb.loss_history[-1] if progress_cb.loss_history else None

    del trainer
    del model
    _cleanup_gpu()

    return str(out), start_loss, end_loss


# ---------------------------------------------------------------------------
# Multi-round training
# ---------------------------------------------------------------------------

def train_rounds(
    data_path: str,
    base_output_dir: str,
    model_id: str = DEFAULT_MODEL_ID,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    target_modules: Optional[List[str]] = None,
    start_round: int = 1,
) -> str:
    """3-round training with decaying LR.  Returns final adapter path."""
    rounds = [
        {"lr": 2e-4, "steps": 400, "warmup": 40, "label": "Primary Learning"},
        {"lr": 5e-5, "steps": 150, "warmup": 15, "label": "Refinement"},
        {"lr": 2e-5, "steps": 75,  "warmup": 8,  "label": "Final Polish"},
    ]

    adapter_path = None
    if start_round > 1:
        prev_dir = f"{base_output_dir}/round_{start_round - 1}"
        if Path(prev_dir).exists() and (Path(prev_dir) / "adapter_model.safetensors").exists():
            adapter_path = prev_dir
            logger.info("Resuming from completed round %d adapter: %s", start_round - 1, adapter_path)

    for i, rnd in enumerate(rounds, 1):
        if i < start_round:
            continue
        round_dir = f"{base_output_dir}/round_{i}"

        print()
        print("+" + "=" * 58 + "+")
        print(f"|  ROUND {i}/{len(rounds)}: {rnd['label']:<48} |")
        print(f"|  LR: {rnd['lr']:<10} | Steps: {rnd['steps']:<10} | Warmup: {rnd['warmup']:<10}  |")
        print("+" + "=" * 58 + "+")
        print()

        adapter_path, start_loss, end_loss = train(
            data_path=data_path,
            output_dir=round_dir,
            model_id=model_id,
            max_seq_len=max_seq_len,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            batch_size=batch_size,
            grad_accum=grad_accum,
            learning_rate=rnd["lr"],
            max_steps=rnd["steps"],
            warmup_steps=rnd["warmup"],
            lr_scheduler_type="cosine",
            logging_steps=DEFAULT_LOGGING_STEPS,
            save_steps=DEFAULT_SAVE_STEPS,
            save_total_limit=DEFAULT_SAVE_TOTAL_LIMIT,
            target_modules=target_modules,
            resume_from_adapter=adapter_path,
        )

        print()
        print("-" * 60)
        print(f"  Round {i} Summary:")
        if start_loss is not None and end_loss is not None:
            improvement = ((start_loss - end_loss) / start_loss) * 100 if start_loss > 0 else 0
            print(f"    Start loss: {start_loss:.4f}")
            print(f"    End loss:   {end_loss:.4f}")
            print(f"    Improvement: {improvement:.1f}%")
        else:
            print("    Loss data unavailable")
        print("-" * 60)

    return adapter_path


# ---------------------------------------------------------------------------
# Post-training evaluation
# ---------------------------------------------------------------------------

EVAL_QUESTIONS = [
    "What is the time and space complexity of quicksort?",
    "How do you detect a cycle in a directed graph?",
    "Explain the sliding window technique with an example.",
    "What is the difference between Dijkstra and Bellman-Ford?",
    "How does a trie differ from a hash table for string storage?",
    "Implement an LRU cache.",
    "What is the Master Theorem and when does it apply?",
    "Explain dynamic programming with the coin change problem.",
    "How does Union-Find with path compression work?",
    "What is a monotonic stack and when do you use it?",
]


def evaluate(adapter_path: str, model_id: str = DEFAULT_MODEL_ID) -> None:
    """Inference on 10 eval questions with the trained adapter."""
    print()
    print("+" + "=" * 58 + "+")
    print("|  POST-TRAINING EVALUATION                                |")
    print("+" + "=" * 58 + "+")
    print()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model_4bit(model_id)

    logger.info("Loading adapter from %s ...", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    for i, question in enumerate(EVAL_QUESTIONS, 1):
        sep = "-" * 60
        print(f"\n{sep}")
        print(f"  EVAL {i}/{len(EVAL_QUESTIONS)}: {question}")
        print(sep)

        messages = [
            {"role": "system", "content": (
                "You are an expert Data Structures and Algorithms (DSA) tutor.\n"
                "Instructions:\n"
                "- Structure your response with <think>...</think> tags for your internal analysis first, "
                "then provide the student-facing answer.\n"
                "- Give concrete code examples in Python when helpful.\n"
                "- Include time and space complexity analysis.\n"
                "- Be accurate and well-structured."
            )},
            {"role": "user", "content": question},
        ]

        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = (
                f"[INST] You are an expert DSA tutor. Be concise and give examples.\n\n"
                f"{question} [/INST]"
            )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        answer = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        print(f"\n{answer.strip()}\n")

    del model
    _cleanup_gpu()

    print()
    print("  Evaluation complete.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for DSA Tutor")
    parser.add_argument("--data", default="finetune/data/combined_v2_training.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--multi-round", action="store_true")
    parser.add_argument("--start-round", type=int, default=1,
                        help="Round to start from (for resuming multi-round training)")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None,
                        help="Path to trainer checkpoint dir, or 'auto' to find latest")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or f"adapters/v_{ts}"

    # Resolve 'auto' checkpoint
    resume_ckpt = getattr(args, 'resume_from_checkpoint', None)
    if resume_ckpt == 'auto':
        import glob as _glob
        candidates = sorted(_glob.glob(f"{output_dir}/**/checkpoint-*", recursive=True))
        resume_ckpt = candidates[-1] if candidates else None
        if resume_ckpt:
            logger.info("Auto-detected checkpoint: %s", resume_ckpt)
        else:
            logger.info("No checkpoint found in %s, starting fresh", output_dir)
            resume_ckpt = None

    if args.multi_round:
        adapter_path = train_rounds(
            data_path=args.data,
            base_output_dir=output_dir,
            start_round=args.start_round,
        )
    else:
        adapter_path, start_loss, end_loss = train(
            data_path=args.data,
            output_dir=output_dir,
            max_steps=args.max_steps,
            resume_from_checkpoint=resume_ckpt,
        )
        if start_loss and end_loss:
            print(f"  Loss: {start_loss:.4f} -> {end_loss:.4f}")

    if args.evaluate:
        evaluate(adapter_path)


if __name__ == "__main__":
    main()
