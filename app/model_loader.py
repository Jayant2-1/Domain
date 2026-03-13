"""Model Loader - lazy-loads a 4-bit quantised LLM (default: Phi-3.5-mini-instruct)
with optional LoRA adapter support.  Handles CUDA OOM gracefully and provides
an explicit unload() method that frees all GPU memory.

Design decisions
----------------
- No global model instance - the loader is instantiated and managed by the
  service layer (or FastAPI lifespan).
- Lazy: the model is not loaded until generate() or load() is called.
- 4-bit NF4 quantisation via bitsandbytes.
- LoRA adapters loaded with PEFT; can be hot-swapped without full reload.
- CUDA OOM is caught explicitly - clears cache and raises a typed error
  that the API layer can map to HTTP 503.
- Uses tokenizer.apply_chat_template() for proper prompt formatting.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)

# -- Defaults ----------------------------------------------------------------

DEFAULT_MODEL_ID = "models/Mistral-7B-Instruct-v0.3"
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_DEVICE_MAP = "auto"


# -- Custom exception --------------------------------------------------------

class ModelOOMError(RuntimeError):
    """Raised when CUDA runs out of memory during model operations."""


# -- Quantisation config -----------------------------------------------------

def _make_bnb_config() -> BitsAndBytesConfig:
    """Return a 4-bit NF4 quantisation config for bitsandbytes."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


# -- Model Loader ------------------------------------------------------------

class ModelLoader:
    """
    Manages the lifecycle of a 4-bit quantised causal LM.

    Usage
    -----
        loader = ModelLoader(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        # Model is NOT loaded yet (lazy).

        loader.load()                       # explicitly load now
        text = loader.generate("Hello")     # or auto-loads on first call

        loader.load_adapter("adapters/v_20260224/")  # hot-swap LoRA

        loader.unload()                     # free GPU memory
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device_map: str = DEFAULT_DEVICE_MAP,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens

        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._adapter_path: Optional[str] = None

    # -- Properties ------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def current_adapter(self) -> Optional[str]:
        return self._adapter_path

    # -- Load / Unload ---------------------------------------------------

    def load(self) -> None:
        """
        Load the base model in 4-bit and its tokenizer.

        If already loaded, this is a no-op.
        Catches torch.cuda.OutOfMemoryError and re-raises as ModelOOMError
        after clearing the CUDA cache.
        """
        if self.is_loaded:
            logger.info("Model already loaded - skipping.")
            return

        logger.info("Loading model %s (4-bit NF4) …", self.model_id)
        try:
            bnb_config = _make_bnb_config()

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Explicit max_memory prevents accelerate from offloading layers
            # to CPU when estimating 90% of VRAM (which undershoots on RTX 2060).
            max_memory = None
            if torch.cuda.is_available():
                free_vram = torch.cuda.get_device_properties(0).total_memory
                max_memory = {0: f"{int(free_vram * 0.92 / 1024**2)}MiB", "cpu": "16GiB"}

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map=self.device_map,
                max_memory=max_memory,
                trust_remote_code=True,
                attn_implementation="eager",
            )
            self._model.eval()

            vram_gb = torch.cuda.memory_allocated(0) / 1e9
            logger.info("Model loaded successfully. VRAM used: %.2f GB", vram_gb)

        except torch.cuda.OutOfMemoryError:
            self._handle_oom("loading model")

    def unload(self) -> None:
        """
        Delete the model and tokenizer, free GPU memory.
        Safe to call even if nothing is loaded.
        """
        logger.info("Unloading model …")
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._adapter_path = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded - GPU memory freed.")

    # -- Adapter management ----------------------------------------------

    def load_adapter(self, adapter_path: str) -> None:
        """
        Load a PEFT / LoRA adapter on top of the base model.

        Parameters
        ----------
        adapter_path : str
            Path to the adapter directory (contains adapter_config.json etc.).

        Raises
        ------
        FileNotFoundError
            If the adapter directory does not exist.
        RuntimeError
            If the base model is not loaded.
        ModelOOMError
            On CUDA OOM.
        """
        if not self.is_loaded:
            raise RuntimeError("Base model must be loaded before loading an adapter.")

        path = Path(adapter_path)
        if not path.exists():
            raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

        logger.info("Loading adapter from %s …", adapter_path)
        try:
            from peft import PeftModel

            # If an adapter is already loaded, unload it first
            if self._adapter_path is not None:
                logger.info("Unloading previous adapter: %s", self._adapter_path)
                self._model = self._model.base_model.model  # type: ignore[union-attr]

            self._model = PeftModel.from_pretrained(self._model, adapter_path)
            self._model.eval()
            self._adapter_path = str(path)
            logger.info("Adapter loaded: %s", adapter_path)

        except torch.cuda.OutOfMemoryError:
            self._handle_oom("loading adapter")

    def unload_adapter(self) -> None:
        """Remove the LoRA adapter, reverting to the base model."""
        if self._adapter_path is None:
            logger.info("No adapter loaded - nothing to unload.")
            return
        logger.info("Unloading adapter: %s", self._adapter_path)
        self._model = self._model.base_model.model  # type: ignore[union-attr]
        self._adapter_path = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Adapter unloaded.")

    # -- Generation ------------------------------------------------------

    def generate(
        self,
        prompt: str | None = None,
        messages: list | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 0.4,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
    ) -> str:
        """
        Generate a response from the model.

        Auto-loads the model on first call (lazy loading).

        Parameters
        ----------
        prompt : str | None
            A raw prompt string (legacy).  Prefer `messages`.
        messages : list | None
            A list of {"role": ..., "content": ...} dicts.
            If provided, tokenizer.apply_chat_template() is used
            to format the prompt correctly for the loaded model.
        max_new_tokens : int | None
            Overrides default max_new_tokens for this call.
        temperature : float
            Sampling temperature (lower = more focused).
        top_p : float
            Nucleus sampling threshold.
        repetition_penalty : float
            Penalise repeated tokens (>1 = penalise, 1.0 = off).

        Returns
        -------
        str
            The generated text (prompt is stripped from output).

        Raises
        ------
        ModelOOMError
            On CUDA OOM during generation.
        """
        # Lazy load
        if not self.is_loaded:
            self.load()

        tokens_limit = max_new_tokens if max_new_tokens is not None else self.max_new_tokens

        try:
            # Use chat template if messages provided, else raw prompt
            if messages is not None:
                tokenized = self._tokenizer.apply_chat_template(  # type: ignore[union-attr]
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True,
                )
                # apply_chat_template may return a tensor, BatchEncoding, or dict
                if isinstance(tokenized, dict):
                    input_ids = tokenized["input_ids"]
                elif hasattr(tokenized, "input_ids"):
                    input_ids = tokenized.input_ids
                else:
                    input_ids = tokenized

                # Ensure 2-D [batch, seq_len]
                if hasattr(input_ids, "dim") and input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                elif input_ids.ndim == 1:
                    input_ids = input_ids.unsqueeze(0)

                input_ids = input_ids.to(self._model.device)  # type: ignore[union-attr]
                attention_mask = torch.ones_like(input_ids)
                inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            else:
                enc = self._tokenizer(prompt, return_tensors="pt")  # type: ignore[union-attr]
                inputs = {k: v.to(self._model.device) for k, v in enc.items()}  # type: ignore[union-attr]

            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = self._model.generate(  # type: ignore[union-attr]
                    **inputs,
                    max_new_tokens=tokens_limit,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.pad_token_id,  # type: ignore[union-attr]
                )

            generated_ids = outputs[0][input_len:]
            text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()  # type: ignore[union-attr]
            return text

        except torch.cuda.OutOfMemoryError:
            self._handle_oom("generating response")
            return ""  # unreachable but satisfies type checker

    # -- Internal helpers ------------------------------------------------

    def _handle_oom(self, context: str) -> None:
        """Clear CUDA cache and raise ModelOOMError."""
        logger.error("CUDA OOM during %s - clearing cache.", context)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.unload()
        raise ModelOOMError(
            f"CUDA out of memory while {context}. "
            "Model has been unloaded. Retry or reduce input length."
        )
