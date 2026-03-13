"""Quick test: Download TinyLlama 4-bit and run one inference."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("Loading TinyLlama 4-bit quantized...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print(f"Model loaded! VRAM used: {torch.cuda.memory_allocated(0)/1e9:.3f} GB")

prompt = "<|user|>\nExplain binary search in simple terms.\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True)
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("--- MODEL RESPONSE ---")
print(response)
print("--- END ---")
print("INFERENCE_OK")
