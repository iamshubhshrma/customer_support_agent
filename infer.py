"""Interactive inference demo — load adapter and chat in the terminal."""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from config.qlora_config import (
    BNB_4BIT_COMPUTE_DTYPE,
    BNB_4BIT_DOUBLE_QUANT,
    BNB_4BIT_QUANT_TYPE,
    EVAL_MAX_NEW_TOKENS,
    LOAD_IN_4BIT,
    MODEL_ID,
    OUTPUT_DIR,
    SYSTEM_PROMPT,
)


def load(adapter_path: str):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_double_quant=BNB_4BIT_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=BNB_4BIT_COMPUTE_DTYPE,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def respond(model, tokenizer, user_message: str) -> str:
    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_message}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=EVAL_MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    reply_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(reply_ids, skip_special_tokens=True).strip()


def main():
    adapter_path = sys.argv[1] if len(sys.argv) > 1 else f"{OUTPUT_DIR}/final"
    print(f"Loading adapter from {adapter_path} ...")
    model, tokenizer = load(adapter_path)
    print("Ready. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("Customer: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in {"quit", "exit", "q"}:
            break
        reply = respond(model, tokenizer, user_input)
        print(f"Agent:    {reply}\n")


if __name__ == "__main__":
    main()
