"""Evaluate the fine-tuned adapter: ROUGE-L + intent accuracy on the 200-sample held-out set."""

import json
import os
import re
import sys

import torch
from datasets import Dataset
from rouge_score import rouge_scorer
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
from data.prepare import load_and_split

RESULTS_PATH = "./results/eval_results.json"

# Keyword patterns per intent — used for intent accuracy scoring
INTENT_KEYWORDS: dict[str, list[str]] = {
    "cancel_order": ["cancel", "cancellation", "cancelled"],
    "track_order": ["track", "tracking", "shipment", "shipped", "delivery status"],
    "change_order": ["change", "modify", "update", "amend"],
    "place_order": ["place", "order", "purchase", "buy"],
    "get_refund": ["refund", "reimburs", "money back"],
    "payment_issue": ["payment", "charge", "billing", "invoice", "charged"],
    "check_cancellation_fee": ["cancellation fee", "fee", "penalty"],
    "contact_human_agent": ["agent", "representative", "human", "speak to", "talk to"],
    "complaint": ["complaint", "unhappy", "dissatisfied", "frustrated", "apologize", "sorry"],
    "review": ["review", "feedback", "rating"],
    "delivery_options": ["delivery option", "shipping option", "shipping method"],
    "delivery_period": ["delivery time", "estimated delivery", "arrive", "arrival"],
    "check_invoices": ["invoice", "receipt", "bill"],
    "get_invoice": ["invoice", "receipt", "bill"],
    "recover_password": ["password", "reset", "recover", "forgot"],
    "registration_problems": ["register", "signup", "account creation", "sign up"],
    "edit_account": ["update account", "edit account", "change account", "profile"],
    "delete_account": ["delete account", "close account", "remove account", "deactivate"],
    "newsletter_subscription": ["newsletter", "subscribe", "unsubscribe", "email opt"],
    "switch_account": ["switch account", "change account", "different account"],
    "set_up_shipping_address": ["shipping address", "delivery address", "address"],
    "check_payment_methods": ["payment method", "payment option", "pay with"],
    "check_refund_policy": ["refund policy", "return policy"],
    "contact_customer_service": ["contact", "reach", "customer service", "support"],
    "get_human_agent": ["agent", "human", "representative", "speak to"],
    "create_account": ["create account", "open account", "new account", "register"],
}


def detect_intent_from_response(response: str, expected_intent: str) -> bool:
    """Return True if any keyword for expected_intent appears in the response."""
    keywords = INTENT_KEYWORDS.get(expected_intent, [])
    if not keywords:
        return False
    lowered = response.lower()
    return any(kw in lowered for kw in keywords)


def generate_response(model, tokenizer, instruction: str) -> str:
    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{instruction}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=EVAL_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_eval(adapter_path: str | None = None):
    if adapter_path is None:
        adapter_path = f"{OUTPUT_DIR}/final"

    print(f"Loading base model {MODEL_ID} in 4-bit...")
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

    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    print("Loading eval split...")
    dataset = load_and_split()
    eval_ds: Dataset = dataset["eval"]

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rouge_scores, intent_hits = [], []
    for i, row in enumerate(eval_ds):
        instruction = row["instruction"]
        reference = row["response"]
        intent = row.get("intent", "")

        prediction = generate_response(model, tokenizer, instruction)

        score = scorer.score(reference, prediction)["rougeL"].fmeasure
        rouge_scores.append(score)

        hit = detect_intent_from_response(prediction, intent)
        intent_hits.append(hit)

        if (i + 1) % 20 == 0:
            print(
                f"  [{i+1}/{len(eval_ds)}] "
                f"avg ROUGE-L={sum(rouge_scores)/len(rouge_scores):.3f}  "
                f"intent_acc={sum(intent_hits)/len(intent_hits):.3f}"
            )

    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    intent_acc = sum(intent_hits) / len(intent_hits)

    results = {
        "adapter_path": adapter_path,
        "num_samples": len(eval_ds),
        "rouge_l": round(avg_rouge, 4),
        "intent_accuracy": round(intent_acc, 4),
    }

    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Evaluation Results ===")
    print(f"ROUGE-L:          {avg_rouge:.4f}")
    print(f"Intent Accuracy:  {intent_acc:.4f}")
    print(f"Results saved to  {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    adapter = sys.argv[1] if len(sys.argv) > 1 else None
    run_eval(adapter)
