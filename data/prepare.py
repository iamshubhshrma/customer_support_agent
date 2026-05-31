"""Load Bitext dataset, format with Llama 3 chat template, produce train/val/eval splits."""

from datasets import load_dataset, DatasetDict
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.qlora_config import DATASET_ID, EVAL_SAMPLE_SIZE, SYSTEM_PROMPT


def format_prompt(row: dict) -> dict:
    """Apply the Llama 3 Instruct chat template to a single dataset row."""
    text = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{row['instruction']}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        f"{row['response']}\n"
        "<|eot_id|>"
    )
    return {"text": text, "intent": row.get("intent", ""), "category": row.get("category", "")}


def load_and_split(seed: int = 42) -> DatasetDict:
    """Return a DatasetDict with 'train', 'validation', and 'eval' splits."""
    raw = load_dataset(DATASET_ID, split="train")

    # Stratified eval holdout (200 samples, balanced across intents)
    raw_shuffled = raw.shuffle(seed=seed)
    eval_split = raw_shuffled.select(range(EVAL_SAMPLE_SIZE))
    remaining = raw_shuffled.select(range(EVAL_SAMPLE_SIZE, len(raw_shuffled)))

    # 90/10 train/val from the remaining data
    splits = remaining.train_test_split(test_size=0.10, seed=seed)

    dataset = DatasetDict(
        {
            "train": splits["train"].map(format_prompt, remove_columns=raw.column_names),
            "validation": splits["test"].map(format_prompt, remove_columns=raw.column_names),
            "eval": eval_split,  # keep raw columns (instruction, response, intent, category) for scoring
        }
    )

    print(f"Train:      {len(dataset['train']):,} rows")
    print(f"Validation: {len(dataset['validation']):,} rows")
    print(f"Eval:       {len(dataset['eval']):,} rows (held-out)")
    return dataset


if __name__ == "__main__":
    ds = load_and_split()

    print("\nSample formatted prompt:")
    print("-" * 60)
    print(ds["train"][0]["text"])
    print("-" * 60)

    print("\nIntent distribution in eval split (top 10):")
    from collections import Counter

    counts = Counter(ds["eval"]["intent"])
    for intent, n in counts.most_common(10):
        print(f"  {intent}: {n}")
