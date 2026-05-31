"""QLoRA fine-tuning script — mirrors the main notebook cells."""

import os
import mlflow
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from config.qlora_config import (
    BF16,
    BNB_4BIT_COMPUTE_DTYPE,
    BNB_4BIT_DOUBLE_QUANT,
    BNB_4BIT_QUANT_TYPE,
    EVAL_STEPS,
    EVAL_STRATEGY,
    FP16,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CHECKPOINTING,
    LEARNING_RATE,
    LOAD_IN_4BIT,
    LOGGING_STEPS,
    LORA_ALPHA,
    LORA_BIAS,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    LORA_TASK_TYPE,
    LR_SCHEDULER_TYPE,
    MAX_GRAD_NORM,
    MAX_SEQ_LENGTH,
    MLFLOW_EXPERIMENT,
    MODEL_ID,
    NUM_TRAIN_EPOCHS,
    OPTIM,
    OUTPUT_DIR,
    PACKING,
    PER_DEVICE_EVAL_BATCH_SIZE,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    SAVE_STEPS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from data.prepare import load_and_split


def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_double_quant=BNB_4BIT_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=BNB_4BIT_COMPUTE_DTYPE,
    )


def load_model_and_tokenizer():
    bnb_cfg = build_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # required for gradient checkpointing
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def attach_lora(model):
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias=LORA_BIAS,
        task_type=LORA_TASK_TYPE,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def build_training_args() -> SFTConfig:
    # SFTConfig = TrainingArguments + SFT-specific params (TRL >=0.12)
    return SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        weight_decay=WEIGHT_DECAY,
        optim=OPTIM,
        bf16=BF16,
        fp16=FP16,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy=EVAL_STRATEGY,
        eval_steps=EVAL_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="mlflow",
        run_name="qlora-support-bot",
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,   # renamed from max_seq_length in TRL 0.24
        packing=PACKING,
    )


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ["MLFLOW_EXPERIMENT_NAME"] = MLFLOW_EXPERIMENT
    os.environ["MLFLOW_RUN_NAME"] = "qlora-support-bot"

    print("Loading dataset...")
    dataset = load_and_split()

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()

    print("Attaching LoRA adapters...")
    model = attach_lora(model)

    training_args = build_training_args()

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {OUTPUT_DIR}/final ...")
    trainer.model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print("Done.")

    mlflow.end_run()


if __name__ == "__main__":
    train()
