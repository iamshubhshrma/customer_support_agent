"""Single source of truth for all QLoRA training hyperparameters."""

import torch

# ── Model & Dataset ──────────────────────────────────────────────────────────
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
OUTPUT_DIR = "./outputs"
HF_REPO_ID = None  # Set to "your-username/qlora-support-bot" before pushing

# ── BitsAndBytes 4-bit NF4 ───────────────────────────────────────────────────
LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_DOUBLE_QUANT = True
# T4 (Volta) → use float16; Ampere+ (A100, 3090, 4090) → use bfloat16
BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# ── LoRA Adapter ─────────────────────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32          # effective LR scaling = alpha / r = 2.0
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
LORA_BIAS = "none"
LORA_TASK_TYPE = "CAUSAL_LM"

# ── SFTTrainer / Training ─────────────────────────────────────────────────────
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4   # effective batch = 8; bump to 8 if OOM
GRADIENT_CHECKPOINTING = True
LEARNING_RATE = 2e-4
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.03
MAX_GRAD_NORM = 0.3
WEIGHT_DECAY = 0.001
OPTIM = "paged_adamw_32bit"
MAX_SEQ_LENGTH = 512
PACKING = False            # requires flash_attention_2 to avoid cross-contamination; disabled by default

# Use bf16 only if the GPU supports it (T4 does not)
BF16 = torch.cuda.is_bf16_supported()
FP16 = not BF16

# ── Logging / Checkpointing ───────────────────────────────────────────────────
LOGGING_STEPS = 10
SAVE_STEPS = 400   # must be a multiple of EVAL_STEPS
EVAL_STEPS = 200
EVAL_STRATEGY = "steps"

# ── MLflow ───────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "qlora-customer-support"

# ── Evaluation ───────────────────────────────────────────────────────────────
EVAL_SAMPLE_SIZE = 200
EVAL_MAX_NEW_TOKENS = 256

# ── System prompt (used in data/prepare.py and infer.py) ─────────────────────
SYSTEM_PROMPT = (
    "You are a helpful customer support agent. "
    "Answer the customer's question clearly and politely."
)
