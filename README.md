# QLoRA Customer Support Chatbot

End-to-end fine-tuning and production deployment of **Llama 3.1 8B Instruct** as a customer support agent — fine-tuned on AWS SageMaker with QLoRA and tracked in MLflow, served via vLLM + FastAPI on EC2 behind an Application Load Balancer with Auto Scaling, and wired to a real-time escalation pipeline (SQS → Lambda → DynamoDB) that routes complaints, refunds, and payment issues to a human queue.

- **Live demo:** [iamshubhshrma/customer_agent on HF Spaces](https://huggingface.co/spaces/iamshubhshrma/customer_agent)
- **Adapter weights:** [iamshubhshrma/llama-3.1-8b-customer-support on HF Hub](https://huggingface.co/iamshubhshrma/llama-3.1-8b-customer-support)
- **Dataset:** [bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)

![Gradio chat demo](screenshots/demo-1.png)

![Gradio chat demo](screenshots/demo-2.png)

---

## Project Q&A

### What problem is this project solving?

Customer support is one of the highest-volume, most repetitive cost centers in e-commerce. A large share of inbound tickets are variations on a small set of intents — *where is my order, I want to cancel, I was charged twice, how do I reset my password, I want a refund* — yet they still consume human agent time and drive up response latency and operational cost. Generic, off-the-shelf LLMs can answer these but tend to be verbose, inconsistent in tone, and unaware of how a real support desk is expected to respond.

This project solves that problem on two fronts:

1. **Domain specialization without the cost of a giant model.** It fine-tunes **Llama 3.1 8B Instruct** on ~24k real-world-style support conversations (the Bitext dataset, covering 27 intents across 11 categories) so the model answers in the clear, polite, on-brand style of a support agent. Because it uses **QLoRA** (4-bit quantization + small LoRA adapters), the entire fine-tune fits on a single 24 GB GPU and produces a ~110 MB adapter instead of a multi-gigabyte model — cheap to train (the whole build cost under $10), cheap to store, and trivial to hot-swap onto a shared base model at serve time.

2. **Knowing when *not* to answer.** A chatbot that confidently handles a refund dispute or a double-charge complaint on its own is a liability. The system therefore pairs the model with a real-time **escalation pipeline**: every request is scanned for sensitive intents (`complaint`, `payment_issue`, `get_refund`, `contact_human_agent`, `get_human_agent`, `check_cancellation_fee`) and those are routed through SQS → Lambda → DynamoDB into a human follow-up queue. The bot deflects the routine, high-volume questions automatically while guaranteeing that money- and trust-sensitive cases reach a person.

The end result is a production-shaped reference implementation showing how to take an open-weight LLM from raw dataset to a deployed, auto-scaling, human-in-the-loop support agent.

### What challenges were faced during this project and how were they tackled?

**Fitting an 8B model onto a single GPU.** Full fine-tuning of an 8B model needs far more than the 24 GB available on the A10G used for training. This was solved with QLoRA: the frozen base model is loaded in **4-bit NF4 with double quantization** (cutting its footprint ~4×), and only LoRA adapters on the attention and MLP projection layers are trained — roughly 0.75% of parameters. Combined with gradient checkpointing and `paged_adamw_32bit`, peak training VRAM stays at ~18–20 GB, comfortably under the 24 GB ceiling, and a full epoch on 24k samples runs in ~4.4 hours.

**Hardware constraints between dev and production.** The local dev box (RTX 4060 Laptop, 8 GB) can run 8B *inference* but cannot *train* 8B. The fix was a split workflow: develop and smoke-test on a smaller 3B model locally, then run the real 8B training job on AWS SageMaker `ml.g5.2xlarge`. The config auto-detects precision (`bf16` on Ampere+, `fp16` on T4) via `torch.cuda.is_bf16_supported()` so the same code runs correctly across T4, A10G, and 4090 without hardcoding.

**Chat-template and label-masking correctness.** Early on, using a raw `### Human:/### Assistant:` format mismatches the Llama 3 Instruct model's expectations and degrades quality. The project standardizes on the exact Llama 3 Instruct template (`<|begin_of_text|>…<|eot_id|>`) and computes loss on **assistant tokens only** (masking prompt and `<pad>` tokens) so the model learns to respond rather than to parrot the prompt.

**Avoiding sample cross-contamination.** Sequence packing maximizes GPU utilization but, without FlashAttention-2, lets tokens from one example attend to another. Since flash_attn isn't installed in this environment, **packing was disabled** (`PACKING = False`) to keep training signal clean, trading a little throughput for correctness.

**Serving the adapter efficiently and safely.** Rather than merging and shipping a 6+ GB model, only the small adapter is pushed to S3 and HF Hub and **hot-loaded onto the base model by vLLM** at serve time, behind a FastAPI wrapper. To handle variable load, the inference fleet runs in an Auto Scaling Group behind an Application Load Balancer with `/health` checks. The escalation path was built to be dependency-free — a lightweight inline keyword detector inside FastAPI publishes to SQS — so it adds essentially zero latency and no extra ML inference to the request path.

**Keeping VRAM and OOM under control as a repeatable procedure.** OOM during fine-tuning is the most common failure mode, so the project bakes in a documented escape hatch: drop `per_device_train_batch_size` to 1 and raise `gradient_accumulation_steps` to 8 to preserve the effective batch size, plus always pass `device_map="auto"` and set `model.config.use_cache = False` when gradient checkpointing is on (the two are incompatible).

### What technologies and frameworks were used in the project?

**Model training & fine-tuning**

- **PyTorch** + **Hugging Face Transformers** (≥4.40) — base model loading and the Llama 3.1 8B Instruct architecture.
- **PEFT** (≥0.10) — `LoraConfig` / `get_peft_model` for the LoRA adapters.
- **BitsAndBytes** (≥0.43) — 4-bit NF4 quantization with double quant.
- **TRL** (≥0.8) — `SFTTrainer` / `SFTConfig` to run supervised fine-tuning.
- **Datasets** (≥2.18) — loading and splitting the Bitext customer-support dataset.

**Evaluation & experiment tracking**

- **MLflow** — automatic logging of hyperparameters, training/validation loss, and token accuracy (driven purely by `report_to="mlflow"`, no `mlflow.init()`).
- **rouge_score** — ROUGE-L for generation quality, plus a keyword-based intent-accuracy check.

**Serving & API**

- **vLLM** (0.21) — continuous batching and LoRA hot-loading for high-throughput inference.
- **FastAPI** + **Uvicorn** — the request wrapper, intent detection, and escalation trigger.
- **Gradio** on **Hugging Face Spaces** — the public chat demo, calling the EC2 API through the load balancer.

**AWS infrastructure**

- **SageMaker** (`ml.g5.2xlarge`, A10G 24 GB) — the training compute.
- **EC2** (`g5.2xlarge` spot) + **Auto Scaling Group** + **Application Load Balancer** — the auto-scaling inference fleet.
- **S3** — adapter storage.
- **SQS → Lambda (Python 3.12) → DynamoDB** — the real-time escalation pipeline (queue, consumer, and structured log store).
- **CloudFormation** (`infrastructure.json`) — the stack as code.

**Model & artifact distribution**

- **Hugging Face Hub** (`huggingface_hub`) — hosting the LoRA adapter weights and model card.
- **Bitext Customer Support dataset** (CC-BY-4.0) — the training data.

---

## Architecture

![Architecture diagram](screenshots/architecture.png)

Three flows make up the system:

1. **Training** (one-time, SageMaker) — `train.py` runs QLoRA fine-tuning on `ml.g5.2xlarge`. Loss curves, eval metrics, hyperparameters, and the final adapter path are logged to **MLflow**. The adapter is uploaded to S3 and pushed to HF Hub at the end of the run.
2. **Request path** — Customer → Gradio (HF Spaces) → ALB `qlora-alb:8080` → Target Group `qlora-tg` → a FastAPI instance in ASG `qlora-asg` → vLLM (Llama 3.1 8B + `support-bot` LoRA).
3. **Escalation path** — FastAPI keyword-detects intent on every request. Escalation intents (`complaint`, `payment_issue`, `get_refund`, `contact_human_agent`, `get_human_agent`, `check_cancellation_fee`) are published to SQS `qlora-support-escalations` → consumed by Lambda `qlora-escalation-handler` → written to DynamoDB `qlora-support-logs` for human follow-up.

Full architecture as code: [`infrastructure.json`](infrastructure.json) (CloudFormation). Source diagram: [`architecture.md`](architecture.md) (Mermaid).

---

## Results

| Metric | Value |
|--------|-------|
| Base model | `meta-llama/Llama-3.1-8B-Instruct` |
| Train loss (final) | **0.5586** |
| Eval loss | **0.4950** |
| Eval token accuracy | **83.2%** |
| Training time | **4.4 hours** on a single A10G (24 GB) |
| Peak training VRAM | ~18–20 GB |
| Trainable params | ~0.75% of total |
| Adapter size | ~110 MB |

To regenerate ROUGE-L and intent-accuracy numbers on the 200-sample held-out set:

```bash
python evaluate.py outputs/final
```

---

## Why QLoRA

QLoRA combines two ideas to make 8B fine-tuning fit on a single GPU:

- **4-bit NF4 quantization** of the frozen base model via BitsAndBytes — drops the base model's memory footprint by ~4× while preserving accuracy through NormalFloat-4, a quantization format tuned to the empirical weight distribution.
- **LoRA adapters** on the attention and MLP projection matrices — only ~0.75% of the parameters are trainable, gradients only flow through the adapters, and the result is a tiny ~110 MB adapter file that can be swapped in/out of the base model at serve time.

The result: peak training VRAM stays under 20 GB on an A10G, a full epoch on 24k samples runs in ~4.4 hours, and the deployable artifact is small enough to push to Hugging Face Hub.

---

## Quick start

### Training, evaluation, and local inference

```bash
pip install -r requirements.txt
python data/prepare.py   # verify dataset loads
python train.py          # 8B needs ~24 GB VRAM (use ml.g5.2xlarge)
python evaluate.py       # ROUGE-L + intent accuracy on 200-sample holdout
python infer.py          # interactive CLI against the saved adapter
```

### Serving (EC2 inference instance)

```bash
sudo apt-get install -y ninja-build      # required for FlashInfer JIT
pip install -r requirements-serve.txt
python api_server.py                     # FastAPI on :8080, vLLM on :8000
```

### Full production pipeline

SageMaker training → S3/HF Hub push → EC2 vLLM + FastAPI → SQS + Lambda + DynamoDB → ALB + ASG → HF Spaces. Two step-by-step walkthroughs are included:

- **[`w2.md`](w2.md)** — via the AWS Console (recommended for first run)
- **[`WORKFLOW.md`](WORKFLOW.md)** — via the AWS CLI

---

## Experiment tracking with MLflow

Training is instrumented end-to-end with MLflow — no `mlflow.init()` call is needed in code. The TRL `SFTTrainer` reads `report_to="mlflow"` from `SFTConfig` and picks up the experiment name from the environment.

**What gets logged automatically:**

- Hyperparameters (LoRA rank, alpha, learning rate, batch size, optimizer, schedule)
- Training loss every 10 steps
- Validation loss + token accuracy every 200 steps
- Final adapter path

**Set the experiment + run name via env vars before training:**

```bash
export MLFLOW_EXPERIMENT_NAME="qlora-customer-support"
export MLFLOW_RUN_NAME="qlora-8b-sagemaker"
# Optional — point at a remote tracking server:
# export MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"

python train.py
```

By default, runs are written to a local SQLite store. To browse them:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

---

## Configuration

All hyperparameters live in one file: [`config/qlora_config.py`](config/qlora_config.py).

| Hyperparameter | Value |
|----------------|-------|
| Quantization | 4-bit NF4, double quant (BitsAndBytes) |
| LoRA rank / alpha | 16 / 32 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Effective batch size | 8 (2 × grad_accum 4) |
| Learning rate | 2e-4, cosine schedule, 3% warmup |
| Epochs | 1 |
| Max sequence length | 512 |
| Packing | Disabled (avoids cross-contamination without flash_attn) |
| Optimizer | `paged_adamw_32bit` |
| Mixed precision | bf16 on Ampere+, fp16 on T4 (auto-detected) |
| Experiment tracking | MLflow |

---

## Prompt format

The training script uses the Llama 3 Instruct chat template:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful customer support agent. Answer the customer's question clearly and politely.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{customer message}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{agent response}
<|eot_id|>
```

---

## AWS stack

| Component | Resource |
|-----------|----------|
| Training | SageMaker Studio · `ml.g5.2xlarge` (1× A10G) |
| Adapter storage | S3 · `iamshubhshrma-customeragent/adapters/final/` |
| Inference fleet | Auto Scaling Group `qlora-asg` (min 1 / desired 2 / max 3) of `g5.2xlarge` from custom AMI `qlora-vllm-ami` |
| Load balancing | Application Load Balancer `qlora-alb` :8080 → Target Group `qlora-tg` (health check `/health`) |
| Inference server | vLLM 0.21 (continuous batching, LoRA hot-load) + FastAPI |
| Escalation queue | SQS `qlora-support-escalations` |
| Escalation consumer | Lambda `qlora-escalation-handler` (Python 3.12) |
| Escalation store | DynamoDB `qlora-support-logs` (PK: `request_id`) |
| Public demo | Hugging Face Spaces (Gradio) calling the ALB endpoint |

Total one-time cost to build: **< $10** (SageMaker training ~$7 + EC2 demo ~$3).

---

## Repository layout

```
.
├── README.md                  # this file
├── PROJECT.md                 # design spec / source of truth
├── WORKFLOW.md                # production pipeline (AWS CLI)
├── w2.md                      # production pipeline (AWS Console)
├── architecture.md            # Mermaid architecture diagram
├── infrastructure.json        # CloudFormation template
├── CLAUDE.md                  # repo guidance for AI-assisted development
├── requirements.txt           # training / eval / inference deps
├── requirements-serve.txt     # vLLM + FastAPI serving deps
├── train.py                   # SFT training entrypoint
├── evaluate.py                # ROUGE-L + intent accuracy on 200-sample holdout
├── infer.py                   # interactive CLI
├── api_server.py              # FastAPI + vLLM + SQS escalation
├── notebook.ipynb             # Colab/Kaggle artifact (mirrors train.py)
├── config/qlora_config.py     # all hyperparameters
├── data/prepare.py            # Bitext load + chat template + splits
├── results/eval_results.json  # written by evaluate.py
├── files/                     # utility scripts (e.g. S3 sync)
└── screenshots/               # demo + architecture images for this README
```

---

## Documentation guide

| If you want to… | Read |
|----------------|------|
| Understand the system end-to-end | this file + [`PROJECT.md`](PROJECT.md) |
| Reproduce the AWS deployment via the Console | [`w2.md`](w2.md) |
| Reproduce the AWS deployment via the CLI | [`WORKFLOW.md`](WORKFLOW.md) |
| See the architecture as code | [`infrastructure.json`](infrastructure.json) |
| See the architecture as a diagram source | [`architecture.md`](architecture.md) |
| Work on this repo with Claude Code | [`CLAUDE.md`](CLAUDE.md) |

---

## Roadmap

| Version | Status |
|---------|--------|
| v1.0 — QLoRA SFT on 3B, adapter on HF Hub | Done |
| v1.1 — 8B upgrade, SageMaker training, vLLM API on EC2 | Done |
| v1.2 — SQS escalation pipeline + Lambda → DynamoDB | Done |
| v1.3 — ALB + ASG fleet, Gradio demo on HF Spaces | Done |

---

## License

- Base model — [Meta Llama 3.1 Community License](https://llama.meta.com/llama3_1/license/)
- Dataset — CC-BY-4.0
- Adapter weights and code in this repo — [MIT](LICENSE)
