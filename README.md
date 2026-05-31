# QLoRA Customer Support Chatbot

End-to-end fine-tuning and production deployment of **Llama 3.1 8B Instruct** as a customer support agent — fine-tuned on AWS SageMaker with QLoRA and tracked in MLflow, served via vLLM + FastAPI on EC2 behind an Application Load Balancer with Auto Scaling, and wired to a real-time escalation pipeline (SQS → Lambda → DynamoDB) that routes complaints, refunds, and payment issues to a human queue.

- **Live demo:** [iamshubhshrma/customer_agent on HF Spaces](https://huggingface.co/spaces/iamshubhshrma/customer_agent)
- **Adapter weights:** [iamshubhshrma/llama-3.1-8b-customer-support on HF Hub](https://huggingface.co/iamshubhshrma/llama-3.1-8b-customer-support)
- **Dataset:** [bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)

![Gradio chat demo](screenshots/demo-1.png)

![Gradio chat demo](screenshots/demo-2.png)

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
