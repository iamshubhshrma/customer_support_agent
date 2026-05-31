"""FastAPI wrapper around vLLM with intent detection and SQS escalation pipeline."""
import json
import os
import time
import uuid
from datetime import datetime, timezone

import boto3
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel

VLLM_BASE_URL = "http://localhost:8000/v1"
API_KEY       = os.environ.get("SUPPORT_API_KEY", "changeme")
SQS_QUEUE_URL = os.environ.get("SQS_QUEUE_URL", "")  # set before starting server
SYSTEM_PROMPT = (
    "You are a helpful customer support agent. "
    "Answer the customer's question clearly and politely."
)

INTENT_KEYWORDS: dict[str, list[str]] = {
    "cancel_order":           ["cancel", "cancellation", "cancelled"],
    "track_order":            ["track", "tracking", "shipment", "shipped", "delivery status"],
    "change_order":           ["change", "modify", "update", "amend"],
    "place_order":            ["place", "order", "purchase", "buy"],
    "get_refund":             ["refund", "reimburs", "money back"],
    "payment_issue":          ["payment", "charge", "billing", "invoice", "charged"],
    "check_cancellation_fee": ["cancellation fee", "fee", "penalty"],
    "contact_human_agent":    ["agent", "representative", "human", "speak to", "talk to"],
    "get_human_agent":        ["agent", "human", "representative", "speak to"],
    "complaint":              ["complaint", "unhappy", "dissatisfied", "frustrated", "sorry"],
    "delivery_period":        ["delivery time", "estimated delivery", "arrive", "arrival"],
    "recover_password":       ["password", "reset", "recover", "forgot"],
    "check_refund_policy":    ["refund policy", "return policy"],
    "delivery_options":       ["delivery option", "shipping option", "shipping method"],
}

# Intents that indicate the customer needs human intervention
ESCALATION_INTENTS = {
    "complaint",
    "contact_human_agent",
    "get_human_agent",
    "payment_issue",
    "get_refund",
    "check_cancellation_fee",
}

client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="vllm-internal")
sqs    = boto3.client("sqs", region_name="us-east-1")
app    = FastAPI(title="Customer Support API")


def detect_intent(text: str) -> str:
    lowered = text.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return intent
    return "unknown"


def verify_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


class GenerateRequest(BaseModel):
    message: str
    max_tokens: int = 256


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(req: GenerateRequest, _=Depends(verify_key)):
    start = time.time()

    response = await client.chat.completions.create(
        model="support-bot",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": req.message},
        ],
        max_tokens=req.max_tokens,
        temperature=0.0,
    )
    bot_reply  = response.choices[0].message.content.strip()
    latency_ms = int((time.time() - start) * 1000)
    intent     = detect_intent(req.message)

    if intent in ESCALATION_INTENTS and SQS_QUEUE_URL:
        payload = {
            "request_id":        str(uuid.uuid4()),
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "user_message":      req.message,
            "bot_response":      bot_reply,
            "detected_intent":   intent,
            "latency_ms":        latency_ms,
            "escalation_reason": intent,
        }
        sqs.send_message(QueueUrl=SQS_QUEUE_URL, MessageBody=json.dumps(payload))

    return {"response": bot_reply, "intent": intent, "latency_ms": latency_ms}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
