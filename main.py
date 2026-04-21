from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
import json
import re
import logging
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis Agent",
    description="A2A-compliant sentiment analysis agent using Groq + Llama3",
    version="1.0.0"
)

# Groq config — reads from environment variable (set on Render, or in .env locally)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME   = "llama-3.1-8b-instant"    # fast, free, better than Phi-3-mini

AGENT_ID      = "sentiment-analysis-agent-v1"
AGENT_VERSION = "1.0.0"

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not set — /analyze calls will fail.")
else:
    logger.info(f"✅ Ready. Using Groq model: {MODEL_NAME}")

# ── A2A Agent Card ─────────────────────────────────────────────────────────────
AGENT_CARD = {
"name": "sentiment-analysis-agent",
  "url": "https://sentiment-agent-palq.onrender.com/",
  "version": "1.0.0",
  "protocolVersion": "1.0.0",
  "description": "Analyzes customer email or text to determine sentiment, confidence score, and churn risk. Helps downstream systems decide retention strategies and customer engagement actions.",
  "skills": [
    {
      "id": "1",
      "name": "Customer Sentiment Analysis",
      "description": "Analyzes raw customer text (emails, messages) to classify sentiment (positive, negative, neutral), estimate confidence score, and detect churn risk indicators. Provides reasoning to support the classification.",
      "tags": [
        "sentiment",
        "churn",
        "nlp",
        "customer-experience"
      ],
      "examples": [
        "I am very unhappy with your service and want to cancel my subscription.",
        "Everything is great, thank you for the support!",
        "The product is okay but delivery was late."
      ],
      "inputModes": [
        "application/json",
        "text/plain"
      ]
    }
  ],
  "capabilities": {
    "streaming": False,
    "pushNotifications": False,
    "stateTransitionHistory": False,
    "extensions": []
  },
  "defaultInputModes": [
    "application/json"
  ],
  "defaultOutputModes": [
    "application/json"
  ],
  "supportsAuthenticatedExtendedCard": False    
}

# ── Schemas ────────────────────────────────────────────────────────────────────
class AgentRequest(BaseModel):
    query: str

class AgentMeta(BaseModel):
    agentId: str
    version: str
    model: str
    processedAt: str

class SentimentResponse(BaseModel):
    sentiment: str
    score: float
    churn_risk: bool
    reason: str
    contextId: str | None
    agent: AgentMeta

# ── A2A Agent Card endpoint ────────────────────────────────────────────────────
@app.get("/analyze/.well-known/agent-card.json", tags=["A2A"])
async def agent_card():
    return JSONResponse(content=AGENT_CARD)

# ── JSON extractor ─────────────────────────────────────────────────────────────
def extract_json(raw_text: str) -> dict:

    logger.warning("JSON malformed — extracting fields individually.")
    result = {}

    m = re.search(r'"sentiment"\s*:\s*"([^"]+)"', raw_text)
    if m:
        result["sentiment"] = m.group(1)

    m = re.search(r'"score"\s*:\s*([0-9.]+)', raw_text)
    if m:
        result["score"] = float(m.group(1))

    m = re.search(r'"churn_risk"\s*:\s*(true|false)', raw_text, re.IGNORECASE)
    if m:
        result["churn_risk"] = m.group(1).lower() == "true"

    m = re.search(r'"reason"\s*:\s*"([^"]+)"', raw_text)
    if m:
        result["reason"] = m.group(1)

    if "sentiment" in result:
        result.setdefault("score", 0.5)
        result.setdefault("churn_risk", False)
        result.setdefault("reason", "Extracted from malformed model output.")
        return result

    raise ValueError(f"Could not extract fields from:\n{raw_text}")

# ── A2A error helper ───────────────────────────────────────────────────────────
def a2a_error(status_code: int, message: str, context_id: str | None):
    return HTTPException(
        status_code=status_code,
        detail={
            "error": message,
            "contextId": context_id,
            "agentId": AGENT_ID,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

# ── /analyze endpoint ──────────────────────────────────────────────────────────
@app.post("/analyze", response_model=SentimentResponse, tags=["Agent"])
async def analyze_sentiment(
    req: AgentRequest,
    x_agent_context_id: str = Header(None)
):
    if not GROQ_API_KEY:
        raise a2a_error(503, "GROQ_API_KEY environment variable not set.", x_agent_context_id)

    # Sanitize input
    clean_query = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', req.query)

    logger.info(f"[contextId={x_agent_context_id}] Received: {clean_query[:80]}...")

    # Call Groq API (OpenAI-compatible format)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an expert in customer sentiment analysis. "
                                "Analyze the customer email and respond ONLY with a valid JSON object. "
                                "No markdown, no explanation, no extra text — raw JSON only.\n"
                                "Required keys:\n"
                                '  "sentiment": one of "positive", "negative", "neutral"\n'
                                '  "score": float 0.0-1.0 (your confidence)\n'
                                '  "churn_risk": true or false\n'
                                '  "reason": one sentence explaining your decision'
                            )
                        },
                        {
                            "role": "user",
                            "content": clean_query
                        }
                    ],
                    "temperature": 0,
                    "max_tokens": 150
                }
            )
            response.raise_for_status()

    except httpx.ConnectError:
        raise a2a_error(503, "Cannot reach Groq API.", x_agent_context_id)
    except httpx.HTTPStatusError as e:
        raise a2a_error(500, f"Groq API error: {e.response.text}", x_agent_context_id)

    raw_output = response.json()["choices"][0]["message"]["content"]

    try:
        parsed = extract_json(raw_output)
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"JSON parse error: {e}")
        raise a2a_error(422, str(e), x_agent_context_id)

    result = SentimentResponse(
        sentiment=parsed.get("sentiment", "neutre"),
        score=float(parsed.get("score", 0.5)),
        churn_risk=bool(parsed.get("churn_risk", False)),
        reason=parsed.get("reason", ""),
        contextId=x_agent_context_id,
        agent=AgentMeta(
            agentId=AGENT_ID,
            version=AGENT_VERSION,
            model=MODEL_NAME,
            processedAt=datetime.now(timezone.utc).isoformat()
        )
    )

    logger.info(
        f"[contextId={x_agent_context_id}] "
        f"sentiment={result.sentiment}, score={result.score}, churn={result.churn_risk}"
    )
    return result

# ── /health endpoint ───────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "agentId": AGENT_ID,
        "version": AGENT_VERSION,
        "model": MODEL_NAME,
        "provider": "Groq",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)