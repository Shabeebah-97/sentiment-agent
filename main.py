from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional
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

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME   = "llama-3.1-8b-instant" 

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

# ── Updated Schemas to match your Payload ──────────────────────────────────────

class MessagePart(BaseModel):
# Use Optional for robustness, though text is usually required
    text: Optional[str] = ""
    kind: Optional[str] = "text"
    model_config = ConfigDict(extra="allow") # Ignore extra fields in parts

class MessageContent(BaseModel):
    role: Optional[str] = "user"
    parts: List[MessagePart] = []
    messageId: Optional[str] = Field(default_factory=lambda: "unknown")
    kind: Optional[str] = None
    model_config = ConfigDict(extra="allow")

class AgentRequest(BaseModel):
# A2A payloads sometimes wrap everything in 'message' or send it flat
    # We make 'message' optional or use an alias to catch variations
    message: Optional[MessageContent] = None
    model_config = ConfigDict(extra="allow")

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
    contextId: Optional[str] = None
    agent: AgentMeta

class JsonRpcResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: str
    result: SentimentResponse  # This nests your existing model inside 'result'
# ── JSON extractor ─────────────────────────────────────────────────────────────
def extract_json(raw_text: str) -> dict:
    # Try clean parse first
    try:
        clean_text = re.sub(r'```json\s?|\s?```', '', raw_text).strip()
        return json.loads(clean_text)
    except:
        pass

    # Fallback to Regex
    result = {}
    m = re.search(r'"sentiment"\s*:\s*"([^"]+)"', raw_text)
    if m: result["sentiment"] = m.group(1)
    
    m = re.search(r'"score"\s*:\s*([0-9.]+)', raw_text)
    if m: result["score"] = float(m.group(1))
    
    m = re.search(r'"churn_risk"\s*:\s*(true|false)', raw_text, re.IGNORECASE)
    if m: result["churn_risk"] = m.group(1).lower() == "true"
    
    m = re.search(r'"reason"\s*:\s*"([^"]+)"', raw_text)
    if m: result["reason"] = m.group(1)

    if "sentiment" in result:
        return result
    raise ValueError(f"Could not extract fields from LLM output")

# ── A2A error helper ───────────────────────────────────────────────────────────
def a2a_error(status_code: int, message: str, context_id: str | None):
    body = {
        "jsonrpc": "2.0",
        "id": context_id,
        "result": {
            "error": message,
            "contextId": context_id,
            "agentId": AGENT_ID,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    }
    return JSONResponse(status_code=status_code, content=body)

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/debug", tags=["Debug"])
async def debug_payload(request: AgentRequest):
    body = await request.body()
    logger.info(f"=== DEBUG HEADERS: {dict(request.headers)}")
    logger.info(f"=== DEBUG BODY: {body.decode('utf-8', errors='replace')}")
    return {
        "headers": dict(request.headers),
        "raw_body": body.decode("utf-8", errors="replace"),
        "content_type": request.headers.get("content-type")
    }

@app.get("/analyze/.well-known/agent-card.json", tags=["A2A"])
async def agent_card():
    # (Agent card logic remains the same)
    return JSONResponse(content=AGENT_CARD)

@app.post("/analyze", response_model=JsonRpcResponse, tags=["Agent"])

async def analyze_sentiment(
    req: AgentRequest,
    x_agent_context_id: str = Header(None)
):
    if not GROQ_API_KEY:
        raise a2a_error(503, "GROQ_API_KEY not set.", x_agent_context_id)

    # Extract text from the parts list (joining if multiple parts exist)
    raw_query = " ".join([p.text for p in req.message.parts if p.kind == "text"])
    
    # Sanitize input
    clean_query = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw_query).strip()

    logger.info(f"[contextId={x_agent_context_id}] Processing: {clean_query[:50]}...")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": "Analyze sentiment. Return ONLY JSON: {\"sentiment\": \"positive/negative/neutral\", \"score\": float, \"churn_risk\": bool, \"reason\": \"string\"}"},
                        {"role": "user", "content": clean_query}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0
                }
            )
            response.raise_for_status()

        raw_output = response.json()["choices"][0]["message"]["content"]
        parsed = extract_json(raw_output)

        # 1. Create the internal result object (your existing SentimentResponse logic)
        sentiment_data = {
            "sentiment": parsed.get("sentiment", "neutral"),
            "score": float(parsed.get("score", 0.5)),
            "churn_risk": bool(parsed.get("churn_risk", False)),
            "reason": parsed.get("reason", ""),
            "contextId": x_agent_context_id or req.message.messageId,
            "agent": {
                "agentId": AGENT_ID,
                "version": AGENT_VERSION,
                "model": MODEL_NAME,
                "processedAt": datetime.now(timezone.utc).isoformat()
            }
        }        

        # 2. Wrap it in the JSON-RPC format required by the A2A spec
        return {
            "jsonrpc": "2.0",
            "id": x_agent_context_id or req.message.messageId, # Use the incoming messageId as the RPC ID
            "result": sentiment_data
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise a2a_error(500, str(e), x_agent_context_id)

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)