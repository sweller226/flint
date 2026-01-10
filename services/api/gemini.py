import os
import google.generativeai as genai
from pydantic import BaseModel

# Schema for the Strategy Explanation response
EXPLANATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "narrative": {"type": "string", "description": "Plain English explanation of the market situation."},
        "concepts": {
            "type": "array", 
            "items": {"type": "string"},
            "description": "List of ICT concepts identified (e.g. 'Bearish FVG', 'BSL Sweep')."
        },
        "bias": {
            "type": "string", 
            "enum": ["long", "short", "neutral"],
            "description": "Recommended trading bias."
        },
        "suggestions": {
            "type": "object",
            "properties": {
                "timeframe": {"type": "string"},
                "risk_per_trade": {"type": "string"}
            }
        }
    },
    "required": ["narrative", "concepts", "bias"]
}

PROMPT_EXPLAIN_STRATEGY = """
You are an expert quantitative trader specialized in ICT (Inner Circle Trader) concepts. 
Analyze the provided market data snapshot and user question.

Market Context:
- Current Price: {price}
- Trend: {trend}
- Nearby Liquidty: {liquidity_levels}
- Recent Signals: {recent_signals}

User Question: "{user_question}"

Output a JSON response explaining the setup, identifying key ICT concepts (FVG, Order Blocks, Liquidity Sweeps), and suggesting a directional bias.
Follow this JSON schema:
{schema}
"""

class GeminiService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-pro') # Using 1.5 Pro for better reasoning
        else:
            self.model = None
            print("WARNING: GEMINI_API_KEY not found.")

    async def explain_strategy(self, market_snapshot: dict, user_question: str):
        if not self.model:
            return {"error": "Gemini API not configured"}

        prompt = PROMPT_EXPLAIN_STRATEGY.format(
            price=market_snapshot.get("price", "Unknown"),
            trend=market_snapshot.get("trend", "Neutral"),
            liquidity_levels=market_snapshot.get("liquidity", "None"),
            recent_signals=market_snapshot.get("signals", []),
            user_question=user_question,
            schema=EXPLANATION_JSON_SCHEMA
        )

        try:
            response = self.model.generate_content(prompt)
            # In a real app, using generation_config={'response_mime_type': 'application/json'} is better
            # For now we assume the model follows instructions or we parse the text
            return response.text 
        except Exception as e:
            return {"error": str(e)}

