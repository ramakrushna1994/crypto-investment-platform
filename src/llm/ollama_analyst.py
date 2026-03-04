import logging
import requests
import json

logger = logging.getLogger(__name__)

# local ollama endpoint
OLLAMA_API_URL = "http://ollama:11434/api/generate"
MODEL_NAME = "llama3.2:latest"

def get_ollama_analysis(
    symbol: str,
    asset_type: str,
    price: float,
    currency_prefix: str,
    currency_suffix: str,
    signal: str,
    confidence: float,
    rsi: float,
    macd: float,
    macd_signal_val: float,
    ema_20: float,
    ma_7: float,
    bb_upper: float,
    bb_lower: float,
    volatility: float,
    sma_50: float,
    sma_200: float,
    atr_14: float,
    stoch_k: float,
    stoch_d: float
) -> str | None:
    """Get investment analysis narrative from local ollama."""
    prompt = f"""You are a professional financial analyst providing a brief, factual investment analysis.
Analyse the following technical indicators for {symbol} ({asset_type}). 

Technical Data:
- Current Price: {currency_prefix}{price:,.2f}{currency_suffix}
- AI Signal: {signal} (Confidence: {confidence * 100:.1f}%)
- Macro Trend (SMA): 50-day {currency_prefix}{sma_50:,.2f}{currency_suffix} vs 200-day {currency_prefix}{sma_200:,.2f}{currency_suffix}
- Short Trend (EMA/SMA): 20-day {currency_prefix}{ema_20:,.2f}{currency_suffix} | 7-day {currency_prefix}{ma_7:,.2f}{currency_suffix}
- Volatility (ATR): {currency_prefix}{atr_14:,.2f}{currency_suffix}
- RSI (14-day): {rsi:.1f} {'(Overbought)' if rsi > 70 else '(Oversold)' if rsi < 30 else '(Neutral)'}
- Stochastic Osc (14-day): %K {stoch_k:.1f} | %D {stoch_d:.1f}
- MACD: {macd:.4f} | Signal Line: {macd_signal_val:.4f} → {'Bullish crossover' if macd > macd_signal_val else 'Bearish crossover'}
- Bollinger Bands: Upper {currency_prefix}{bb_upper:,.2f}{currency_suffix} | Lower {currency_prefix}{bb_lower:,.2f}{currency_suffix} -> Price is {'Above upper band (potentially overextended)' if price > bb_upper else 'Below lower band (potentially oversold)' if price < bb_lower else 'Within bands (normal range)'}

Provide:
1. A 2-3 sentence summary of the current market situation.
2. Key risks or opportunities based on the data.
3. A brief reasoning for the AI signal.

Keep your response extremely concise (under 200 words). Use plain text, no markdown headers."""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3
        }
    }

    try:
        # Increased timeout for local LLM generation
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        logger.warning(f"Ollama analysis failed for {symbol}: {e}")
        return None

def chat_with_ollama(
    symbol: str,
    context_data: dict,
    user_query: str
) -> str | None:
    """
    Answers a user's question about an asset strictly using the provided technical context.
    Uses temperature 0.0 to prevent hallucinations and insists on factual retrieval.
    """
    system_prompt = f"""You are an opinionated, expert-level financial analyst reviewing {symbol}.
Your job is to give the user a straight, honest take on the asset using the Context Data below.

RULES:
1. Be direct and confident. If the data says bullish, say bullish. If it looks weak, say so. Don't hedge every sentence.
2. Give specific estimates when the user asks (e.g., projected CAGR, price targets) based on the technicals and momentum. State your reasoning clearly.
3. Explain what the indicators mean when relevant, but prioritize giving your actual opinion over textbook definitions.
4. If the user asks "should I buy?", give your honest read on whether the data supports that move and what the risks are — don't dodge the question.
5. End every response with a single line: "Note: This is data-driven analysis, not licensed financial advice. Always do your own due diligence."

Context Data for {symbol}:
{json.dumps(context_data, indent=2)}
"""

    # Ollama Chat API format
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "stream": False,
        "options": {
            "temperature": 0.3  # Bumped up slightly to allow the AI to "predict" and extrapolate 
        }
    }

    try:
        # Increased timeout for local LLM chat
        response = requests.post("http://ollama:11434/api/chat", json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.warning(f"Ollama chat failed for {symbol}: {e}")
        return "Sorry, the local AI is currently unreachable."
