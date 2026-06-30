"""
Thin LLM dispatcher — routes calls to either the local Ollama model or Google Gemini.

Set LLM_PROVIDER=local  to use Ollama (default)
Set LLM_PROVIDER=gemini to use Google Gemini API

All callers use:
    from src.llm import call
    text = call(prompt, temperature=0.1)
"""

import logging
import time

from src.config import GEMINI_API_KEY, GEMINI_MODEL, LIGHTWEIGHT_LLM, LLM_PROVIDER

logger = logging.getLogger(__name__)

# Lazy-initialised Gemini client — created on first use
_gemini_client = None

# Seconds to wait between Gemini requests; override with GEMINI_REQUEST_PAUSE env var
import os as _os
_GEMINI_PAUSE: float = float(_os.getenv("GEMINI_REQUEST_PAUSE", "0.0"))
_GEMINI_MAX_RETRIES: int = 5


def call(prompt: str, *, temperature: float = 0.1) -> str:
    """Call the active LLM provider and return the response text."""
    if LLM_PROVIDER == "gemini":
        return _call_gemini(prompt, temperature)
    return _call_ollama(prompt, temperature)


def _call_ollama(prompt: str, temperature: float) -> str:
    import ollama
    resp = ollama.chat(
        model=LIGHTWEIGHT_LLM,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return resp.message.content


def _call_gemini(prompt: str, temperature: float) -> str:
    global _gemini_client
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set in .env")

    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("llm | Gemini client initialised | model=%s", GEMINI_MODEL)

    from google import genai

    delay = _GEMINI_PAUSE
    for attempt in range(1, _GEMINI_MAX_RETRIES + 1):
        try:
            time.sleep(_GEMINI_PAUSE)
            response = _gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=genai.types.GenerateContentConfig(temperature=temperature),
            )
            return response.text
        except Exception as exc:
            exc_str = str(exc).lower()
            is_quota = "429" in exc_str or "503" in exc_str or "quota" in exc_str or "resource_exhausted" in exc_str or "rate" in exc_str or "unavailable" in exc_str
            if is_quota and attempt < _GEMINI_MAX_RETRIES:
                logger.warning(
                    "llm | Gemini rate limit hit (attempt %d/%d) — backing off %.0fs",
                    attempt, _GEMINI_MAX_RETRIES, delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 60)
            else:
                raise
