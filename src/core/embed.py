import ollama

from src.config import EMBED_MODEL, OLLAMA_BASE_URL


def embed(text: str) -> list[float]:
    """Embed text using the configured Ollama embedding model."""
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return resp["embedding"]
