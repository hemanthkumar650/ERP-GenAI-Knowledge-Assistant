from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from dotenv import load_dotenv
from openai import AzureOpenAI


load_dotenv()


@dataclass(frozen=True)
class Settings:
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    azure_openai_chat_deployment: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
    # Must match the deployment name in Azure Portal (Deployments), not the model ID.
    azure_openai_embedding_deployment: str = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
    )
    policies_path: str = os.getenv("POLICIES_PATH", "data/policies")
    chroma_path: str = os.getenv("CHROMA_PATH", "data/chroma_db")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "erp_policies")
    top_k: int = int(os.getenv("TOP_K", "2"))
    max_answer_tokens: int = int(os.getenv("MAX_ANSWER_TOKENS", "200"))
    context_max_chars: int = int(os.getenv("CONTEXT_MAX_CHARS", "2400"))

    @property
    def azure_ready(self) -> bool:
        return all(
            [
                self.azure_openai_api_key,
                self.azure_openai_endpoint,
                self.azure_openai_chat_deployment,
                self.azure_openai_embedding_deployment,
            ]
        )


settings = Settings()


@lru_cache(maxsize=1)
def get_azure_client() -> AzureOpenAI:
    if not settings.azure_ready:
        raise RuntimeError(
            "Azure OpenAI credentials are missing. Set AZURE_OPENAI_API_KEY, "
            "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_CHAT_DEPLOYMENT, and "
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT."
        )
    return AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
    )


def get_embedding(text: str) -> list[float]:
    client = get_azure_client()
    response = client.embeddings.create(
        model=settings.azure_openai_embedding_deployment,
        input=text,
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: Iterable[str]) -> list[list[float]]:
    batch = [t for t in texts if t and t.strip()]
    if not batch:
        return []
    client = get_azure_client()
    response = client.embeddings.create(
        model=settings.azure_openai_embedding_deployment,
        input=batch,
    )
    return [item.embedding for item in response.data]


def generate_chat(prompt: str, max_tokens: int | None = None) -> str:
    max_tokens = max_tokens if max_tokens is not None else settings.max_answer_tokens
    client = get_azure_client()
    response = client.chat.completions.create(
        model=settings.azure_openai_chat_deployment,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def generate_chat_stream(prompt: str, max_tokens: int | None = None):
    """Yield content deltas from Azure OpenAI chat completion (streaming)."""
    max_tokens = max_tokens if max_tokens is not None else settings.max_answer_tokens
    client = get_azure_client()
    stream = client.chat.completions.create(
        model=settings.azure_openai_chat_deployment,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
