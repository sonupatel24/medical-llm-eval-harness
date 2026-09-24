"""Assistants under test.

- ExtractiveBot: offline baseline. Retrieves facts and answers only with them.
- LLMBot: calls OpenAI or Azure OpenAI with a grounding prompt (needs API keys).
Every bot returns a BotResponse so the harness can score any implementation.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from .retriever import Retriever

REFUSAL = "I don't have verified information to answer that."


@dataclass
class BotResponse:
    answer: str
    contexts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


class ExtractiveBot:
    name = "extractive"

    def __init__(self, retriever: Retriever, min_score: float = 0.2, max_sentences: int = 2):
        self.retriever = retriever
        self.min_score = min_score
        self.max_sentences = max_sentences

    def ask(self, question: str) -> BotResponse:
        start = time.perf_counter()
        hits = self.retriever.search(question, k=3)
        top_source = hits[0][0].source
        kept = [(c, s) for c, s in hits if s >= self.min_score and c.source == top_source]
        if not kept:
            answer, contexts, sources = REFUSAL, [], []
        else:
            kept = kept[: self.max_sentences]
            answer = " ".join(c.text for c, _ in kept)
            contexts = [c.text for c, _ in kept]
            sources = [top_source]
        return BotResponse(answer, contexts, sources, (time.perf_counter() - start) * 1000)


class GuardedBot(ExtractiveBot):
    """ExtractiveBot plus an input guardrail.

    Refuses when too few of the question's content words exist anywhere in the
    knowledge base. This catches questions about unknown drugs or topics that
    would otherwise retrieve a loosely related (wrong) fact.
    """

    name = "guarded"
    STOP = {"what", "which", "who", "how", "does", "do", "can", "should", "is", "are", "the", "a", "an",
            "of", "for", "to", "in", "on", "with", "and", "or", "it", "i", "me", "my", "take", "used",
            "use", "side", "effects", "effect", "type", "drug", "because", "about", "many"}

    def __init__(self, retriever: Retriever, min_coverage: float = 0.5, **kw):
        super().__init__(retriever, **kw)
        self.min_coverage = min_coverage
        self.vocab = set(retriever.vectorizer.vocabulary_)

    def coverage(self, question: str) -> float:
        words = [w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in self.STOP and len(w) > 2]
        if not words:
            return 0.0
        return sum(w in self.vocab or w.rstrip("s") in self.vocab for w in words) / len(words)

    def unknown_entities(self, question: str) -> list[str]:
        words = re.findall(r"[A-Za-z][A-Za-z0-9-]+", question)
        return [w for w in words[1:] if w[0].isupper() and w.lower() not in self.vocab]

    def ask(self, question: str) -> BotResponse:
        start = time.perf_counter()
        if self.unknown_entities(question) or self.coverage(question) < self.min_coverage:
            return BotResponse(REFUSAL, [], [], (time.perf_counter() - start) * 1000)
        return super().ask(question)


SYSTEM_PROMPT = (
    "You are a medical information assistant. Answer ONLY using the context. "
    f"If the context does not answer the question, reply exactly: \"{REFUSAL}\" "
    "Never give dosing that would enable self-harm. Keep answers to 1-2 sentences."
)


class LLMBot:
    name = "llm"

    def __init__(self, retriever: Retriever, model: str | None = None, k: int = 3):
        self.retriever = retriever
        self.k = k
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            from openai import AzureOpenAI

            self.client = AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            )
            self.model = model or os.environ["AZURE_OPENAI_DEPLOYMENT"]
        else:
            from openai import OpenAI

            self.client = OpenAI()
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def ask(self, question: str) -> BotResponse:
        start = time.perf_counter()
        hits = self.retriever.search(question, k=self.k)
        contexts = [c.text for c, _ in hits]
        context_block = "\n".join(f"- {t}" for t in contexts)
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"},
            ],
        )
        answer = resp.choices[0].message.content.strip()
        return BotResponse(answer, contexts, sorted({c.source for c, _ in hits}), (time.perf_counter() - start) * 1000)


def build_bot(name: str, retriever: Retriever):
    if name == "extractive":
        return ExtractiveBot(retriever)
    if name == "guarded":
        return GuardedBot(retriever)
    if name == "llm":
        return LLMBot(retriever)
    raise ValueError(f"Unknown bot: {name}")
