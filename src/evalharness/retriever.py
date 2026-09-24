"""TF-IDF retriever over a folder of markdown documents (one fact per line)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    source: str
    text: str


class Retriever:
    def __init__(self, kb_dir: str | Path):
        self.chunks: list[Chunk] = []
        for path in sorted(Path(kb_dir).glob("*.md")):
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    self.chunks.append(Chunk(path.stem, line))
        if not self.chunks:
            raise ValueError(f"No documents found in {kb_dir}")
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(c.text for c in self.chunks)

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        ranked = scores.argsort()[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in ranked]
