"""Hybrid retrieval: BM25 keyword search + optional dense embeddings, fused with RRF.

BM25 is exact on names, codes and numbers; embeddings catch paraphrases.
Reciprocal Rank Fusion combines the two rankings without having to calibrate
their very different score scales.
"""
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

STOPWORDS = set("""a an and are as at be by can do does for from how i in is it my of on or
the this to was what when where which who why will with you your""".split())


def _stem(token: str) -> str:
    """Tiny plural stemmer so 'passwords' matches 'password'."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


@dataclass
class Chunk:
    id: str
    source: str
    section: str
    text: str


@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict = field(default_factory=dict)


def chunk_markdown(source: str, text: str, max_chars: int = 700) -> list[Chunk]:
    """Split on headings, then on paragraphs, keeping each chunk under max_chars."""
    chunks, section, buf = [], "", []

    def flush():
        if buf:
            chunks.append(Chunk(f"{source}#{len(chunks) + 1}", source, section, "\n".join(buf).strip()))
            buf.clear()

    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        heading = re.match(r"^#{1,6}\s+(.*)", para)
        if heading:
            flush()
            section = heading.group(1).splitlines()[0]
            para = para[heading.end():].strip()
            if not para:
                continue
        if buf and len("\n".join(buf)) + len(para) > max_chars:
            flush()
        buf.append(para)
    flush()
    return chunks


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avgdl = sum(map(len, docs)) / max(len(docs), 1)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, doc in zip(self.tf, self.docs):
            s = 0.0
            for t in query:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
            out.append(s)
        return out


def rrf(rankings: dict[str, list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankers of 1 / (k + rank)."""
    fused: dict[str, float] = {}
    for ranking in rankings.values():
        for rank, doc_id in enumerate(ranking, 1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder=None, cache_path: Path | None = None):
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}
        self.bm25 = BM25([tokenize(f"{c.section} {c.text}") for c in chunks])
        self.embedder = embedder
        self.vectors: dict[str, list[float]] = {}
        if embedder:
            self._embed_corpus(cache_path)

    @classmethod
    def from_folder(cls, folder: Path, **kwargs) -> "Retriever":
        chunks = []
        for path in sorted(Path(folder).glob("*.md")):
            chunks += chunk_markdown(path.name, path.read_text(encoding="utf-8"))
        return cls(chunks, **kwargs)

    def _embed_corpus(self, cache_path: Path | None):
        cache = {}
        if cache_path and cache_path.exists():
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        try:
            for c in self.chunks:
                key = f"{self.embedder.model}:{hashlib.sha256(c.text.encode()).hexdigest()[:16]}"
                if key not in cache:
                    cache[key] = self.embedder.embed(c.text)
                self.vectors[c.id] = cache[key]
        except Exception:
            self.vectors = {}  # embeddings unavailable: keyword-only retrieval
            return
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache), encoding="utf-8")

    def search(self, query: str, k: int = 3) -> list[Hit]:
        rankings = {}
        bm = self.bm25.scores(tokenize(query))
        rankings["bm25"] = [self.chunks[i].id for i in sorted(range(len(bm)), key=lambda i: -bm[i]) if bm[i] > 0]
        if self.vectors:
            try:
                q = self.embedder.embed(query)
                sims = {cid: _cosine(q, v) for cid, v in self.vectors.items()}
                rankings["dense"] = sorted(sims, key=lambda cid: -sims[cid])[:20]
            except Exception:
                pass
        fused = rrf(rankings)
        top = sorted(fused, key=lambda cid: -fused[cid])[:k]
        return [Hit(self.by_id[cid], fused[cid],
                    {name: r.index(cid) + 1 for name, r in rankings.items() if cid in r}) for cid in top]
