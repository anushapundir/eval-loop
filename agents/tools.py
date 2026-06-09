"""Retrieval tool over the local knowledge base (CLAUDE.md Day 2).

A deliberately simple, embedding-free retriever: it loads the markdown KB,
splits each doc into section chunks, and ranks chunks by token overlap with the
query. It is pure and deterministic (no randomness, stable tie-breaking) so
retrieval is fully reproducible — a requirement for trustworthy evaluation.

This is the agent's only tool. Keeping the scoring logic here as pure functions
(separate from the graph) makes it testable without LangGraph or a live model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Tiny stopword set: function words that add no retrieval signal. Kept small on
# purpose — over-aggressive filtering hurts more than it helps on a small KB.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "how", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "this", "to", "what", "when", "why", "with", "you", "your",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class KbChunk:
    """One retrievable passage: a section of a KB document."""

    doc_id: str  # filename stem of the source document
    title: str  # section heading the chunk came from
    text: str  # the chunk's text (heading line + body)


@dataclass(frozen=True)
class RetrievalResult:
    """The outcome of a retrieval: ranked chunks plus convenience views."""

    chunks: list[KbChunk]
    doc_ids: list[str]  # unique source doc ids, most-relevant first
    context: str  # chunk texts concatenated for prompt injection


def tokenize(text: str) -> set[str]:
    """Lowercase, split into word tokens, and drop stopwords.

    Public so the deterministic evaluators (grounding/coverage) score against
    the exact same tokenization the retriever uses — one tokenizer, no drift.
    """
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def load_kb(kb_dir: Path) -> list[KbChunk]:
    """Load all markdown docs under ``kb_dir`` as section-level chunks.

    Each ``## Heading`` starts a new chunk; any text before the first ``##``
    (the doc intro under its ``# Title``) becomes an intro chunk so nothing is
    lost. Returns chunks in document then section order.
    """
    chunks: list[KbChunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        doc_id = path.stem
        chunks.extend(_split_doc(doc_id, path.read_text(encoding="utf-8")))
    return chunks


def _split_doc(doc_id: str, content: str) -> list[KbChunk]:
    """Split one document's text into section chunks."""
    doc_title = doc_id
    sections: list[tuple[str, list[str]]] = []
    current_title = "Introduction"
    current_lines: list[str] = []

    for line in content.splitlines():
        if line.startswith("# ") and not line.startswith("## "):
            doc_title = line[2:].strip()
            current_title = doc_title
            continue
        if line.startswith("## "):
            if any(s.strip() for s in current_lines):
                sections.append((current_title, current_lines))
            current_title = line[3:].strip()
            current_lines = []
            continue
        current_lines.append(line)

    if any(s.strip() for s in current_lines):
        sections.append((current_title, current_lines))

    chunks: list[KbChunk] = []
    for title, lines in sections:
        body = "\n".join(lines).strip()
        text = f"{title}\n{body}" if body else title
        chunks.append(KbChunk(doc_id=doc_id, title=title, text=text))
    return chunks


def retrieve(query: str, chunks: list[KbChunk], k: int = 3) -> RetrievalResult:
    """Return the ``k`` chunks most relevant to ``query`` by token overlap.

    Scoring is the count of shared tokens between query and chunk. Chunks with
    zero overlap are excluded, so an off-topic query yields an empty result and
    the agent can honestly say it lacks the context. Ties break by ``doc_id``
    then ``title`` for deterministic ordering.
    """
    query_tokens = tokenize(query)

    scored: list[tuple[int, KbChunk]] = []
    for chunk in chunks:
        overlap = len(query_tokens & tokenize(chunk.text))
        if overlap > 0:
            scored.append((overlap, chunk))

    scored.sort(key=lambda pair: (-pair[0], pair[1].doc_id, pair[1].title))
    top = [chunk for _, chunk in scored[:k]]

    doc_ids: list[str] = []
    for chunk in top:
        if chunk.doc_id not in doc_ids:
            doc_ids.append(chunk.doc_id)

    context = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in top)
    return RetrievalResult(chunks=top, doc_ids=doc_ids, context=context)
