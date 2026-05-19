"""
Mini-RAG Assistant — core pipeline.

Pipeline stages:
  1. INGEST   — read PDFs/TXT files, split into chunks
  2. EMBED    — turn each chunk into a vector (sentence-transformers, runs locally)
  3. INDEX    — store vectors in a FAISS index for fast similarity search
  4. RETRIEVE — embed the user question, find top-k similar chunks
  5. GENERATE — send retrieved chunks + question to an LLM (Gemini), get a grounded answer
  6. SCORE    — compute a confidence score from retrieval similarities

Everything that touches an external service is isolated in `generate_answer`.
Everything else runs locally with no network calls.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# We use Groq for generation (Llama models). Free tier, no credit card required.
# Get a key at https://console.groq.com
from groq import Groq


# ---------- Data types ----------

@dataclass
class Chunk:
    """A single piece of text from the knowledge base, plus where it came from."""
    text: str
    source: str        # filename
    chunk_id: int      # position within the corpus (used for citations like [3])

@dataclass
class RetrievedChunk:
    chunk: Chunk
    similarity: float  # 0..1, higher is more similar to the query

@dataclass
class RAGResponse:
    answer: str
    retrieved: List[RetrievedChunk]
    confidence: float           # 0..1, overall confidence in the answer
    confidence_label: str       # "High" / "Medium" / "Low"


# ---------- 1. INGEST ----------

def read_file(path: Path) -> str:
    """Read a PDF or text file and return its plain text."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        # extract_text() returns None for image-only pages; guard against that
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    # Default: treat as UTF-8 text (covers .txt, .md)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks of roughly `chunk_size` words.

    Why overlap? If an answer spans the boundary between two chunks, overlap
    ensures at least one chunk contains the full context.
    """
    # Normalize whitespace so chunk boundaries are predictable
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split(" ")
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def build_corpus(file_paths: List[Path]) -> List[Chunk]:
    """Read every file, chunk it, and return a flat list of Chunks with global IDs."""
    corpus: List[Chunk] = []
    chunk_id = 0
    for path in file_paths:
        raw_text = read_file(path)
        for piece in chunk_text(raw_text):
            corpus.append(Chunk(text=piece, source=path.name, chunk_id=chunk_id))
            chunk_id += 1
    return corpus


# ---------- 2 & 3. EMBED + INDEX ----------

class VectorStore:
    """
    Wraps an embedding model and a FAISS index together.

    We use `all-MiniLM-L6-v2`: small (~80MB), fast, good general-purpose embeddings.
    Vectors are L2-normalized so we can use inner product as cosine similarity.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        if not chunks:
            return
        texts = [c.text for c in chunks]
        # normalize_embeddings=True -> each vector has length 1
        # -> inner product equals cosine similarity, which is bounded in [-1, 1]
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # IP = inner product
        self.index.add(embeddings)

    def search(self, query: str, k: int = 4) -> List[RetrievedChunk]:
        """Return the top-k chunks most similar to the query."""
        if self.index is None or not self.chunks:
            return []
        query_vec = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        # FAISS returns parallel arrays of scores and indices
        scores, indices = self.index.search(query_vec, min(k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS sentinel for "no result"
                continue
            # Cosine sim is in [-1, 1]; clip negatives to 0 for a clean 0..1 confidence
            similarity = float(max(0.0, score))
            results.append(RetrievedChunk(chunk=self.chunks[idx], similarity=similarity))
        return results


# ---------- 5. GENERATE ----------

PROMPT_TEMPLATE = """You are a careful research assistant. Answer the user's question \
using ONLY the numbered context passages below. Cite the passage numbers you used \
inline like [1] or [2, 3]. If the context does not contain the answer, say exactly: \
"I don't have enough information in the provided documents to answer that."

Context:
{context}

Question: {question}

Answer:"""


def format_context(retrieved: List[RetrievedChunk]) -> str:
    """Number the retrieved chunks so the model can cite them."""
    lines = []
    for i, r in enumerate(retrieved, start=1):
        lines.append(f"[{i}] (source: {r.chunk.source})\n{r.chunk.text}")
    return "\n\n".join(lines)


def generate_answer(
    question: str,
    retrieved: List[RetrievedChunk],
    api_key: str,
    model_name: str = "llama-3.3-70b-versatile",
) -> str:
    """Call Groq (Llama) with the retrieved context and return the answer text."""
    client = Groq(api_key=api_key)
    prompt = PROMPT_TEMPLATE.format(
        context=format_context(retrieved),
        question=question,
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


# ---------- 6. CONFIDENCE ----------

def compute_confidence(
    retrieved: List[RetrievedChunk], answer: str
) -> Tuple[float, str]:
    """
    Confidence is the mean similarity of CITED chunks (or top chunks if none cited).

    Rationale: if the model relied on chunks the retriever rated highly, we trust the
    answer more. If the model refused to answer ("I don't have enough information"),
    confidence drops to 0.
    """
    if not retrieved:
        return 0.0, "Low"

    # If the model bailed, we shouldn't claim high confidence
    if "don't have enough information" in answer.lower():
        return 0.0, "Low"

    # Find citation markers like [1], [2, 3], [1][4] in the answer
    cited_indices = set()
    for match in re.findall(r"\[([\d,\s]+)\]", answer):
        for num_str in match.split(","):
            num_str = num_str.strip()
            if num_str.isdigit():
                # citations are 1-indexed in the prompt; convert to 0-indexed list position
                cited_indices.add(int(num_str) - 1)

    if cited_indices:
        scores = [
            retrieved[i].similarity
            for i in cited_indices
            if 0 <= i < len(retrieved)
        ]
    else:
        # Model didn't cite — fall back to top-3 mean similarity
        scores = [r.similarity for r in retrieved[:3]]

    confidence = float(np.mean(scores)) if scores else 0.0

    if confidence >= 0.6:
        label = "High"
    elif confidence >= 0.4:
        label = "Medium"
    else:
        label = "Low"
    return confidence, label


# ---------- Top-level convenience ----------

def answer_question(
    question: str,
    store: VectorStore,
    api_key: str,
    k: int = 4,
) -> RAGResponse:
    """Run the full retrieve → generate → score pipeline for one question."""
    retrieved = store.search(question, k=k)
    answer = generate_answer(question, retrieved, api_key=api_key)
    confidence, label = compute_confidence(retrieved, answer)
    return RAGResponse(
        answer=answer,
        retrieved=retrieved,
        confidence=confidence,
        confidence_label=label,
    )


# ---------- CLI entry point for quick testing ----------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mini-RAG CLI")
    parser.add_argument("folder", help="Folder containing .pdf or .txt files")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Set GROQ_API_KEY environment variable first.")

    files = [p for p in Path(args.folder).iterdir()
             if p.suffix.lower() in {".pdf", ".txt", ".md"}]
    print(f"Loading {len(files)} files...")
    corpus = build_corpus(files)
    print(f"Built {len(corpus)} chunks. Embedding...")
    store = VectorStore()
    store.build(corpus)

    print(f"\nQ: {args.question}\n")
    result = answer_question(args.question, store, api_key=api_key, k=args.k)
    print(f"A: {result.answer}\n")
    print(f"Confidence: {result.confidence:.2f} ({result.confidence_label})\n")
    print("Retrieved sources:")
    for i, r in enumerate(result.retrieved, 1):
        print(f"  [{i}] {r.chunk.source}  sim={r.similarity:.3f}")
