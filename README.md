# Mini-RAG Assistant

A lightweight Retrieval-Augmented Generation prototype that answers questions
grounded in a local document corpus, with inline citations and confidence scores.
Built with Streamlit + FAISS + sentence-transformers + Groq (Llama).

## Architecture

```
┌────────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐
│  PDF/TXT   │──▶│ Chunker  │──▶│ Embedder │──▶│  FAISS   │   │  User   │
│  uploads   │   │ (500 wd, │   │ (MiniLM) │   │  Index   │   │ question│
└────────────┘   │  50 ovr) │   └──────────┘   └────┬─────┘   └────┬────┘
                 └──────────┘                       │              │
                                                    │   ┌──────────▼─────┐
                                                    └──▶│  Top-k search  │
                                                        └──────┬─────────┘
                                                               │
                                                  ┌────────────▼────────┐
                                                  │ Llama 3.3 70B       │
                                                  │ (via Groq, grounded │
                                                  │   prompt)           │
                                                  └────────────┬────────┘
                                                               │
                                                  ┌────────────▼────────┐
                                                  │ Answer + citations  │
                                                  │ + confidence score  │
                                                  └─────────────────────┘
```

**Pipeline:**

1. **Ingest** - PDFs parsed with `pypdf`; text normalized and split into ~500-word chunks with 50-word overlap.
2. **Embed** - chunks encoded with `sentence-transformers/all-MiniLM-L6-v2` (384-dim, runs locally, no API cost).
3. **Index** - vectors L2-normalized and added to a FAISS `IndexFlatIP`, so inner product equals cosine similarity.
4. **Retrieve** - user question embedded the same way; top-k nearest chunks returned with similarity scores.
5. **Generate** - retrieved chunks numbered and injected into a strict prompt that instructs Llama to cite passages inline (`[1]`, `[2]`) or refuse if context is insufficient.
6. **Score** - confidence = mean cosine similarity of the chunks the model actually cited. If the model refuses ("I don't have enough information"), confidence is forced to 0.

## Confidence Scoring

| Score      | Label  | Meaning                                             |
|------------|--------|-----------------------------------------------------|
| ≥ 0.60     | High   | Retrieved chunks closely match the question         |
| 0.40–0.60  | Medium | Partial match; answer may be incomplete             |
| < 0.40     | Low    | Weak match; treat answer with skepticism            |

Citations are parsed from the answer text via regex (`\[(\d+(?:,\s*\d+)*)\]`). If
the model returns no citations, we fall back to the mean similarity of the top 3
retrieved chunks.

## Retrieval Modes (the `k` slider)

How many chunks to feed the model per question:

| k     | Mode     | Best for                                        | Trade-off                                          |
|-------|----------|-------------------------------------------------|----------------------------------------------------|
| 1–2   | Focused  | Specific factual lookups                        | May miss context if answer spans multiple sections |
| 3–5   | Balanced *(recommended)* | Most everyday questions          | Sweet spot of quality vs. cost                     |
| 6–8   | Broad    | Summaries, multi-doc comparisons                | Slower, more API tokens, can dilute the answer    |

## Setup

```bash
git clone <repo>
cd mini_rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Get a free Groq API key at <https://console.groq.com> (no credit card required).

## Run

**Streamlit UI:**
```bash
streamlit run app.py
```
Paste your API key in the sidebar, upload PDFs/text files, click **Build Index**, ask questions.

**CLI (for quick testing):**
```bash
export GROQ_API_KEY=your_key_here       # macOS / Linux
$env:GROQ_API_KEY="your_key_here"       # Windows PowerShell

python rag_pipeline.py ./docs "What is the refund policy?"
```

## UI Features

The Streamlit app includes built-in guidance for first-time users:

- **📖 New here? Start with this** - plain-language explanation of RAG and what a "knowledge base" is
- **✨ Key features** - at-a-glance list of what the assistant does (grounded answers, citations, confidence, refusal handling)
- **🔍 How does it work under the hood?** - the 6-stage pipeline above, in user-friendly form
- **💡 Tips for better questions** - guidance on phrasing, paraphrasing, and interpreting confidence
- **Sample question buttons** - one-click examples that auto-fill the input
- **Live `k` mode guide** - comparison table of Focused / Balanced / Broad modes next to the slider
- **Color-coded confidence badge** with a tooltip explaining what each level means
- **Expandable "Retrieved sources"** under each answer so you can audit exactly what the model saw

## Choosing a Llama model

Default is `llama-3.3-70b-versatile`. Change `model_name` in `rag_pipeline.py:generate_answer()`:

| Model ID                          | Size | Use when                                       |
|-----------------------------------|------|------------------------------------------------|
| `llama-3.3-70b-versatile`         | 70B  | **Default** - best quality, still very fast    |
| `llama-3.1-8b-instant`            | 8B   | Faster + cheaper, good for high-volume testing |
| `llama-3.2-90b-vision-preview`    | 90B  | Only if you add image input                    |
| `llama-3.2-1b-preview`            | 1B   | Tiny - latency-sensitive demos                 |

Full list: <https://console.groq.com/docs/models>

## Example

**Question:** *What is the cancellation window for premium subscriptions?*

**Answer:** Premium subscriptions may be cancelled within 14 days of purchase for a full refund [2]. After 14 days, partial refunds are prorated based on usage [2, 3].

**Confidence:** High (0.74)

**Retrieved sources:**
- `[1] terms.pdf` - sim 0.81
- `[2] refund_policy.pdf` - sim 0.77
- `[3] terms.pdf` - sim 0.62
- `[4] faq.txt` - sim 0.48

## Project Structure

```
mini_rag/
├── rag_pipeline.py    # Core RAG logic (ingest, embed, retrieve, generate, score)
├── app.py             # Streamlit UI
├── requirements.txt
└── README.md
```

## Design Choices

- **FAISS over Chroma** - zero setup, in-memory, perfect for a prototype.
- **MiniLM over OpenAI embeddings** - free, fast, no API costs, ~80 MB download.
- **Groq + Llama over Claude / GPT-4** - generous free tier, extremely fast inference, no credit card required.
- **Citations parsed from answer text** - simple and transparent; no special LLM output format required.
- **Refusal handling** - explicit "I don't know" instruction in the prompt is what reduces hallucination most.
- **User brings their own API key** - no shared secrets to manage when deployed; each session's key lives only in the browser.

## Deployment

Both options below have free tiers and work out of the box with this app.

**Hugging Face Spaces** (recommended - 16 GB RAM, free):
1. Create a new Space at <https://huggingface.co/new-space> with SDK = **Streamlit**, Hardware = **CPU basic (free)**.
2. Push your repo files to the Space.
3. App auto-builds from `requirements.txt`.

**Streamlit Community Cloud**:
1. Push your project to a public GitHub repo.
2. Go to <https://share.streamlit.io> → sign in with GitHub → "New app" → pick repo + `app.py`.
3. First build takes 3–5 min (sentence-transformers + torch are heavy).

No environment variables required - users supply their own Groq key in the sidebar.

## Potential Enhancements

- Multi-turn conversation (rephrase follow-up questions before retrieval)
- `precision@k` evaluation on a held-out question set
- Hybrid search (BM25 + dense) for better recall on keyword queries
- Re-ranker (cross-encoder) on top-k results before generation
- Persist the FAISS index to disk to avoid re-embedding on every restart
- OCR for scanned PDFs (currently only text-based PDFs work)
- Support for `.docx`, `.xlsx`, `.pptx` (currently PDF / TXT / MD only)
