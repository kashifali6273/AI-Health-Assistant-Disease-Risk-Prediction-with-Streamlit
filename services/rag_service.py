# services/rag_service.py
"""
RAG helper for ai-health-assistant.

Usage:
- Run build_index() once to create the FAISS index from services/kb/health_faq.md
- In your app, call load_index() once, then call retrieve(...) or answer_query(...)
"""

import os
import faiss
import joblib
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import pathlib
import requests

# Paths (file lives in services/)
BASE_DIR = pathlib.Path(__file__).parent.resolve()
KB_PATH = BASE_DIR / "kb" / "health_faq.md"
INDEX_PATH = BASE_DIR / "kb" / "faiss_index.bin"
DOCS_PATH = BASE_DIR / "kb" / "documents.pkl"
EMB_MODEL_NAME = "all-MiniLM-L6-v2"   # compact, fast

# --- Utilities ---
def _load_kb_text(kb_path: pathlib.Path) -> str:
    with open(kb_path, "r", encoding="utf-8") as f:
        return f.read()

def _split_into_passages(text: str, max_chars: int = 600) -> List[str]:
    """
    Very simple splitter: split by blank line, then further split long blocks
    into ~max_chars chunks (preserving sentences roughly).
    """
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    passages = []
    for b in blocks:
        if len(b) <= max_chars:
            passages.append(b)
        else:
            # naive chunking on sentence boundaries
            sentences = [s.strip() for s in b.split(". ") if s.strip()]
            chunk = ""
            for s in sentences:
                if len(chunk) + len(s) + 2 <= max_chars:
                    chunk = (chunk + " " + s).strip()
                else:
                    if chunk:
                        passages.append(chunk)
                    chunk = s
            if chunk:
                passages.append(chunk)
    return passages

# --- Index building & saving ---
def build_index(model_name: str = EMB_MODEL_NAME, overwrite: bool = False) -> None:
    """
    Build FAISS index from services/kb/health_faq.md and save index + docs.
    Run once after editing the KB, or with overwrite=True.
    """
    if INDEX_PATH.exists() and DOCS_PATH.exists() and not overwrite:
        print("Index already exists. Use overwrite=True to rebuild.")
        return

    print("Loading KB from:", KB_PATH)
    text = _load_kb_text(KB_PATH)
    passages = _split_into_passages(text)
    print(f"Split KB into {len(passages)} passages.")

    # Load embedding model
    model = SentenceTransformer(model_name)
    embeddings = model.encode(passages, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]

    # Build FAISS index (IndexFlatIP with normalized vectors -> cosine sim)
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Save index and passages
    faiss.write_index(index, str(INDEX_PATH))
    joblib.dump(passages, DOCS_PATH)

    print("Saved index to:", INDEX_PATH)
    print("Saved passages to:", DOCS_PATH)

# --- Loading index & retrieval ---
_global_index = None
_global_passages = None
_global_model = None

def load_index(model_name: str = EMB_MODEL_NAME):
    """
    Load FAISS index, passages, and sentence-transformer model into global vars.
    Call once at app start.
    """
    global _global_index, _global_passages, _global_model
    if _global_index is not None:
        return

    if not INDEX_PATH.exists() or not DOCS_PATH.exists():
        raise FileNotFoundError("Index or documents not found. Run build_index() first.")

    _global_index = faiss.read_index(str(INDEX_PATH))
    _global_passages = joblib.load(DOCS_PATH)
    _global_model = SentenceTransformer(model_name)
    print("RAG index loaded. Passages:", len(_global_passages))

def retrieve(query: str, k: int = 3) -> List[Tuple[int, float, str]]:
    """
    Retrieve top-k passages for the query.
    Returns list of tuples: (rank_index, score, passage_text)
    """
    if _global_index is None:
        load_index()

    q_emb = _global_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = _global_index.search(q_emb, k)
    scores = D[0].tolist()
    ids = I[0].tolist()
    results = []
    for idx, score in zip(ids, scores):
        if idx < 0 or idx >= len(_global_passages):
            continue
        results.append((idx, float(score), _global_passages[idx]))
    return results

# --- Answer composition ---
def _compose_fallback_answer(retrieved: List[Tuple[int, float, str]], risk_score: float = None) -> str:
    """
    Create a simple, safe answer using retrieved passages and an optional risk_score.
    """
    parts = []
    if risk_score is not None:
        parts.append(f"Model risk score: {risk_score:.2f} (0-1 scale).")
        if risk_score < 0.33:
            parts.append("Interpretation: Low risk.")
        elif risk_score < 0.66:
            parts.append("Interpretation: Medium risk.")
        else:
            parts.append("Interpretation: High risk.")
    parts.append("Relevant info from knowledge base:")
    for i, (_, score, passage) in enumerate(retrieved, start=1):
        parts.append(f"{i}. {passage}")
    parts.append("\nNote: This is informational only and not medical advice.")
    return "\n\n".join(parts)

def _call_gemini_api(query: str, passages: list, risk_score: float = None, api_key: str = None):
    """
    Call Google Gemini API (free quota) for RAG answers.
    """
    if api_key is None:
        raise ValueError("Gemini API key is required.")

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}

    context = "\n".join(passages)
    prompt = (
        "You are a friendly health assistant for education only (not medical advice).\n"
        "Use ONLY the provided knowledge passages. Explain clearly, in 3-4 sentences.\n\n"
        f"Knowledge passages:\n{context}\n\n"
        f"User question: {query}\n"
        + (f"Model risk score: {risk_score:.2f}\n" if risk_score else "")
        + "\nAnswer:"
    )

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, headers=headers, params=params, json=payload)
    response.raise_for_status()
    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return "Sorry, I couldn’t generate a Gemini answer."

def answer_query(query: str, risk_score: float = None, k: int = 3,
                 use_llm: bool = False, gemini_api_key: str = None) -> dict:
    """
    Main helper to answer user queries.
    - If use_llm=True and gemini_api_key provided, will call Gemini API (free quota).
    - Otherwise returns fallback concatenation of retrieved KB passages + simple interpretation.
    Returns a dict: { "answer": str, "retrieved": [texts], "use_llm": bool }
    """
    if _global_index is None:
        load_index()

    retrieved = retrieve(query, k=k)
    passages = [p for (_, _, p) in retrieved]

    # If LLM requested and Gemini key provided
    if use_llm and gemini_api_key:
        try:
            answer = _call_gemini_api(query, passages, risk_score, api_key=gemini_api_key)
            return {"answer": answer, "retrieved": passages, "use_llm": True, "provider": "gemini"}
        except Exception as e:
            fallback = _compose_fallback_answer(retrieved, risk_score)
            return {"answer": fallback, "retrieved": passages, "use_llm": False, "error": str(e)}

    # Default: fallback
    fallback = _compose_fallback_answer(retrieved, risk_score)
    return {"answer": fallback, "retrieved": passages, "use_llm": False}
if __name__ == "__main__":
    build_index(overwrite=True)
