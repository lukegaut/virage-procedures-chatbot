"""
Search engine for procedure documents.
Uses keyword matching and relevance scoring to find the best sections
that answer a user's question.
"""

import json
import re
from pathlib import Path

INDEX_FILE = Path(__file__).parent / "extracted" / "index.json"


def load_index():
    """Load the document index from disk."""
    if not INDEX_FILE.exists():
        return {"documents": []}
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text):
    """Simple tokenizer: lowercase, split on non-alphanumeric, remove short words."""
    words = re.findall(r'[a-z0-9]+', text.lower())
    # Keep short words that might be important (e.g., "psi", "bar", "nut")
    return [w for w in words if len(w) >= 2]


def score_section(query_tokens, section):
    """Score how relevant a section is to the query."""
    title_tokens = tokenize(section["title"])
    content_text = " ".join(section["content"])
    content_tokens = tokenize(content_text)
    all_tokens = title_tokens + content_tokens

    if not all_tokens:
        return 0

    score = 0
    for qt in query_tokens:
        # Exact match in title (high weight)
        if qt in title_tokens:
            score += 10
        # Exact match in content
        if qt in content_tokens:
            score += 3
        # Partial match (substring) in content
        for ct in all_tokens:
            if qt in ct or ct in qt:
                score += 1
                break

    # Bonus for matching multiple query terms (phrase-like matching)
    query_str = " ".join(query_tokens)
    if query_str in content_text.lower():
        score += 20

    # Normalize by number of query tokens to avoid bias
    if query_tokens:
        score = score / len(query_tokens)

    return score


def search(query, top_k=5, min_score=1.0):
    """
    Search the procedure index for sections relevant to the query.
    Returns a list of (document_name, section, score) tuples.
    """
    index = load_index()
    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    results = []
    for doc in index["documents"]:
        for section in doc["sections"]:
            score = score_section(query_tokens, section)
            if score >= min_score:
                results.append({
                    "doc_name": doc["doc_name"],
                    "filename": doc["filename"],
                    "section_title": section["title"],
                    "content": section["content"],
                    "images": section["images"],
                    "score": score,
                })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def get_all_sections():
    """Return all sections from all documents (for browsing)."""
    index = load_index()
    sections = []
    for doc in index["documents"]:
        for section in doc["sections"]:
            sections.append({
                "doc_name": doc["doc_name"],
                "section_title": section["title"],
                "content": section["content"],
                "images": section["images"],
            })
    return sections


def get_document_list():
    """Return list of all indexed documents."""
    index = load_index()
    return [doc["doc_name"] for doc in index["documents"]]


def get_context_for_llm(query, max_sections=3, max_chars=4000):
    """
    Build a context string from the most relevant sections
    to feed into an LLM for answer generation.
    Caps total context size to stay within token limits.
    """
    results = search(query, top_k=max_sections)
    if not results:
        return None, []

    context_parts = []
    images = []
    total_chars = 0
    for r in results:
        section_text = "\n".join(r["content"])
        section_block = f"## {r['section_title']} (from: {r['doc_name']})\n{section_text}\n"
        if total_chars + len(section_block) > max_chars and context_parts:
            break  # Stop adding sections if we'd exceed the limit
        context_parts.append(section_block)
        total_chars += len(section_block)
        images.extend(r["images"])

    return "\n".join(context_parts), images
