"""
Search engine for procedure documents.
Uses keyword matching with synonym expansion and fuzzy matching
to find the best sections that answer a user's question.
"""

import json
import re
from pathlib import Path

INDEX_FILE = Path(__file__).parent / "extracted" / "index.json"

# Common synonyms and abbreviations used in motorsport/mechanical context
SYNONYMS = {
    "booklet": ["book", "manual", "guide", "log"],
    "book": ["booklet", "manual", "guide", "log"],
    "manual": ["book", "booklet", "guide"],
    "tyre": ["tire", "tyres", "tires"],
    "tire": ["tyre", "tyres", "tires"],
    "tyres": ["tyre", "tire", "tires"],
    "tires": ["tyre", "tire", "tyres"],
    "pressure": ["pressures", "psi", "bar"],
    "pressures": ["pressure", "psi", "bar"],
    "temp": ["temperature", "temps", "temperatures"],
    "temperature": ["temp", "temps", "temperatures"],
    "temps": ["temp", "temperature", "temperatures"],
    "install": ["installation", "installing", "fit", "fitting", "mount", "mounting"],
    "installation": ["install", "installing", "fit", "fitting", "mount", "mounting"],
    "remove": ["removal", "removing", "take off", "unmount"],
    "removal": ["remove", "removing", "take off", "unmount"],
    "check": ["checks", "checking", "inspect", "inspection", "verify"],
    "checks": ["check", "checking", "inspect", "inspection"],
    "torque": ["tighten", "tightening", "nm"],
    "storage": ["store", "storing", "transport"],
    "store": ["storage", "storing", "transport"],
    "oven": ["ovens", "heater", "heating", "warm"],
    "ovens": ["oven", "heater", "heating", "warm"],
    "rim": ["rims", "wheel", "wheels"],
    "rims": ["rim", "wheel", "wheels"],
    "wheel": ["wheels", "rim", "rims"],
    "wheels": ["wheel", "rim", "rims"],
    "sensor": ["sensors", "tpms"],
    "tpms": ["sensor", "sensors"],
    "cold": ["cold pressures", "before session"],
    "hot": ["hot pressures", "after session"],
    "bleed": ["bleeding", "bled"],
    "bleeding": ["bleed", "bled"],
    "change": ["changing", "swap", "swapping", "replace", "replacing"],
    "fit": ["fitting", "mount", "mounting", "install"],
    "fitting": ["fit", "mount", "mounting", "install"],
}


def load_index():
    """Load the document index from disk."""
    if not INDEX_FILE.exists():
        return {"documents": []}
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text):
    """Simple tokenizer: lowercase, split on non-alphanumeric, remove short words."""
    words = re.findall(r'[a-z0-9]+', text.lower())
    return [w for w in words if len(w) >= 2]


def expand_query(tokens):
    """Expand query tokens with synonyms."""
    expanded = set(tokens)
    for token in tokens:
        if token in SYNONYMS:
            expanded.update(SYNONYMS[token])
    return list(expanded)


def score_section(query_tokens, expanded_tokens, section):
    """Score how relevant a section is to the query."""
    title_text = section["title"].lower()
    title_tokens = tokenize(section["title"])
    content_text = " ".join(section["content"]).lower()
    content_tokens = tokenize(content_text)
    all_tokens = title_tokens + content_tokens

    if not all_tokens:
        return 0

    score = 0

    # Score original query tokens (higher weight)
    for qt in query_tokens:
        if qt in title_tokens:
            score += 15
        if qt in content_tokens:
            score += 5
        # Substring match in title
        if qt in title_text:
            score += 8

    # Score expanded/synonym tokens (lower weight)
    for et in expanded_tokens:
        if et in query_tokens:
            continue  # Already scored above
        if et in title_tokens:
            score += 8
        if et in content_tokens:
            score += 2

    # Bonus for multiple original query terms matching the same section
    matches = sum(1 for qt in query_tokens if qt in all_tokens)
    if matches >= 2:
        score += matches * 5

    # Bonus for phrase match in content
    query_str = " ".join(query_tokens)
    if query_str in content_text:
        score += 25
    if query_str in title_text:
        score += 30

    # Normalize by number of query tokens
    if query_tokens:
        score = score / len(query_tokens)

    return score


def search(query, top_k=5, min_score=1.0):
    """Search the procedure index for sections relevant to the query."""
    index = load_index()
    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    expanded_tokens = expand_query(query_tokens)

    results = []
    for doc in index["documents"]:
        for section in doc["sections"]:
            score = score_section(query_tokens, expanded_tokens, section)
            if score >= min_score:
                results.append({
                    "doc_name": doc["doc_name"],
                    "filename": doc["filename"],
                    "section_title": section["title"],
                    "content": section["content"],
                    "images": section["images"],
                    "score": score,
                })

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
    Build a context string from the most relevant sections.
    Only includes images from sections that are actually sent as context.
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
            break
        context_parts.append(section_block)
        total_chars += len(section_block)
        # Only add images from this specific section
        images.extend(r["images"])

    return "\n".join(context_parts), images
