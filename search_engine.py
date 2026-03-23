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


STOPWORDS = {
    "how", "do", "does", "did", "the", "is", "are", "was", "were", "be",
    "to", "of", "and", "in", "on", "at", "for", "with", "from", "by",
    "an", "it", "its", "this", "that", "what", "which", "who", "whom",
    "can", "could", "would", "should", "will", "shall", "may", "might",
    "have", "has", "had", "not", "no", "or", "if", "then", "than",
    "so", "as", "up", "out", "about", "into", "over", "after", "before",
    "me", "my", "we", "our", "you", "your", "they", "them", "their",
    "use", "used", "using", "need", "want", "get", "tell", "show",
    "where", "when", "why",
}


def tokenize(text):
    """Simple tokenizer: lowercase, split on non-alphanumeric, remove short words."""
    words = re.findall(r'[a-z0-9]+', text.lower())
    return [w for w in words if len(w) >= 2]


def tokenize_query(text):
    """Tokenize a user query, removing stopwords to focus on meaningful terms."""
    tokens = tokenize(text)
    meaningful = [w for w in tokens if w not in STOPWORDS]
    # If all words were stopwords, fall back to all tokens
    return meaningful if meaningful else tokens


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
    content_tokens = set(tokenize(content_text))  # Use set to avoid counting duplicates
    all_title_tokens = set(title_tokens)

    if not all_title_tokens and not content_tokens:
        return 0

    score = 0

    # Title matches are heavily weighted — the title is the best signal
    for qt in query_tokens:
        if qt in all_title_tokens:
            score += 25  # Strong title match
        elif qt in title_text:
            score += 15  # Substring in title
        if qt in content_tokens:
            score += 3   # Content match (low weight to avoid noise)

    # Synonym matches in title only (not content — too noisy)
    for et in expanded_tokens:
        if et in query_tokens:
            continue
        if et in all_title_tokens:
            score += 12
        elif et in title_text:
            score += 8

    # Big bonus for ALL query terms matching in title
    title_matches = sum(1 for qt in query_tokens if qt in all_title_tokens or qt in title_text)
    if title_matches == len(query_tokens) and len(query_tokens) >= 2:
        score += 40

    # Bonus for phrase match
    query_str = " ".join(query_tokens)
    if query_str in title_text:
        score += 50
    if query_str in content_text:
        score += 15

    # Normalize by number of query tokens
    if query_tokens:
        score = score / len(query_tokens)

    return score


def search(query, top_k=5, min_score=1.0):
    """Search the procedure index for sections relevant to the query."""
    index = load_index()
    query_tokens = tokenize_query(query)

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
