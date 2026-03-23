"""
Search engine for procedure documents.
Uses keyword matching with synonym expansion, document name matching,
and fuzzy matching to find the best sections that answer a user's question.
"""

import json
import re
from pathlib import Path
from difflib import SequenceMatcher

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
    "naming": ["markings", "marking", "identification", "label", "labelling", "name", "format"],
    "name": ["naming", "markings", "identification", "label", "format"],
    "markings": ["marking", "naming", "identification", "label", "labelling", "name"],
    "marking": ["markings", "naming", "identification", "label", "labelling"],
    "identification": ["markings", "naming", "label", "identify"],
    "label": ["labelling", "markings", "marking", "naming", "identification", "sticker"],
    "sticker": ["stickers", "label", "marking"],
    "stickers": ["sticker", "label", "marking"],
    "clean": ["cleaning", "wash", "washing"],
    "cleaning": ["clean", "wash", "washing"],
    "weight": ["weights", "balance"],
    "weights": ["weight", "balance"],
    "balance": ["weights", "weight"],
    "damage": ["damaged", "crack", "cracked", "broken"],
    "procedure": ["process", "steps", "method", "guide", "setup"],
    "setup": ["set up", "setting up", "procedure", "process"],
    "process": ["procedure", "setup", "steps", "method"],
    "set": ["sets"],
    "sets": ["set"],
    "session": ["run", "stint"],
    "grid": ["pre-grid", "pregrid"],
    "lmp3": ["lmp4"],
    "lmp4": ["lmp3"],
    "camber": ["alignment", "angle"],
    "ride": ["ride height", "height"],
    "height": ["ride height", "ride"],
    "toe": ["alignment", "angle"],
    "corner": ["corners", "wheel"],
    "spring": ["springs", "stiffness"],
    "springs": ["spring", "stiffness"],
    "arb": ["anti roll bar", "antiroll", "roll bar", "sway bar"],
    "damper": ["dampers", "shock", "shocks", "absorber"],
    "dampers": ["damper", "shock", "shocks"],
    "front": ["fr", "fl"],
    "rear": ["rr", "rl"],
    "explain": ["explanation", "describe", "description"],
    "replace": ["replacing", "replacement", "swap", "change"],
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
    "where", "when", "why", "give", "short", "explain", "please",
    "description", "explanation", "summary",
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


def fuzzy_match(word, target_words, threshold=0.75):
    """Check if a word fuzzy-matches any word in target_words."""
    for tw in target_words:
        if len(tw) >= 3 and len(word) >= 3:
            ratio = SequenceMatcher(None, word, tw).ratio()
            if ratio >= threshold:
                return True
    return False


def score_section(query_tokens, expanded_tokens, section, doc_name):
    """Score how relevant a section is to the query."""
    title_text = section["title"].lower()
    title_tokens = tokenize(section["title"])

    # Include parent heading in title matching
    parent = section.get("parent", "")
    if parent:
        title_text = f"{parent.lower()} {title_text}"
        title_tokens = tokenize(parent) + title_tokens

    # Document name is a critical signal
    doc_name_lower = doc_name.lower()
    doc_name_tokens = set(tokenize(doc_name))

    content_text = " ".join(section["content"]).lower()
    content_tokens = set(tokenize(content_text))
    all_title_tokens = set(title_tokens)

    # Combined searchable text (title + doc name)
    combined_title = f"{doc_name_lower} {title_text}"
    combined_tokens = all_title_tokens | doc_name_tokens

    if not combined_tokens and not content_tokens:
        return 0

    score = 0

    # --- DOCUMENT NAME MATCHING (highest priority) ---
    doc_matches = 0
    for qt in query_tokens:
        if qt in doc_name_tokens:
            score += 30  # Very strong signal
            doc_matches += 1
        elif qt in doc_name_lower:
            score += 20  # Substring match in doc name
            doc_matches += 1
        elif fuzzy_match(qt, doc_name_tokens):
            score += 15  # Fuzzy match in doc name
            doc_matches += 1

    # --- TITLE MATCHING ---
    title_matches = 0
    for qt in query_tokens:
        if qt in all_title_tokens:
            score += 25
            title_matches += 1
        elif qt in title_text:
            score += 15
            title_matches += 1
        elif fuzzy_match(qt, all_title_tokens):
            score += 10
            title_matches += 1

    # --- CONTENT MATCHING ---
    content_matches = 0
    for qt in query_tokens:
        if qt in content_tokens:
            score += 3
            content_matches += 1
        elif qt in content_text:
            score += 2
            content_matches += 1

    # --- SYNONYM MATCHING (title + doc name only) ---
    for et in expanded_tokens:
        if et in query_tokens:
            continue
        if et in all_title_tokens:
            score += 15  # Synonym in section title is a strong signal
        elif et in doc_name_tokens:
            score += 8   # Synonym in doc name is weaker
        elif et in combined_title:
            score += 6

    # --- BONUSES ---

    # Big bonus when ALL query terms found somewhere in title + doc name
    combined_matches = 0
    for qt in query_tokens:
        if qt in combined_tokens or qt in combined_title or fuzzy_match(qt, combined_tokens):
            combined_matches += 1
    if combined_matches == len(query_tokens) and len(query_tokens) >= 2:
        score += 50

    # Phrase match bonus
    query_str = " ".join(query_tokens)
    if query_str in combined_title:
        score += 60
    if query_str in content_text:
        score += 15

    # Content relevance bonus: if most query terms appear in content
    if len(query_tokens) >= 2 and content_matches >= len(query_tokens) * 0.7:
        score += 15

    # Bonus for sections with substantial content (they're more useful)
    content_length = len(content_text)
    if content_length > 200:
        score += 5
    if content_length > 500:
        score += 5

    # Penalty for generic/low-value section names
    generic_titles = {"notes", "notes & improvements", "summary", "overview", "general", "other", "misc"}
    if section["title"].lower().strip() in generic_titles:
        score *= 0.5

    # Normalize by number of query tokens to keep scoring consistent
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
        doc_name = doc["doc_name"]
        for section in doc["sections"]:
            score = score_section(query_tokens, expanded_tokens, section, doc_name)
            if score >= min_score:
                results.append({
                    "doc_name": doc_name,
                    "filename": doc["filename"],
                    "section_title": section["title"],
                    "content": section["content"],
                    "images": section["images"],
                    "parent": section.get("parent", ""),
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


def get_sibling_images(section_title, section_parent):
    """Get images from sibling sections (same parent heading)."""
    if not section_parent:
        return []
    index = load_index()
    images = []
    for doc in index["documents"]:
        for section in doc["sections"]:
            if section.get("parent") == section_parent and section["images"]:
                images.extend(section["images"])
    return images


def get_context_for_llm(query, max_sections=3, max_chars=4000):
    """
    Build a context string from the most relevant sections.
    Includes images from the matched section and its siblings (same parent).
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
        images.extend(r["images"])
        # Also grab images from sibling sections under the same parent
        images.extend(get_sibling_images(r["section_title"], r.get("parent", "")))

    # Deduplicate images while preserving order
    seen = set()
    unique_images = []
    for img in images:
        if img not in seen:
            seen.add(img)
            unique_images.append(img)

    return "\n".join(context_parts), unique_images
