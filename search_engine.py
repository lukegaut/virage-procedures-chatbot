"""
Semantic search engine for procedure documents.
Uses sentence-transformers to understand meaning, not just keywords.
Any question phrased any way will find the right section.
"""

import json
import numpy as np
from pathlib import Path

INDEX_FILE = Path(__file__).parent / "extracted" / "index.json"
EMBEDDINGS_FILE = Path(__file__).parent / "extracted" / "embeddings.npz"

# Cache the model and embeddings in memory
_model = None
_embeddings_cache = None
_sections_cache = None


def _get_model():
    """Load the sentence-transformer model (cached after first call)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # Small, fast, high-quality model — runs on CPU, ~80MB
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def load_index():
    """Load the document index from disk."""
    if not INDEX_FILE.exists():
        return {"documents": []}
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_spelling(text):
    """Normalize common spelling variants so search matches regardless of spelling."""
    import re
    # tire/tires (American) → tyre/tyres (British) — standardize to British
    text = re.sub(r'\btires\b', 'tyres', text, flags=re.IGNORECASE)
    text = re.sub(r'\btire\b', 'tyre', text, flags=re.IGNORECASE)
    return text


def _build_section_text(section, doc_name):
    """
    Build a rich text representation of a section for embedding.
    Includes doc name, title, and substantial content so the embedding
    captures what the section is actually about — not just its title.
    For PDF sections (sparse text, mostly images), the title is repeated
    to ensure it dominates the embedding since the content is visual.
    """
    parts = []
    parts.append(f"Document: {doc_name}")
    parent = section.get("parent", "")
    if parent and parent != doc_name:
        parts.append(f"Parent: {parent}")
    parts.append(f"Section: {section['title']}")

    content = "\n".join(section.get("content", []))

    if section.get("is_page_render", False):
        # PDF sections: content is sparse (mostly image labels).
        # Repeat title to anchor the embedding on the section topic.
        parts.append(f"Topic: {section['title']}")
        if content:
            parts.append(content[:1500])
    else:
        # DOCX sections: rich text content, use more of it.
        # The model truncates at ~256 tokens but including more text
        # ensures important terms near the end aren't lost.
        if content:
            parts.append(content[:3000])

    return _normalize_spelling("\n".join(parts))


def build_embeddings():
    """
    Generate embeddings for all sections in the index.
    Called after documents are processed / index is rebuilt.
    """
    index = load_index()
    if not index["documents"]:
        return

    model = _get_model()
    texts = []
    sections_meta = []

    for doc in index["documents"]:
        doc_name = doc["doc_name"]
        for section in doc["sections"]:
            text = _build_section_text(section, doc_name)
            texts.append(text)
            sections_meta.append({
                "doc_name": doc_name,
                "filename": doc["filename"],
                "section_title": section["title"],
                "content": section["content"],
                "images": section["images"],
                "parent": section.get("parent", ""),
                "is_page_render": section.get("is_page_render", False),
            })

    if not texts:
        return

    # Generate embeddings in batch (fast)
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    # Save embeddings and metadata
    np.savez(
        EMBEDDINGS_FILE,
        embeddings=embeddings,
        meta=json.dumps(sections_meta),
    )

    # Update cache
    global _embeddings_cache, _sections_cache
    _embeddings_cache = embeddings
    _sections_cache = sections_meta

    return len(texts)


def _load_embeddings():
    """Load cached embeddings from disk."""
    global _embeddings_cache, _sections_cache
    if _embeddings_cache is not None:
        return _embeddings_cache, _sections_cache

    if not EMBEDDINGS_FILE.exists():
        # Auto-build if missing
        build_embeddings()
        if _embeddings_cache is not None:
            return _embeddings_cache, _sections_cache
        return None, None

    data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    _embeddings_cache = data["embeddings"]
    _sections_cache = json.loads(str(data["meta"]))
    return _embeddings_cache, _sections_cache


def search(query, top_k=5, min_score=0.15):
    """
    Semantic search: find the most relevant sections by meaning.
    Uses cosine similarity from embeddings, with a small title-match
    bonus to help distinguish sections with similar content.
    """
    embeddings, sections = _load_embeddings()
    if embeddings is None or sections is None or len(sections) == 0:
        return []

    model = _get_model()

    # Normalize spelling variants before encoding
    query = _normalize_spelling(query)

    # Encode the query
    query_embedding = model.encode(query, normalize_embeddings=True)

    # Cosine similarity (embeddings are already normalized, so dot product = cosine sim)
    similarities = np.dot(embeddings, query_embedding)

    # Build query word set for title matching
    query_words = set(query.lower().split())

    for i, section in enumerate(sections):
        # Penalise sections with very little content (likely empty/filler)
        content_length = sum(len(line) for line in section.get("content", []))
        if content_length < 20:
            similarities[i] *= 0.3
        elif content_length < 80:
            similarities[i] *= 0.7

        # Small title-match bonus: when query words appear in the section title,
        # add a small additive bonus. This helps distinguish semantically similar
        # sections (e.g. "Tyre Changes" vs "Driver Change" when query says "tyre").
        title_lower = section["section_title"].lower().replace("-", " ")
        title_words = set(title_lower.split())
        matching = query_words & title_words
        # Only count meaningful matches (3+ chars, not stopwords)
        meaningful = {w for w in matching if len(w) >= 3}
        if meaningful:
            similarities[i] += 0.03 * len(meaningful)


    # Get top results
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        if score < min_score:
            continue
        section = sections[idx]
        results.append({
            "doc_name": section["doc_name"],
            "filename": section["filename"],
            "section_title": section["section_title"],
            "content": section["content"],
            "images": section["images"],
            "parent": section.get("parent", ""),
            "is_page_render": section.get("is_page_render", False),
            "score": score,
        })

    return results


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


def get_context_for_llm(query, max_chars=6000, max_images_to_ai=5, max_images_display=10):
    """
    Build context from the best matching SECTION.
    For PDFs: find the best section, return its images for vision.
    For DOCX: return text from matched sections.
    Returns (context_text, ai_images, display_images, use_vision).
    """
    results = search(query, top_k=5, min_score=0.15)
    if not results:
        return None, [], [], False

    # The top result is our primary match
    best = results[0]
    has_page_renders = best.get("is_page_render", False)

    if has_page_renders:
        # PDF mode: use the SINGLE best-matching section
        # All images and text come from this one section — no mixing
        context_parts = []
        text = "\n".join(best["content"])
        context_parts.append(f"## {best['section_title']} (from: {best['doc_name']})\n{text}")

        ai_images = best["images"][:max_images_to_ai]
        display_images = best["images"][:max_images_display]
        # Sort display images by filename to maintain page order
        display_images.sort()

        context = "\n".join(context_parts)
        return context, ai_images, display_images, True

    else:
        # DOCX mode: text only, cheap. Can include multiple sections from same doc.
        primary_doc = best["doc_name"]
        primary_results = [r for r in results if r["doc_name"] == primary_doc]

        context_parts = []
        all_images = []
        seen_images = set()
        total_chars = 0

        for r in primary_results:
            if not r["content"]:
                continue
            section_text = "\n".join(r["content"])
            section_block = f"## {r['section_title']} (from: {primary_doc})\n{section_text}\n"
            if total_chars + len(section_block) > max_chars and context_parts:
                break
            context_parts.append(section_block)
            total_chars += len(section_block)

            for img in r["images"]:
                if img not in seen_images:
                    all_images.append(img)
                    seen_images.add(img)

        return "\n".join(context_parts), [], all_images, False
