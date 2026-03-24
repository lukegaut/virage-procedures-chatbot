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


def _build_section_text(section, doc_name):
    """
    Build a rich text representation of a section for embedding.
    Combines doc name, parent, title, and content for best semantic matching.
    """
    parts = []
    parts.append(f"Document: {doc_name}")
    parent = section.get("parent", "")
    if parent:
        parts.append(f"Section: {parent}")
    parts.append(f"Topic: {section['title']}")
    content = "\n".join(section.get("content", []))
    if content:
        # Limit content to avoid overwhelming the embedding
        parts.append(content[:1500])
    return "\n".join(parts)


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
    Returns sections sorted by relevance (cosine similarity).
    """
    embeddings, sections = _load_embeddings()
    if embeddings is None or sections is None or len(sections) == 0:
        return []

    model = _get_model()

    # Encode the query
    query_embedding = model.encode(query, normalize_embeddings=True)

    # Cosine similarity (embeddings are already normalized, so dot product = cosine sim)
    similarities = np.dot(embeddings, query_embedding)

    # Extract words from query for exact name matching
    query_words = set(query.lower().split())

    for i, section in enumerate(sections):
        # Penalise sections with very little content
        content_length = sum(len(line) for line in section.get("content", []))
        if content_length < 20:
            similarities[i] *= 0.3
        elif content_length < 80:
            similarities[i] *= 0.7

        # Boost sections where the document name or section title contains
        # an exact word match from the query (e.g. "LES" matches "- LES" not "- ELMS")
        doc_name_upper = section["doc_name"].upper()
        title_upper = section["section_title"].upper()
        for word in query_words:
            w = word.upper()
            if len(w) >= 2:  # Skip very short words
                # Check for exact word boundary match in doc name
                doc_words = set(doc_name_upper.replace("-", " ").replace("_", " ").split())
                title_words = set(title_upper.replace("-", " ").replace("_", " ").split())
                if w in doc_words or w in title_words:
                    similarities[i] *= 1.3  # Boost for exact name match

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



def get_context_for_llm(query, max_chars=6000, max_images_to_ai=5, max_images_display=10):
    """
    Build context from the top search results.
    For PDFs (page renders): sends best-matching pages to Claude via vision.
    For DOCX: sends extracted text (cheap).
    Returns (context_text, ai_images, display_images, use_vision).
    - ai_images: images to send to Claude API (limited, for vision)
    - display_images: images to show the user (can be more)
    """
    results = search(query, top_k=10, min_score=0.15)
    if not results:
        return None, [], [], False

    # The top result determines the primary document
    primary_doc_name = results[0]["doc_name"]
    primary_filename = results[0]["filename"]
    has_page_renders = results[0].get("is_page_render", False)

    if has_page_renders:
        # PDF mode: get ALL pages from primary document in document order
        index = load_index()
        primary_doc = None
        for doc in index["documents"]:
            if doc["filename"] == primary_filename:
                primary_doc = doc
                break

        if not primary_doc:
            return None, [], [], False

        # Get the top-scoring page indices from search results
        matched_pages = [r for r in results if r["doc_name"] == primary_doc_name]
        matched_images = set()
        for r in matched_pages:
            for img in r["images"]:
                matched_images.add(img)

        # Collect ALL pages from the document in order for display
        all_display_images = []
        context_parts = []
        for section in primary_doc["sections"]:
            for img in section["images"]:
                all_display_images.append(img)
            # Add text for context (used alongside images)
            if section["content"]:
                text = "\n".join(section["content"])
                context_parts.append(f"## {section['title']}\n{text}")

        # For AI: only send the top-matched pages (sorted by score, capped)
        ai_images = []
        for r in matched_pages[:max_images_to_ai]:
            for img in r["images"]:
                if img not in ai_images:
                    ai_images.append(img)

        context = "\n".join(context_parts) if context_parts else f"Document: {primary_doc_name}"
        return context, ai_images, all_display_images[:max_images_display], True

    else:
        # DOCX mode: text only, cheap
        primary_results = [r for r in results if r["doc_name"] == primary_doc_name]

        context_parts = []
        all_images = []
        seen_images = set()
        total_chars = 0

        for r in primary_results:
            if not r["content"]:
                continue
            section_text = "\n".join(r["content"])
            section_block = f"## {r['section_title']} (from: {primary_doc_name})\n{section_text}\n"
            if total_chars + len(section_block) > max_chars and context_parts:
                break
            context_parts.append(section_block)
            total_chars += len(section_block)

            for img in r["images"]:
                if img not in seen_images:
                    all_images.append(img)
                    seen_images.add(img)

        return "\n".join(context_parts), [], all_images, False
