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

    # Penalise sections with very little content — they match semantically
    # via parent/title context but have nothing useful to show
    for i, section in enumerate(sections):
        content_length = sum(len(line) for line in section.get("content", []))
        if content_length < 20:
            similarities[i] *= 0.3  # Heavy penalty for empty/near-empty sections
        elif content_length < 80:
            similarities[i] *= 0.7  # Moderate penalty for very short sections

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


def get_sibling_sections(section_title, section_parent):
    """Get all sibling sections under the same parent heading."""
    if not section_parent:
        return []
    index = load_index()
    siblings = []
    for doc in index["documents"]:
        for section in doc["sections"]:
            if (section.get("parent") == section_parent
                    and section["title"] != section_title
                    and (section.get("content") or section.get("images"))):
                siblings.append({
                    "doc_name": doc["doc_name"],
                    "section_title": section["title"],
                    "content": section.get("content", []),
                    "images": section.get("images", []),
                    "parent": section.get("parent", ""),
                })
    return siblings


def _get_all_image_contexts():
    """Build a map of image filename -> context description from the index."""
    index = load_index()
    image_ctx = {}
    for doc in index["documents"]:
        for section in doc["sections"]:
            for img_file, ctx in section.get("image_contexts", {}).items():
                image_ctx[img_file] = ctx.get("description", section["title"])
    return image_ctx


def filter_relevant_images(query, candidate_images, threshold=0.25):
    """
    Use semantic similarity to filter images to only those relevant to the query.
    Compares the query against each image's context description.
    """
    if not candidate_images:
        return []

    image_ctx = _get_all_image_contexts()
    model = _get_model()

    # Build descriptions for candidate images
    image_descriptions = []
    image_files = []
    for img in candidate_images:
        desc = image_ctx.get(img, "")
        if desc:
            image_descriptions.append(desc)
            image_files.append(img)

    if not image_descriptions:
        return []

    # Encode query and image descriptions
    query_embedding = model.encode(query, normalize_embeddings=True)
    desc_embeddings = model.encode(image_descriptions, normalize_embeddings=True)

    # Calculate similarity
    similarities = np.dot(desc_embeddings, query_embedding)

    # Only return images above the relevance threshold
    relevant = []
    for i, score in enumerate(similarities):
        if score >= threshold:
            relevant.append((image_files[i], float(score)))

    # Sort by relevance
    relevant.sort(key=lambda x: x[1], reverse=True)
    return [img for img, _score in relevant]


def get_context_for_llm(query, max_sections=3, max_chars=5000):
    """
    Build a context string from the most relevant sections.
    When a matched section has a parent, also includes sibling sections
    for complete context. Images are filtered by semantic relevance.
    """
    results = search(query, top_k=max_sections)
    if not results:
        return None, []

    context_parts = []
    all_candidate_images = []
    total_chars = 0
    included_titles = set()

    for r in results:
        if r["section_title"] in included_titles:
            continue

        section_text = "\n".join(r["content"])
        section_block = f"## {r['section_title']} (from: {r['doc_name']})\n{section_text}\n"
        if total_chars + len(section_block) > max_chars and context_parts:
            break
        context_parts.append(section_block)
        total_chars += len(section_block)
        included_titles.add(r["section_title"])

        # Collect candidate images from matched sections
        all_candidate_images.extend(r["images"])

        # Include sibling section TEXT for complete context (but NOT their images)
        if r.get("parent"):
            siblings = get_sibling_sections(r["section_title"], r["parent"])
            for sib in siblings:
                if sib["section_title"] in included_titles:
                    continue
                sib_text = "\n".join(sib["content"])
                sib_block = f"## {sib['section_title']} (from: {sib['doc_name']})\n{sib_text}\n"
                if total_chars + len(sib_block) > max_chars:
                    break
                context_parts.append(sib_block)
                total_chars += len(sib_block)
                included_titles.add(sib["section_title"])

    # Deduplicate candidate images
    seen = set()
    unique_candidates = []
    for img in all_candidate_images:
        if img not in seen:
            seen.add(img)
            unique_candidates.append(img)

    return "\n".join(context_parts), unique_candidates
