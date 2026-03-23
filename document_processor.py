"""
Processes .docx procedure files into structured searchable data.
Extracts text organized by sections and saves embedded images.
Tables are processed in document order so they attach to the correct section.
"""

import os
import json
import hashlib
import re
from pathlib import Path
from docx import Document
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph
from lxml import etree

PROCEDURES_DIR = Path(__file__).parent / "procedures"
EXTRACTED_DIR = Path(__file__).parent / "extracted"
IMAGES_DIR = EXTRACTED_DIR / "images"
INDEX_FILE = EXTRACTED_DIR / "index.json"


def extract_images(doc, doc_id):
    """Extract all images from a document and save them, returning a mapping of rId -> filename."""
    image_map = {}
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            image_data = rel.target_part.blob
            ext = os.path.splitext(rel.target_part.partname)[1] or ".png"
            image_hash = hashlib.md5(image_data).hexdigest()[:8]
            filename = f"{doc_id}_{image_hash}{ext}"
            filepath = IMAGES_DIR / filename
            if not filepath.exists():
                with open(filepath, "wb") as f:
                    f.write(image_data)
            image_map[rel.rId] = filename
    return image_map


def get_paragraph_images(paragraph, image_map):
    """Find any images referenced in a paragraph."""
    images = []
    for run in paragraph.runs:
        xml = run._element.xml
        if "blip" in xml:
            rids = re.findall(r'r:embed="([^"]+)"', xml)
            for rid in rids:
                if rid in image_map:
                    images.append(image_map[rid])
    return images


def iter_block_items(doc):
    """
    Iterate over paragraphs AND tables in document order.
    This is critical — python-docx's doc.paragraphs and doc.tables
    are separate lists that lose ordering. This walks the XML tree
    to yield items in the order they appear in the document.
    """
    body = doc.element.body
    for child in body:
        tag = etree.QName(child).localname
        if tag == "p":
            yield Paragraph(child, doc)
        elif tag == "tbl":
            yield DocxTable(child, doc)


def format_table(table):
    """Convert a docx table to readable text."""
    rows = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        # Deduplicate merged cells (python-docx repeats merged cell text)
        deduped = []
        prev = None
        for c in cells:
            if c != prev:
                deduped.append(c)
            prev = c
        rows.append(" | ".join(deduped))
    return "\n".join(rows)


def process_document(filepath):
    """Process a single .docx file into structured sections."""
    doc = Document(filepath)
    doc_name = Path(filepath).stem
    doc_id = hashlib.md5(doc_name.encode()).hexdigest()[:8]

    # Extract images
    image_map = extract_images(doc, doc_id)

    sections = []
    current_section = {
        "title": doc_name,
        "level": 0,
        "content": [],
        "images": [],
    }

    for item in iter_block_items(doc):
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if not text and not get_paragraph_images(item, image_map):
                continue

            style_name = item.style.name if item.style else ""
            para_images = get_paragraph_images(item, image_map)

            # Check if this is a heading
            is_heading = "Heading" in style_name or "heading" in style_name
            heading_level = 0
            if is_heading:
                try:
                    heading_level = int("".join(filter(str.isdigit, style_name)) or "1")
                except ValueError:
                    heading_level = 1

            if is_heading and text:
                # Save the current section if it has content
                if current_section["content"] or current_section["images"]:
                    sections.append(current_section)

                current_section = {
                    "title": text,
                    "level": heading_level,
                    "content": [],
                    "images": [],
                }
            else:
                if text:
                    current_section["content"].append(text)
                if para_images:
                    current_section["images"].extend(para_images)

        elif isinstance(item, DocxTable):
            # Tables are now processed in order, attached to current section
            table_text = format_table(item)
            if table_text.strip():
                current_section["content"].append(f"[Table]\n{table_text}")

    # Don't forget the last section
    if current_section["content"] or current_section["images"]:
        sections.append(current_section)

    return {
        "filename": Path(filepath).name,
        "doc_name": doc_name,
        "doc_id": doc_id,
        "sections": sections,
    }


def build_index():
    """Process all .docx files in the procedures directory and build the search index."""
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    index = {"documents": []}

    docx_files = list(PROCEDURES_DIR.glob("*.docx"))
    if not docx_files:
        print(f"No .docx files found in {PROCEDURES_DIR}")
        print(f"Please place your procedure documents in: {PROCEDURES_DIR.resolve()}")
        return index

    for filepath in docx_files:
        print(f"Processing: {filepath.name}")
        try:
            doc_data = process_document(filepath)
            index["documents"].append(doc_data)
            print(f"  -> {len(doc_data['sections'])} sections extracted")
        except Exception as e:
            print(f"  -> Error processing {filepath.name}: {e}")

    # Save index
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"\nIndex saved to {INDEX_FILE}")
    print(f"Images saved to {IMAGES_DIR}")
    return index


if __name__ == "__main__":
    build_index()
