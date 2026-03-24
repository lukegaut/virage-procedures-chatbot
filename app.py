"""
Virage Procedures Chatbot
A chatbot for mechanics and engineers to query procedure documents.
Uses Claude AI with vision to understand text, tables, and diagrams.
"""

import base64
import io
import streamlit as st
from pathlib import Path
from PIL import Image
from anthropic import Anthropic
from search_engine import search, get_context_for_llm, get_document_list, load_index, build_embeddings
from document_processor import build_index, PROCEDURES_DIR, IMAGES_DIR, EXTRACTED_DIR

# --- Page Config ---
st.set_page_config(
    page_title="Virage Procedures Chatbot",
    page_icon="🔧",
    layout="wide",
)


def check_admin_login():
    """Login gate for the admin page only."""
    if st.session_state.get("admin_authenticated"):
        return True

    st.title("🔒 Admin Login")
    st.caption("Enter credentials to access the admin center.")

    with st.form("admin_login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", use_container_width=True)

    if submitted:
        valid_user = st.secrets.get("APP_USERNAME", "engineers")
        valid_pass = st.secrets.get("APP_PASSWORD", "RACE2win")
        if username == valid_user and password == valid_pass:
            st.session_state["admin_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect username or password.")
    return False


# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)


SYSTEM_PROMPT = """You are a technical procedures assistant for Virage motorsport engineers and mechanics.
Your role is to help them understand and follow procedures accurately.

CRITICAL RULES:
1. ONLY use the provided procedure context to answer questions. Do NOT make up information.
2. If the context does not contain enough information to answer, say so clearly and suggest what they could ask instead.
3. Never contradict or modify the procedure steps - relay them accurately.
4. Use clear, simple language suitable for workshop use.
5. If safety warnings or torque specs are mentioned, ALWAYS include them.
6. When referencing specific steps, mention which procedure document they come from.
7. Format your response for easy reading - use bullet points and numbered steps where appropriate.
8. Keep answers focused and concise - mechanics need quick answers, not essays.
9. You will be shown page images from procedure documents. Read and use ALL information visible in them including diagrams, tables, annotations, and text.
10. When a diagram or image is important for understanding (e.g. layout diagrams, step-by-step photos), tell the user to check the images below for the visual reference."""



def encode_image_for_api(img_path, max_width=800):
    """Encode and resize an image for the Claude API."""
    img = Image.open(img_path)
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    if img.mode in ("RGBA", "LA", "P"):
        img.save(buf, format="PNG", optimize=True)
        media_type = "image/png"
    else:
        img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=75)
        media_type = "image/jpeg"
    return media_type, base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def get_ai_response(query, context, images=None, use_vision=False, chat_history=""):
    """
    Get a Claude AI response.
    - For DOCX content: text-only (cheap, fast)
    - For PDF content: sends page images so Claude can read diagrams (vision mode)
    """
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "⚠️ API key not configured. Add ANTHROPIC_API_KEY in the app settings."

    client = Anthropic(api_key=api_key)

    history_section = ""
    if chat_history:
        history_section = f"\nRECENT CONVERSATION:\n{chat_history}\n"

    if use_vision and images:
        # Vision mode: send page images for Claude to read
        content = []
        content.append({
            "type": "text",
            "text": f"Below are pages from procedure documents. Read ALL text, tables, diagrams, and annotations visible in the images to answer the question.\n{history_section}\nUSER QUESTION: {query}\n\nAnswer using ONLY what you can see in the page images. Tell the user to check the procedure page images shown below your answer for visual details."
        })

        images_added = 0
        for img_file in images:
            img_path = IMAGES_DIR / img_file
            if img_path.exists() and images_added < 5:
                try:
                    media_type, img_data = encode_image_for_api(img_path)
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": img_data,
                        }
                    })
                    images_added += 1
                except Exception:
                    pass

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
            temperature=0.3,
        )
    else:
        # Text mode: cheap, fast, for well-extracted DOCX content
        user_prompt = f"""Based on the following procedure documentation, answer the user's question.
{history_section}
PROCEDURE CONTEXT:
{context}

USER QUESTION: {query}

Answer using ONLY the information from the procedure context above. If the user is asking a follow-up question, use the conversation history to understand what they are referring to. Related images from the procedures will be shown separately below your answer."""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.3,
        )

    return response.content[0].text


def display_images(images):
    """Display procedure images in the chat."""
    shown = set()
    for img_file in images:
        if img_file in shown:
            continue
        shown.add(img_file)
        img_path = IMAGES_DIR / img_file
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)


def get_recent_chat_context():
    """Get recent chat history for follow-up question understanding."""
    messages = st.session_state.get("messages", [])
    if not messages:
        return ""
    recent = messages[-4:]
    parts = []
    for m in recent:
        role = "User" if m["role"] == "user" else "Assistant"
        parts.append(f"{role}: {m['content'][:300]}")
    return "\n".join(parts)


def build_search_query(prompt):
    """Combine current prompt with previous context for follow-up questions."""
    chat_context = get_recent_chat_context()
    if not chat_context:
        return prompt
    words = prompt.strip().split()
    if len(words) <= 6:
        messages = st.session_state.get("messages", [])
        for m in reversed(messages):
            if m["role"] == "user":
                return f"{m['content']} {prompt}"
        return prompt
    return prompt


# --- Page routing ---
page = st.sidebar.radio("", ["💬 Chat", "⚙️ Admin"], label_visibility="collapsed")

if page == "⚙️ Admin":
    # ========================
    #  ADMIN CENTER (login required)
    # ========================
    if not check_admin_login():
        st.stop()

    st.title("⚙️ Admin Center")
    st.caption("Upload new procedure documents or remove existing ones.")

    st.divider()

    # --- Upload new documents ---
    st.subheader("📤 Upload Procedure Documents")
    uploaded_files = st.file_uploader(
        "Upload procedure documents (.docx, .pdf, .pptx)",
        type=["docx", "pdf", "pptx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            filepath = PROCEDURES_DIR / uploaded_file.name
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded: {uploaded_file.name}")

        # Auto-rebuild index and embeddings after upload
        with st.spinner("Rebuilding document index and search embeddings..."):
            index = build_index(api_key=st.secrets.get("ANTHROPIC_API_KEY", ""))
            build_embeddings()
            doc_count = len(index["documents"])
            section_count = sum(len(d["sections"]) for d in index["documents"])
        st.success(f"Index rebuilt: {doc_count} document(s), {section_count} sections")

    st.divider()

    # --- Manage existing documents ---
    st.subheader("📄 Existing Documents")
    docs = sorted(
        list(PROCEDURES_DIR.glob("*.docx"))
        + list(PROCEDURES_DIR.glob("*.pdf"))
        + list(PROCEDURES_DIR.glob("*.pptx")),
        key=lambda p: p.name,
    )

    if not docs:
        st.info("No procedure documents loaded yet. Upload some above.")
    else:
        for doc_path in docs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"📄 {doc_path.name}")
            with col2:
                if st.button("🗑️ Remove", key=f"del_{doc_path.name}"):
                    doc_path.unlink()
                    st.success(f"Removed: {doc_path.name}")
                    with st.spinner("Rebuilding index..."):
                        build_index(api_key=st.secrets.get("ANTHROPIC_API_KEY", ""))
                        build_embeddings()
                    st.rerun()

    st.divider()

    # --- Rebuild index manually ---
    if st.button("🔄 Rebuild Document Index", use_container_width=True):
        with st.spinner("Processing documents and building search embeddings..."):
            index = build_index(api_key=st.secrets.get("ANTHROPIC_API_KEY", ""))
            num_sections = build_embeddings()
            doc_count = len(index["documents"])
            section_count = sum(len(d["sections"]) for d in index["documents"])
            st.success(f"Indexed {doc_count} document(s), {section_count} sections, {num_sections} embeddings")

    # --- API key status ---
    st.divider()
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if api_key:
        st.success("🤖 AI: Connected (Claude)")
    else:
        st.error("🤖 AI: No API key")
        st.caption("Add ANTHROPIC_API_KEY to secrets")

else:
    # ========================
    #  CHAT INTERFACE
    # ========================
    st.title("🔧 Virage Procedures Chatbot")
    st.caption("Ask questions about procedures — answers are generated from your documentation only.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("images"):
                display_images(message["images"])

    # Chat input
    if prompt := st.chat_input("Ask about a procedure..."):
        index = load_index()
        if not index["documents"]:
            st.warning("⚠️ No documents indexed yet. Go to the Admin page to upload procedure documents.")
        else:
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Build search query with conversation context for follow-ups
            search_query = build_search_query(prompt)
            chat_history = get_recent_chat_context()

            # Search for relevant context
            context, ai_images, display_imgs, use_vision = get_context_for_llm(search_query)

            # Generate AI response
            with st.chat_message("assistant"):
                if not context and not ai_images and not display_imgs:
                    display_text = "I couldn't find any relevant information in the procedures for that question. Could you try rephrasing or being more specific?"
                    display_imgs = []
                else:
                    with st.spinner("Searching procedures..."):
                        try:
                            display_text = get_ai_response(
                                prompt, context, ai_images,
                                use_vision=use_vision,
                                chat_history=chat_history,
                            )
                        except Exception as e:
                            display_text = f"Error generating response: {e}"

                st.markdown(display_text)

                # Show procedure images directly (always visible)
                if display_imgs:
                    display_images(display_imgs)

                # Show sources
                results = search(search_query, top_k=3)
                if results:
                    with st.expander("📋 Sources"):
                        for r in results:
                            st.markdown(f"- **{r['section_title']}** from _{r['doc_name']}_")

            st.session_state.messages.append({
                "role": "assistant",
                "content": display_text,
                "images": display_imgs if display_imgs else [],
            })
