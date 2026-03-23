"""
Virage Procedures Chatbot
A chatbot for mechanics and engineers to query procedure documents.
"""

import streamlit as st
from pathlib import Path
from groq import Groq
from search_engine import search, get_context_for_llm, get_document_list, load_index
from document_processor import build_index, PROCEDURES_DIR, IMAGES_DIR

# --- Page Config ---
st.set_page_config(
    page_title="Virage Procedures Chatbot",
    page_icon="🔧",
    layout="wide",
)

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
2. If the context does not contain enough information to answer, say so clearly and suggest what they could ask instead based on topics you can see in the context.
3. Never contradict or modify the procedure steps - relay them accurately.
4. Use clear, simple language suitable for workshop use.
5. If safety warnings or torque specs are mentioned in the procedures, ALWAYS include them.
6. When referencing specific steps, mention which procedure document they come from.
7. Format your response for easy reading - use bullet points and numbered steps where appropriate.
8. Keep answers focused and concise - mechanics need quick answers, not essays."""


def get_ai_response(query, context):
    """Get an AI response from Groq, grounded in the procedure context."""
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        return "⚠️ API key not configured. Add GROQ_API_KEY in the app settings."

    client = Groq(api_key=api_key)

    user_prompt = f"""Based on the following procedure documentation, answer the user's question.

PROCEDURE CONTEXT:
{context}

USER QUESTION: {query}

Answer using ONLY the information from the procedure context above."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


def display_images(images):
    """Display procedure images in the chat."""
    for img_file in images:
        img_path = IMAGES_DIR / img_file
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)


# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")

    # Rebuild index button
    if st.button("🔄 Rebuild Document Index", use_container_width=True):
        with st.spinner("Processing documents..."):
            index = build_index()
            doc_count = len(index["documents"])
            section_count = sum(len(d["sections"]) for d in index["documents"])
            st.success(f"Indexed {doc_count} document(s), {section_count} sections")

    st.divider()

    # Show indexed documents
    st.subheader("📄 Indexed Documents")
    docs = get_document_list()
    if docs:
        for doc in docs:
            st.write(f"• {doc}")
    else:
        st.warning("No documents indexed yet. Place .docx files in the `procedures` folder and click 'Rebuild Document Index'.")

    st.divider()

    # API key status
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if api_key:
        st.success("🤖 AI: Connected (Groq)")
    else:
        st.error("🤖 AI: No API key")
        st.caption("Add GROQ_API_KEY to secrets")

    st.divider()
    st.caption("Place procedure .docx files in:")
    st.code(str(PROCEDURES_DIR.resolve()), language=None)

# --- Main Chat Interface ---
st.title("🔧 Virage Procedures Chatbot")
st.caption("Ask questions about procedures — answers are generated from your documentation only.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "images" in message and message["images"]:
            display_images(message["images"])

# Chat input
if prompt := st.chat_input("Ask about a procedure... (e.g., 'How do I change a tyre?')"):
    # Check if index exists
    index = load_index()
    if not index["documents"]:
        st.warning("⚠️ No documents indexed yet. Place .docx files in the `procedures` folder and click 'Rebuild Document Index' in the sidebar.")
    else:
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Search for relevant context
        context, images = get_context_for_llm(prompt, max_sections=3)

        # Generate AI response
        with st.chat_message("assistant"):
            if not context:
                response_text = "I couldn't find any relevant information in the procedures for that question. Could you try rephrasing or being more specific?"
            else:
                with st.spinner("Searching procedures..."):
                    try:
                        response_text = get_ai_response(prompt, context)
                    except Exception as e:
                        response_text = f"Error generating response: {e}"

            st.markdown(response_text)

            # Show relevant images
            if images:
                st.divider()
                st.caption("📸 Related procedure images:")
                display_images(images)

            # Show sources
            results = search(prompt, top_k=5)
            if results:
                with st.expander("📋 Sources"):
                    for r in results:
                        st.markdown(f"- **{r['section_title']}** from _{r['doc_name']}_")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "images": images if images else [],
        })
