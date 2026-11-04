import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extract import (
    extract_action_items,
    extract_key_decisions,
    extract_question,
)
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="centered",
)

st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 2rem; max-width: 820px;}

    .hero {
        text-align: center;
        padding: 2.2rem 1rem 1.6rem 1rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
    }
    .hero h1 {margin: 0; font-size: 2.2rem; font-weight: 800; color: white;}
    .hero p {margin: .5rem 0 0 0; font-size: 1.02rem; opacity: .92;}

    .card {
        background: rgba(127, 127, 127, 0.07);
        border: 1px solid rgba(127, 127, 127, 0.18);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }
    .card h4 {margin: 0 0 .5rem 0; font-size: 1.05rem;}

    .video-title {
        font-size: 1.5rem; font-weight: 700; margin: .2rem 0 1rem 0;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        padding: .65rem 1rem;
        font-weight: 600;
        border: none;
        color: white;
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
    }
    div.stButton > button:hover {
        filter: brightness(1.1);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🎬 AI Video Assistant</h1>
        <p>Turn any video into a transcript, a summary, and a conversation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def reset():
    st.session_state.results = None
    st.session_state.messages = []


# ----------------------------------------------------------------------------
# Pipeline (same steps as the CLI version, with live progress)
# ----------------------------------------------------------------------------
def run_pipeline_ui(source: str) -> dict:
    with st.status("Analyzing your video…", expanded=True) as status:
        st.write("📥 Processing video/audio input…")
        chunks = process_input(source)

        st.write("📝 Transcribing audio…")
        transcript = transcribe_all(chunks, language="english")

        st.write("🏷️ Generating title…")
        title = generate_title(transcript)

        st.write("🧠 Writing summary…")
        summary = summarize(transcript)

        st.write("✅ Extracting action items…")
        action_items = extract_action_items(transcript)

        st.write("🔎 Extracting decisions and open questions…")
        decisions = extract_key_decisions(transcript)
        questions = extract_question(transcript)

        st.write("🔗 Building Q&A engine…")
        rag_chain = build_rag_chain(transcript)

        status.update(label="Analysis complete!", state="complete", expanded=False)

    return {
        "title": title,
        "summary": summary,
        "transcript": transcript,
        "action_items": action_items,
        "decisions": decisions,
        "questions": questions,
        "rag_chain": rag_chain,
    }


def save_upload(uploaded_file) -> str:
    """Save an uploaded file to a temp path and return that path."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


# ----------------------------------------------------------------------------
# Input section
# ----------------------------------------------------------------------------
if st.session_state.results is None:
    tab_url, tab_upload = st.tabs(["🔗  Paste a URL", "📁  Upload a file"])

    source = None

    with tab_url:
        url = st.text_input(
            "Video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed",
        )
        if st.button("Analyze video", key="go_url"):
            if url.strip():
                source = url.strip()
            else:
                st.warning("Please enter a video URL.")

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a video or audio file",
            type=["mp4", "mov", "mkv", "avi", "webm", "mp3", "wav", "m4a"],
            label_visibility="collapsed",
        )
        if uploaded:
            st.caption(f"{uploaded.name} · {uploaded.size / (1024 * 1024):.1f} MB")
        if st.button("Analyze video", key="go_file"):
            if uploaded:
                source = save_upload(uploaded)
            else:
                st.warning("Please upload a file first.")

    if source:
        try:
            st.session_state.results = run_pipeline_ui(source)
            st.session_state.messages = []
            st.rerun()
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ----------------------------------------------------------------------------
# Results + Q&A
# ----------------------------------------------------------------------------
else:
    r = st.session_state.results

    st.markdown(f'<div class="video-title">{r["title"]}</div>', unsafe_allow_html=True)

    tab_sum, tab_chat, tab_tx = st.tabs(["📋  Summary", "💬  Ask questions", "📄  Transcript"])

    with tab_sum:
        st.markdown(
            f'<div class="card"><h4>Summary</h4></div>', unsafe_allow_html=True
        )
        st.write(r["summary"])

        with st.expander("✅ Action items", expanded=True):
            st.write(r["action_items"])
        with st.expander("🎯 Key decisions"):
            st.write(r["decisions"])
        with st.expander("❓ Open questions"):
            st.write(r["questions"])

    with tab_chat:
        if not st.session_state.messages:
            st.info("Ask anything about the video — the answer will come from its content.")

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if prompt := st.chat_input("Ask a question about the video…"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        answer = ask_question(r["rag_chain"], prompt)
                    except Exception as e:
                        answer = f"Error generating answer: {e}"
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    with tab_tx:
        st.text_area(
            "Transcript",
            r["transcript"],
            height=420,
            label_visibility="collapsed",
        )

    st.divider()
    if st.button("⬅ Analyze another video"):
        reset()
        st.rerun()