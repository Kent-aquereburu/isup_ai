
# ==========================================================
# Streamlit (FR) :
#  - Pré-traitement PDF avec DOCLING (layout, sections, tables, images)
#  - Transcription AUDIO (MP3/M4A/WAV) -> texte
#  - RAG moderne : embeddings + recherche sémantique + LLM
#  - Backends : Ollama (local) OU OpenAI (cloud)
#
# Lancer :
#   streamlit run app_rag.py
#.  put your open ai key in .streamlit/secrets.toml file : OPENAI_API_KEY="sk-xxxx"

# ==========================================================

import io
import re
import tempfile
from typing import List, Tuple

import numpy as np
import requests
import streamlit as st

# ---------- PDF fallback ----------
from pypdf import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
import subprocess
import os
import tempfile

# ---------- OpenAI ----------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------- Docling ----------
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling_core.types.doc import PictureItem, TableItem
    DOCLING_OK = True
except Exception:
    DOCLING_OK = False

# ---------- Audio Whisper (local) ----------
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_OK = True
except Exception:
    FASTER_WHISPER_OK = False


# ==========================================================
# UI CONFIG
# ==========================================================
st.set_page_config(page_title="Docling + Audio + RAG (FR)", layout="wide")
st.title("Démo Pipeline de traitement des documents IA  : Docling (PDF OU Audio) → Conversion Decoupage en paragraphes → RAG et LLM)")


# ==========================================================
# UTILITAIRES TEXTE / CHUNKING
# ==========================================================
def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, chunk_size=900, overlap=150) -> List[str]:
    text = normalize_text(text)
    chunks = []
    step = max(1, chunk_size - overlap)
    i = 0
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        i += step
    return chunks


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


# ==========================================================
# EMBEDDINGS
# ==========================================================
def embed_openai(text: str, model: str) -> np.ndarray:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    res = client.embeddings.create(model=model, input=text)
    return np.array(res.data[0].embedding, dtype=np.float32)


def embed_ollama(text: str, model: str, base_url: str) -> np.ndarray:
    # 1) endpoint le plus courant
    r = requests.post(
        f"{base_url}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=180
    )
    if r.status_code == 404:
        # 2) fallback pour certaines versions
        r = requests.post(
            f"{base_url}/api/embed",
            json={"model": model, "input": text},
            timeout=180
        )

    r.raise_for_status()
    j = r.json()

    # /api/embeddings -> {"embedding":[...]}
    if "embedding" in j and j["embedding"]:
        vec = j["embedding"]
        return np.array(vec, dtype=np.float32)

    # /api/embed -> {"embeddings":[[...]]} parfois
    if "embeddings" in j and j["embeddings"]:
        vec = j["embeddings"][0]
        return np.array(vec, dtype=np.float32)

    raise RuntimeError(f"Réponse embeddings Ollama inattendue: {j}")


def render_gallery(title: str, images, cols: int = 3):
    st.markdown(f"### {title} ({len(images)})")
    if not images:
        st.info(f"Aucun élément détecté pour: {title}")
        return

    # Affichage en grille
    grid = st.columns(cols)
    for i, img in enumerate(images):
        with grid[i % cols]:
            st.image(img, use_container_width=True)


# ==========================================================
# CHAT
# ==========================================================
def chat_openai(prompt: str, model: str) -> str:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    resp = client.responses.create(model=model, input=prompt)
    out = []
    for item in resp.output:
        if item.type == "message":
            for c in item.content:
                if c.type == "output_text":
                    out.append(c.text)
    return "\n".join(out)

def chat_ollama(prompt: str, model: str, base_url: str) -> str:
    # Utilise l'API OpenAI-compatible d'Ollama
    r = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=300,
    )

    # fallback completions si jamais
    if r.status_code == 404:
        r = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "temperature": 0.2,
                "max_tokens": 900,
            },
            timeout=300,
        )

    r.raise_for_status()
    j = r.json()

    # /v1/chat/completions
    if "choices" in j and j["choices"]:
        msg = j["choices"][0].get("message", {})
        if "content" in msg:
            return str(msg["content"]).strip()
        if "text" in j["choices"][0]:
            return str(j["choices"][0]["text"]).strip()

    raise RuntimeError(f"Réponse Ollama inattendue: {j}")

# ==========================================================
# RAG PROMPT
# ==========================================================
def build_rag_prompt(question: str, passages: List[Tuple[int, float, str]]) -> str:
    ctx = "\n\n".join([f"[{rank}] {txt}" for rank, _, txt in passages])
    return f"""
Tu es un assistant d’analyse documentaire.

Règles :
- Réponds uniquement avec les informations présentes dans les extraits.
- Si l’information n’est pas dans les extraits, dis :
  "Je ne le vois pas dans les extraits fournis."
- Réponse claire, structurée.
- Ajoute des citations au format [1], [2].

Question : {question}

Extraits :
{ctx}
"""


# ==========================================================
# DOCLING PDF
# ==========================================================
def docling_convert(pdf_bytes: bytes):
    pipeline = PdfPipelineOptions()
    pipeline.images_scale = 2.0
    pipeline.generate_page_images = True
    pipeline.generate_picture_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)
        }
    )

    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/doc.pdf"
        with open(path, "wb") as f:
            f.write(pdf_bytes)

        conv = converter.convert(path)
        doc = conv.document

        md = doc.export_to_markdown()

        pages, tables, pictures = [], [], []
        for _, page in doc.pages.items():
            if page.image:
                pages.append(page.image.pil_image)

        for el, _ in doc.iterate_items():
            if isinstance(el, TableItem):
                try:
                    tables.append(el.get_image(doc))
                except:
                    pass
            if isinstance(el, PictureItem):
                try:
                    pictures.append(el.get_image(doc))
                except:
                    pass

        return md, pages, tables, pictures


# ==========================================================
# AUDIO
# ==========================================================
@st.cache_resource
def load_whisper(model_size: str):
    return WhisperModel(model_size, device="cpu", compute_type="int8")

@st.cache_resource
def transcribe_audio_local(audio_bytes: bytes, filename: str, model_size="small", language="fr") -> str:
    # IMPORTANT : audio_bytes doit venir de audio_up.getvalue()
    ext = "." + filename.split(".")[-1].lower()

    # Si c'est déjà un wav => on le garde tel quel (pas de ffmpeg)
    is_wav = ext == ".wav" and len(audio_bytes) > 12 and audio_bytes[0:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f_in:
        f_in.write(audio_bytes)
        f_in.flush()
        in_path = f_in.name

    # Si WAV valide => pas de conversion
    wav_path = in_path if is_wav else (in_path + ".wav")

    try:
        if not is_wav:
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", in_path, "-ac", "1", "-ar", "16000", "-vn", wav_path
            ]
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0:
                raise RuntimeError(
                    "FFmpeg n'arrive pas à décoder ce fichier.\n"
                    "👉 Essaie un WAV PCM (pcm_s16le) ou un M4A propre.\n\n"
                    f"Erreur ffmpeg:\n{p.stderr.strip()}"
                )

        model = load_whisper(model_size)
        segments, _ = model.transcribe(wav_path, language=language)
        return " ".join([seg.text.strip() for seg in segments if seg.text])

    finally:
        # Nettoyage
        for path in {in_path, wav_path}:
            try:
                os.remove(path)
            except Exception:
                pass

@st.cache_resource
def transcribe_audio_openai(audio_bytes: bytes, filename: str, model: str) -> str:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    bio = io.BytesIO(audio_bytes)
    bio.name = filename
    res = client.audio.transcriptions.create(
        file=bio,
        model=model,
        language="fr",
    )
    return res.text


# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.header("Mode LLM")
mode = st.sidebar.selectbox("Backend", ["Ollama (local)", "OpenAI (cloud)"])

if mode == "Ollama (local)":
    ollama_url = st.sidebar.text_input("Ollama URL", "http://localhost:11434")
    llm_model = st.sidebar.text_input("LLM", "llama3.1")
    emb_model = st.sidebar.text_input("Embeddings", "nomic-embed-text")
else:
    llm_model = st.sidebar.text_input("LLM", "gpt-4.1-mini")
    emb_model = st.sidebar.text_input("Embeddings", "text-embedding-3-small")
    



st.sidebar.header("RAG")
top_k = st.sidebar.slider("Top-K", 2, 8, 4)
chunk_size = st.sidebar.slider("Chunk size", 500, 1400, 900)



overlap = st.sidebar.slider("Overlap", 50, 300, 150)


# ==========================================================
# STATE
# ==========================================================
if "pivot_text" not in st.session_state:
    st.session_state.pivot_text = None
    st.session_state.chunks = None
    st.session_state.vecs = None


# ==========================================================
# TABS
# ==========================================================
tab1, tab2, tab3 = st.tabs(
    ["A) PDF (Docling)", "B) Audio (MP3)", "C) RAG (Q&A)"]
)

# ---------------- PDF ----------------
with tab1:
    st.subheader("PDF → structure + texte + visuels")
    pdf = st.file_uploader("Charge un PDF", type=["pdf"])
    if pdf and DOCLING_OK:
        if st.button("Analyser avec Docling"):
            md, pages, tables, pics = docling_convert(pdf.read())
            st.session_state.pivot_text = md

            st.success("PDF analysé — TEXTE pivot prêt")
            st.text_area("Markdown extrait", md, height=250)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("### Pages")
                for im in pages[:8]:
                    st.image(im, use_container_width=True)
            with c2:
                st.markdown("### Tables")
                for im in tables:
                    st.image(im, use_container_width=True)
            with c3:
                st.markdown("### Images")
                for im in pics:
                    st.image(im, use_container_width=True)

# ---------------- AUDIO ----------------
with tab2:
    st.subheader("Audio → transcription → TEXTE pivot")
    audio = st.file_uploader("Charge un audio", type=["mp3", "wav", "m4a"])
    if audio:
        st.audio(audio.read())
        if st.button("Transcrire audio"):
            audio_bytes = audio.getvalue()
            if mode == "OpenAI (cloud)":
                text = transcribe_audio_openai(
                    audio_bytes, audio.name, "gpt-4o-mini-transcribe"
                )
            else:
                text = transcribe_audio_local(
                    audio_bytes, "." + audio.name.split(".")[-1], "small"
                )
            st.session_state.pivot_text = text
            st.success("Audio transcrit — TEXTE pivot prêt")
            st.text_area("Transcript", text[:8000], height=250)

# ---------------- RAG ----------------
with tab3:
    st.subheader("RAG : recherche sémantique + LLM")
    question = st.text_area(
        "Question",
        "1/ Quels sont les principales garanties du contrat auto ? 2/Est ce que la formule tiers plus garantit le bris de glace ?",
    )

    if st.session_state.pivot_text:
        if st.button("Indexer"):
            chunks = chunk_text(
                st.session_state.pivot_text, chunk_size, overlap
            )
            vecs = []
            for c in chunks:
                if mode == "Ollama (local)":
                    vecs.append(embed_ollama(c, emb_model, ollama_url))
                else:
                    vecs.append(embed_openai(c, emb_model))
            st.session_state.chunks = chunks
            st.session_state.vecs = vecs
            st.success(f"Index créé ({len(chunks)} chunks)")

        if st.session_state.chunks:
            if st.button("Répondre"):
                if mode == "Ollama (local)":
                    qv = embed_ollama(question, emb_model, ollama_url)
                else:
                    qv = embed_openai(question, emb_model)

                scores = [
                    (cosine_sim(qv, v), i)
                    for i, v in enumerate(st.session_state.vecs)
                ]
                scores.sort(reverse=True)
                top = scores[:top_k]

                passages = [
                    (rank, s, st.session_state.chunks[i])
                    for rank, (s, i) in enumerate(top, 1)
                ]

                with st.expander("Passages utilisés"):
                    for r, s, t in passages:
                        st.markdown(f"**[{r}] score={s:.3f}**")
                        st.write(t[:800])

                prompt = build_rag_prompt(question, passages)
                if mode == "Ollama (local)":
                    answer = chat_ollama(prompt, llm_model, ollama_url)
                else:
                    answer = chat_openai(prompt, llm_model)

                st.markdown("### Réponse")
                st.write(answer)
    else:
        st.info("Charge un PDF ou un audio pour commencer.")

st.markdown("---")

