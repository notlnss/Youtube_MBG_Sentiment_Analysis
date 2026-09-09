"""
Streamlit App - Analisis Sentimen Komentar YouTube tentang Program MBG
========================================================================
Alur:
1. User masukkan link video YouTube + API Key YouTube Data API v3
2. App fetch semua komentar publik dari video tersebut
3. Tiap komentar dibersihkan (preprocessing sama seperti saat training)
4. Model ensemble IndoBERT (5-fold) memprediksi sentimen tiap komentar
5. Hasil diagregasi jadi persentase Positive/Neutral/Negative
"""

import re
import os
import numpy as np
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
from googleapiclient.discovery import build
import plotly.express as px

# ==========================================================================================
# KONFIGURASI
# ==========================================================================================

st.set_page_config(
    page_title="Sentimen Komentar YouTube - Program MBG",
    page_icon="📊",
    layout="wide",
)

LABEL_NAMES = ["Negative", "Neutral", "Positive"]   # HARUS sama urutannya dengan LabelEncoder saat training
MAX_LEN = 128

# Repo Hugging Face Hub tempat model ONNX quantized di-upload
# Ganti "username-kamu" sesuai akun HF tempat kamu upload modelnya
HF_MODEL_REPO = "notlnss/mbg-sentiment-indobert-onnx"


# ==========================================================================================
# TEXT PREPROCESSING (harus konsisten dengan preprocessing saat training)
# ==========================================================================================

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_SYMBOL_PATTERN = re.compile(r"#")
REPEAT_CHAR_PATTERN = re.compile(r"(.)\1{2,}")
NON_STANDARD_CHAR_PATTERN = re.compile(r"[^a-zA-Z0-9\s.,!?'\"-]")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_SYMBOL_PATTERN.sub("", text)
    text = REPEAT_CHAR_PATTERN.sub(r"\1\1", text)
    text = NON_STANDARD_CHAR_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==========================================================================================
# YOUTUBE — EKSTRAKSI VIDEO ID & FETCH KOMENTAR
# ==========================================================================================

def extract_video_id(url: str):
    """Ekstrak video ID dari berbagai format link YouTube."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_youtube_comments(video_id: str, api_key: str, max_comments: int = 500):
    """Ambil komentar publik dari sebuah video YouTube via Data API v3."""
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments - len(comments)),
            pageToken=next_page_token,
            textFormat="plainText",
            order="relevance",
        )
        response = request.execute()

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "author": snippet.get("authorDisplayName", ""),
                "comment": snippet.get("textDisplay", ""),
                "like_count": snippet.get("likeCount", 0),
                "published_at": snippet.get("publishedAt", ""),
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return pd.DataFrame(comments)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_video_details(video_id: str, api_key: str):
    """Ambil judul & thumbnail video YouTube."""
    youtube = build("youtube", "v3", developerKey=api_key)
    request = youtube.videos().list(part="snippet", id=video_id)
    response = request.execute()

    items = response.get("items", [])
    if not items:
        return None

    snippet = items[0]["snippet"]
    thumbnails = snippet.get("thumbnails", {})
    # Ambil resolusi terbaik yang tersedia
    thumbnail_url = (
        thumbnails.get("high", {}).get("url")
        or thumbnails.get("medium", {}).get("url")
        or thumbnails.get("default", {}).get("url")
    )

    return {
        "title": snippet.get("title", ""),
        "channel": snippet.get("channelTitle", ""),
        "thumbnail_url": thumbnail_url,
    }


# ==========================================================================================
# LOAD MODEL (cached — cuma di-load sekali per session, langsung dari Hugging Face Hub)
# ==========================================================================================

@st.cache_resource(show_spinner="Mengunduh & memuat model dari Hugging Face Hub...")
def load_model_and_tokenizer(repo_id):
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = ORTModelForSequenceClassification.from_pretrained(repo_id)
    return tokenizer, model


def softmax_np(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


def predict_sentiment(texts, tokenizer, model, batch_size=16):
    """Prediksi sentimen memakai model ONNX quantized (single model, tanpa ensemble)."""
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        outputs = model(**inputs)
        logits = outputs.logits
        if hasattr(logits, "numpy"):
            logits = logits.detach().cpu().numpy()
        probs = softmax_np(logits)
        all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    pred_ids = np.argmax(all_probs, axis=-1)
    pred_labels = [LABEL_NAMES[i] for i in pred_ids]
    confidences = np.max(all_probs, axis=-1)

    return pred_labels, confidences, all_probs


# ==========================================================================================
# UI - SIDEBAR
# ==========================================================================================

st.sidebar.title("⚙️ Konfigurasi")

# Coba ambil API Key dari Streamlit Secrets (kalau sudah di-setup di dashboard Streamlit Cloud)
# Kalau tidak ada, user bisa input manual sebagai fallback
default_api_key = st.secrets.get("YOUTUBE_API_KEY", "")

if default_api_key:
    api_key_input = default_api_key
    st.sidebar.success("✅ API Key sudah tersedia (bawaan aplikasi)")
else:
    api_key_input = st.sidebar.text_input(
        "YouTube Data API Key",
        type="password",
        help="Dapatkan gratis di Google Cloud Console → Enable 'YouTube Data API v3' → Buat API Key.",
    )

max_comments = st.sidebar.slider(
    "Maksimum komentar diambil", min_value=50, max_value=2000, value=300, step=50
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Cara dapat API Key gratis:**\n"
    "1. Buka [Google Cloud Console](https://console.cloud.google.com/)\n"
    "2. Buat project baru (atau pakai yang ada)\n"
    "3. Aktifkan **YouTube Data API v3**\n"
    "4. Buka menu Credentials → Create Credentials → API Key\n"
    "5. Copy API Key ke kolom di atas"
)

# ==========================================================================================
# UI - MAIN
# ==========================================================================================

st.title("📊 Analisis Sentimen Komentar YouTube — Program MBG")
st.caption("Model: IndoBERT ONNX Quantized (Int8) — di-load dari Hugging Face Hub")

video_url = st.text_input(
    "Masukkan link video YouTube:",
    placeholder="https://www.youtube.com/watch?v=xxxxxxxxxxx",
)

# --- Tampilkan preview judul & thumbnail video begitu link valid dimasukkan ---
if video_url:
    preview_video_id = extract_video_id(video_url)
    if preview_video_id and api_key_input:
        try:
            video_details = fetch_video_details(preview_video_id, api_key_input)
        except Exception:
            video_details = None

        if video_details:
            col_thumb, col_info = st.columns([1, 2])
            with col_thumb:
                if video_details["thumbnail_url"]:
                    st.image(video_details["thumbnail_url"], use_container_width=True)
            with col_info:
                st.markdown(f"**{video_details['title']}**")
                st.caption(f"Channel: {video_details['channel']}")
        else:
            st.warning("Video tidak ditemukan. Periksa kembali link-nya.")
    elif preview_video_id and not api_key_input:
        st.info("Masukkan API Key di sidebar untuk melihat preview video.")

analyze_button = st.button("🔍 Analisis Sentimen", type="primary")

# --- Proses analisis: hanya jalan saat tombol diklik, hasilnya disimpan ke session_state ---
if analyze_button:
    if not api_key_input:
        st.error("Mohon masukkan YouTube API Key di sidebar terlebih dahulu.")
    elif not video_url:
        st.error("Mohon masukkan link video YouTube.")
    else:
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("Link YouTube tidak valid. Pastikan formatnya benar.")
        else:
            with st.spinner("Mengambil komentar dari YouTube..."):
                try:
                    df_comments = fetch_youtube_comments(video_id, api_key_input, max_comments)
                except Exception as e:
                    st.error(f"Gagal mengambil komentar. Pastikan API Key valid dan video mengizinkan komentar publik.\n\nDetail: {e}")
                    st.stop()

            if df_comments.empty:
                st.warning("Tidak ada komentar ditemukan pada video ini.")
                st.stop()

            st.success(f"Berhasil mengambil {len(df_comments)} komentar.")

            with st.spinner("Memuat model & menjalankan prediksi sentimen..."):
                try:
                    tokenizer, model = load_model_and_tokenizer(HF_MODEL_REPO)
                except Exception as e:
                    st.error(
                        f"Gagal memuat model dari Hugging Face Hub ({HF_MODEL_REPO}). "
                        f"Pastikan repo ID benar dan modelnya publik.\n\nDetail: {e}"
                    )
                    st.stop()

                df_comments["clean_comment"] = df_comments["comment"].apply(clean_text)
                df_comments = df_comments[df_comments["clean_comment"].str.len() > 0].reset_index(drop=True)

                pred_labels, confidences, all_probs = predict_sentiment(
                    df_comments["clean_comment"].tolist(), tokenizer, model
                )
                df_comments["sentiment"] = pred_labels
                df_comments["confidence"] = confidences

            # Simpan hasil ke session_state supaya tidak hilang saat widget lain di-interaksi
            st.session_state["df_comments"] = df_comments
            st.session_state["video_id"] = video_id

# --- Tampilan hasil: selalu render berdasarkan session_state, tidak tergantung status tombol ---
if "df_comments" in st.session_state:
    df_comments = st.session_state["df_comments"]
    video_id = st.session_state["video_id"]

    # ==================================================================
    # RINGKASAN PERSENTASE
    # ==================================================================
    st.markdown("## 📈 Ringkasan Sentimen")

    total = len(df_comments)
    counts = df_comments["sentiment"].value_counts()
    pct_negative = counts.get("Negative", 0) / total * 100
    pct_neutral = counts.get("Neutral", 0) / total * 100
    pct_positive = counts.get("Positive", 0) / total * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("😡 Negative", f"{pct_negative:.1f}%", f"{counts.get('Negative', 0)} komentar")
    col2.metric("😐 Neutral", f"{pct_neutral:.1f}%", f"{counts.get('Neutral', 0)} komentar")
    col3.metric("😊 Positive", f"{pct_positive:.1f}%", f"{counts.get('Positive', 0)} komentar")

    if pct_negative >= 30:
        st.warning(f"⚠️ Sentimen negatif cukup tinggi ({pct_negative:.1f}%) — perlu perhatian lebih lanjut.")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        pie_df = pd.DataFrame({
            "Sentimen": ["Negative", "Neutral", "Positive"],
            "Jumlah": [counts.get("Negative", 0), counts.get("Neutral", 0), counts.get("Positive", 0)],
        })
        fig_pie = px.pie(
            pie_df, names="Sentimen", values="Jumlah",
            color="Sentimen",
            color_discrete_map={"Negative": "#EF553B", "Neutral": "#B0B0B0", "Positive": "#2CA02C"},
            title="Distribusi Sentimen",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        fig_bar = px.bar(
            pie_df, x="Sentimen", y="Jumlah", color="Sentimen",
            color_discrete_map={"Negative": "#EF553B", "Neutral": "#B0B0B0", "Positive": "#2CA02C"},
            title="Jumlah Komentar per Kelas",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ==================================================================
    # TABEL DETAIL KOMENTAR
    # ==================================================================
    st.markdown("## 🗒️ Detail Komentar")

    filter_sentiment = st.multiselect(
        "Filter berdasarkan sentimen:",
        options=["Negative", "Neutral", "Positive"],
        default=["Negative", "Neutral", "Positive"],
        key="filter_sentiment",
    )

    display_df = df_comments[df_comments["sentiment"].isin(filter_sentiment)][
        ["author", "comment", "sentiment", "confidence", "like_count"]
    ].sort_values("confidence", ascending=False)

    st.dataframe(display_df, use_container_width=True, height=400)
