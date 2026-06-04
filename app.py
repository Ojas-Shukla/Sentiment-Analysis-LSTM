import streamlit as st
import pickle
import numpy as np
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.big-title {
    text-align: center;
    font-size: 3rem;
    font-weight: bold;
    color: #FF4B4B;
}

.subtitle {
    text-align: center;
    color: #888888;
    margin-bottom: 2rem;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}

.positive {
    background-color:#d4edda;
    color:#155724;
}

.negative {
    background-color:#f8d7da;
    color:#721c24;
}

.footer {
    text-align:center;
    color:gray;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_resources():
    model = load_model("sentiment_lstm.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

model, tokenizer = load_resources()

lemmatizer = WordNetLemmatizer()

MAX_LEN = 200

# -------------------------
# Preprocessing
# -------------------------
def preprocess_text(text):

    words = text.lower().split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)

# -------------------------
# Prediction
# -------------------------
def predict_sentiment(review):

    review = preprocess_text(review)

    sequence = tokenizer.texts_to_sequences([review])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN
    )

    prediction = model.predict(
        padded,
        verbose=0
    )[0][0]

    return prediction

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:

    st.title("📌 Project Info")

    st.markdown("""
    **Model:** LSTM

    **Dataset:** IMDb 50K Reviews

    **Preprocessing**
    - Lowercasing
    - HTML Removal
    - URL Removal
    - Stopword Removal
    - Lemmatization

    **Frameworks**
    - TensorFlow
    - NLTK
    - Streamlit
    """)

    st.divider()

    st.markdown("### Sample Reviews")

    if st.button("Positive Example"):
        st.session_state.sample = \
        "This movie was fantastic. Great acting and storyline."

    if st.button("Negative Example"):
        st.session_state.sample = \
        "Worst movie ever. Complete waste of time."

# -------------------------
# Header
# -------------------------
st.markdown(
    "<div class='big-title'>🎬 IMDb Sentiment Analyzer</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Analyze Movie Reviews using Deep Learning (LSTM)</div>",
    unsafe_allow_html=True
)

# -------------------------
# Input
# -------------------------
review = st.text_area(
    "Enter a movie review",
    value=st.session_state.get("sample", ""),
    height=220
)

col1, col2 = st.columns([1, 1])

with col1:
    predict_btn = st.button(
        "🔍 Analyze Sentiment",
        use_container_width=True
    )

with col2:
    clear_btn = st.button(
        "🗑 Clear",
        use_container_width=True
    )

if clear_btn:
    st.rerun()

# -------------------------
# Prediction
# -------------------------
if predict_btn:

    if review.strip() == "":
        st.warning("Please enter a review.")
        st.stop()

    with st.spinner("Analyzing review..."):

        score = predict_sentiment(review)

        positive_prob = float(score)
        negative_prob = 1 - positive_prob

    st.divider()

    if score >= 0.5:

        st.markdown(
            """
            <div class='result-box positive'>
            😊 Positive Review
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class='result-box negative'>
            😞 Negative Review
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("### Confidence Score")

    confidence = max(
        positive_prob,
        negative_prob
    )

    st.progress(confidence)

    st.metric(
        "Model Confidence",
        f"{confidence*100:.2f}%"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Positive Probability",
            f"{positive_prob*100:.2f}%"
        )

    with col2:
        st.metric(
            "Negative Probability",
            f"{negative_prob*100:.2f}%"
        )

# -------------------------
# Footer
# -------------------------
st.markdown(
    """
    <div class='footer'>
    Built with ❤️ using Streamlit & TensorFlow
    </div>
    """,
    unsafe_allow_html=True
)
