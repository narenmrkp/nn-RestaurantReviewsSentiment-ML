import streamlit as st
import joblib
import re
import nltk
import numpy as np
import os

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -----------------------
# Safe NLTK Setup
# -----------------------
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Restaurant Sentiment Analyzer",
    page_icon="🍽️",
    layout="centered"
)

# -----------------------
# Custom CSS
# -----------------------
st.markdown("""
<style>
.big-title {
    font-size:40px !important;
    font-weight: bold;
    text-align:center;
    color: #ff4b4b;
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
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🍽 Restaurant Review Sentiment Analyzer</p>', unsafe_allow_html=True)

# -----------------------
# Load Model
# -----------------------
@st.cache_resource
def load_model():
    model = joblib.load("model/Restaurant_review_model.pkl")
    cv = joblib.load("model/count_v_res.pkl")
    return model, cv

model, cv = load_model()

# -----------------------
# Preprocessing
# -----------------------
ps = PorterStemmer()

custom_stopwords = {
    'don',"don't",'ain','aren',"aren't",
    'no','nor','not'
}

stop_words = set(stopwords.words("english")) - custom_stopwords

def preprocess(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    return " ".join(review)

# -----------------------
# UI
# -----------------------
review_input = st.text_area("✍️ Enter Restaurant Review:", height=150)

if st.button("🔍 Analyze Sentiment"):

    if review_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        processed = preprocess(review_input)
        vector = cv.transform([processed]).toarray()

        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        confidence = np.max(probability) * 100

        if prediction == 1:
            st.markdown(
                f'<div class="result-box positive">✅ Positive Review<br>Confidence: {confidence:.2f}%</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box negative">❌ Negative Review<br>Confidence: {confidence:.2f}%</div>',
                unsafe_allow_html=True
            )

st.markdown("---")
st.caption("Built with Streamlit + Scikit-Learn")
