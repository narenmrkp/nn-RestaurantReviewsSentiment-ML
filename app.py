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

        negative_prob = probability[0] * 100
        positive_prob = probability[1] * 100
        confidence = max(negative_prob, positive_prob)

        # -----------------------------
        # 🎯 Result Display
        # -----------------------------
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

        # -----------------------------
        # 🥧 Donut Pie Chart
        # -----------------------------
        import matplotlib.pyplot as plt

        st.write("### 🥧 Sentiment Probability Distribution")

        labels = ["Negative", "Positive"]
        sizes = [negative_prob, positive_prob]
        colors = ["#ff6b6b", "#4CAF50"]

        fig, ax = plt.subplots()

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops=dict(width=0.45, edgecolor='white'),  # Donut effect
            textprops=dict(color="black", fontsize=12, weight="bold")
        )

        ax.axis('equal')  # Keep circle shape

        st.pyplot(fig)

        st.markdown("---")

        # -----------------------------
        # 📈 Review Analytics
        # -----------------------------
        st.write("### 📈 Review Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Character Count", len(review_input))

        with col2:
            st.metric("Word Count", len(review_input.split()))












# if st.button("🔍 Analyze Sentiment"):

#     if review_input.strip() == "":
#         st.warning("Please enter a review.")
#     else:
#         processed = preprocess(review_input)
#         vector = cv.transform([processed]).toarray()

#         prediction = model.predict(vector)[0]
#         probability = model.predict_proba(vector)[0]

#         confidence = np.max(probability) * 100

#         if prediction == 1:
#             st.markdown(
#                 f'<div class="result-box positive">✅ Positive Review<br>Confidence: {confidence:.2f}%</div>',
#                 unsafe_allow_html=True
#             )
#         else:
#             st.markdown(
#                 f'<div class="result-box negative">❌ Negative Review<br>Confidence: {confidence:.2f}%</div>',
#                 unsafe_allow_html=True
#             )
# import pandas as pd

# prob_df = pd.DataFrame({
#     "Sentiment": ["Negative", "Positive"],
#     "Probability": probability
# })

# st.bar_chart(prob_df.set_index("Sentiment"))

# st.markdown("---")
# st.caption("Built with Streamlit + Scikit-Learn")
