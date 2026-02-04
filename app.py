import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ------------------------------
# Load saved models
# ------------------------------
@st.cache_resource
def load_models():
    with open("count_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("tfidf_transformer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    return vectorizer, tfidf, feature_names


vectorizer, tfidf, feature_names = load_models()

# ------------------------------
# Keyword extraction function
# ------------------------------
def extract_keywords(text, top_n=10):
    vec = vectorizer.transform([text])
    tfidf_vec = tfidf.transform(vec)

    scores = tfidf_vec.toarray()[0]
    indices = np.argsort(scores)[::-1][:top_n]

    keywords = [feature_names[i] for i in indices if scores[i] > 0]
    return keywords


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Keyword Extraction App", layout="centered")

st.title(" Keyword Extraction App")
st.write("Enter text below and extract the most important keywords.")

user_input = st.text_area(
    "Enter your text:",
    height=200,
    placeholder="Paste your article, abstract, or document here..."
)

top_n = st.slider("Number of keywords", min_value=5, max_value=30, value=10)

if st.button("Extract Keywords"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        keywords = extract_keywords(user_input, top_n)

        if keywords:
            st.success("Keywords extracted successfully!")
            st.write("###  Extracted Keywords:")

            for kw in keywords:
                st.write(f"• {kw}")

        else:
            st.info("No keywords found.")

