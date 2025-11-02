import streamlit as st
from sentence_transformers import SentenceTransformer, util

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

st.title("🧠 AI-Powered Plagiarism Checker")
st.write("Check if two texts are similar using AI-based semantic comparison.")

text1 = st.text_area("✏️ Enter First Text (e.g., Student's Work):", height=150)
text2 = st.text_area("📘 Enter Second Text (e.g., Reference Material):", height=150)

if st.button("Check Similarity"):
    if text1 and text2:
        emb1 = model.encode(text1, convert_to_tensor=True)
        emb2 = model.encode(text2, convert_to_tensor=True)

        similarity = util.pytorch_cos_sim(emb1, emb2)
        score = float(similarity[0][0]) * 100

        st.markdown(f"### 🔍 Similarity Score: **{score:.2f}%**")

        if score > 75:
            st.error("⚠️ Possible Plagiarism Detected!")
        elif score > 40:
            st.warning("🟠 Partial Similarity Detected – Check for Rephrasing.")
        else:
            st.success("✅ Content Seems Original.")
    else:
        st.warning("Please enter text in both boxes.")
