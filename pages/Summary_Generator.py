import streamlit as st
from utils import summarize_text, read_document

st.title("📄 Summary Generator")

tab1, tab2 = st.tabs(["Text Input", "Document Upload"])

with tab1:
    text = st.text_area("Enter text", key="sum_text")
    model = st.selectbox("Choose summarizer", ["Pegasus", "BART", "Sumy"])

    if st.button("Summarize", key="sum_btn"):
        st.success(summarize_text(text, model))


with tab2:
    file = st.file_uploader("Upload document", type=["pdf", "docx", "txt"])
    model = st.selectbox("Choose summarizer", ["Pegasus", "BART", "Sumy"], key="sum_model_doc")

    if file and st.button("Summarize Document", key="sum_doc_btn"):
        text = read_document(file)
        st.success(summarize_text(text, model))