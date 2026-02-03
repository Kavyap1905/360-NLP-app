import streamlit as st
from utils import ner_analysis, read_document

st.title("🏷 Named Entity Recognition")

tab1, tab2 = st.tabs(["Text Input", "Document Upload"])

with tab1:
    text = st.text_area("Enter text", key="ner_text")
    if st.button("Analyze", key="ner_btn"):
        st.json(ner_analysis(text))

with tab2:
    file = st.file_uploader("Upload document", type=["pdf", "docx", "txt"])
    if file and st.button("Analyze Document", key="ner_doc_btn"):
        text = read_document(file)
        st.json(ner_analysis(text))
