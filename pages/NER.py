import streamlit as st
from utils import ner_analysis, read_document, load_spacy

st.title("🏷 Named Entity Recognition")

@st.cache_resource
def get_nlp():
    return load_spacy()

nlp = get_nlp()

tab1, tab2 = st.tabs(["Text Input", "Document Upload"])

with tab1:
    text = st.text_area("Enter text", key="ner_text")

    if st.button("Analyze", key="ner_btn") and text.strip():
        entities = ner_analysis(text, nlp)
        st.json(entities)

with tab2:
    file = st.file_uploader(
        "Upload document",
        type=["pdf", "docx", "txt"]
    )

    if file and st.button("Analyze Document", key="ner_doc_btn"):
        text = read_document(file)

        if text.strip():
            entities = ner_analysis(text, nlp)
            st.json(entities)
        else:
            st.warning("No readable text found in the document.")

