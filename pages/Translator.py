import streamlit as st
from utils import LANGUAGES, translate_text, read_document

st.title("🌍 Translator")


from_lang_name = st.selectbox("From Language", list(LANGUAGES.keys()))
to_lang_name = st.selectbox("To Language", list(LANGUAGES.keys()))

# Convert names → codes
from_lang_code = LANGUAGES[from_lang_name]
to_lang_code = LANGUAGES[to_lang_name]

tab1, tab2 = st.tabs(["Text Input", "Document Upload"])

with tab1:
    text = st.text_area("Enter text", key="translate_text")
    if st.button("Translate", key="translate_btn"):
        result = translate_text(text, from_lang_code, to_lang_code)
        st.success(result)

with tab2:
    file = st.file_uploader("Upload document", type=["pdf", "docx", "txt"])
    if file and st.button("Translate Document", key="translate_doc_btn"):
        text = read_document(file)
        result = translate_text(text, from_lang_code, to_lang_code)
        st.success(result)
