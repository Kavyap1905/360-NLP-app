import streamlit as st
from utils import text_to_speech, read_document

st.title("🔊 Text to Speech")

tab1, tab2 = st.tabs(["Text Input", "Document Upload"])

with tab1:
    text = st.text_area("Enter text", key="tts_text")
    if st.button("Convert", key="tts_btn"):
        text_to_speech(text)
        st.success("Audio generated")

with tab2:
    file = st.file_uploader("Upload document", type=["pdf", "docx", "txt"])
    if file and st.button("Convert Document", key="tts_doc_btn"):
        text = read_document(file)
        text_to_speech(text)
        st.success("Audio generated")
