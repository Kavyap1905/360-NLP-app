import streamlit as st
from utils import (
    summarize_text,
    read_document,
    load_bart,
    load_pegasus
)

st.title("📄 Summary Generator")

# -----------------------
# Load models safely
# -----------------------

@st.cache_resource
def get_bart():
    return load_bart()

@st.cache_resource
def get_pegasus():
    return load_pegasus()

bart = get_bart()
pegasus = get_pegasus()

tab1, tab2 = st.tabs(["Text Input", "Document Upload"])

# -------- Text Input --------
with tab1:
    text = st.text_area("Enter text", key="sum_text")
    model_choice = st.selectbox(
        "Choose summarizer",
        ["Pegasus", "BART"],
        key="sum_model_text"
    )

    if st.button("Summarize", key="sum_btn") and text.strip():
        with st.spinner("Generating summary..."):
            summary = summarize_text(
                text=text,
                model_type=model_choice,
                bart=bart,
                pegasus=pegasus
            )
        st.success(summary)

# -------- Document Upload --------
with tab2:
    file = st.file_uploader(
        "Upload document",
        type=["pdf", "docx", "txt"]
    )

    model_choice = st.selectbox(
        "Choose summarizer",
        ["Pegasus", "BART"],
        key="sum_model_doc"
    )

    if file and st.button("Summarize Document", key="sum_doc_btn"):
        text = read_document(file)

        if text.strip():
            with st.spinner("Generating summary..."):
                summary = summarize_text(
                    text=text,
                    model_type=model_choice,
                    bart=bart,
                    pegasus=pegasus
                )
            st.success(summary)
        else:
            st.warning("No readable text found in the document.")
