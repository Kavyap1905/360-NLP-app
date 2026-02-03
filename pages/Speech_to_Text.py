import streamlit as st
from utils import speech_to_text, mic_to_text

st.title("🎤 Speech to Text")

tab1, tab2 = st.tabs(["Upload Audio", "Use Microphone"])

with tab1:
    audio = st.file_uploader("Upload audio", type="wav")
    if audio and st.button("Convert", key="stt_file"):
        st.success(speech_to_text(audio))

with tab2:
    if st.button("Start Speaking", key="stt_mic"):
        st.success(mic_to_text())
