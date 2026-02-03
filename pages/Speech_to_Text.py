import streamlit as st
from utils import mic_to_text
import whisper
import tempfile

@st.cache_resource
def load_model():
    return whisper.load_model("base")
model = load_model()
st.title("🎤 Speech to Text")

tab1, tab2 = st.tabs(["Upload Audio", "Use Microphone"])

with tab1:
    audio_file = st.file_uploader("Upload audio", type="wav")
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(audio_file.read())
            with st.spinner("Transcribing audio..."):
                result = model.transcribe(tmp.name)
        st.text_area("Transcription", result["text"])

with tab2:
    if st.button("Start Speaking", key="stt_mic"):
        st.success(mic_to_text())


