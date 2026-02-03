import streamlit as st
from utils import text_sentiment, speech_sentiment, camera_emotion_detection

st.title("😊 Sentiment Analysis")

tab1, tab2, tab3 = st.tabs(["Text", "Speech", "Camera"])

with tab1:
    text = st.text_area("Enter text", key="sent_text")
    if st.button("Analyze", key="sent_text_btn"):
        st.success(text_sentiment(text))

with tab2:
    if st.button("Speak", key="sent_speech_btn"):
        st.success(speech_sentiment())

with tab3:
    st.info("Camera for sentiment recognition")
    if st.button("Start Camera", key="sent_cam_btn"):
        camera_emotion_detection()
