from transformers import pipeline
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from deep_translator import GoogleTranslator
import io
import pdfplumber
import docx
from langdetect import detect
import speech_recognition as sr
import pyttsx3
import spacy
from textblob import TextBlob
import cv2
import nltk
from fer.fer import FER

nltk.download('punkt')
nltk.download('punkt_tab')

LANGUAGES = LANGUAGES = {
    "Afrikaans": "af",
    "Albanian": "sq",
    "Arabic": "ar",
    "Bengali": "bn",
    "Bulgarian": "bg",
    "Catalan": "ca",
    "Chinese (Simplified)": "zh-CN",
    "Chinese (Traditional)": "zh-TW",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr",
    "German": "de",
    "Greek": "el",
    "Gujarati": "gu",
    "Hebrew": "iw",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Kannada": "kn",
    "Korean": "ko",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Norwegian": "no",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Punjabi": "pa",
    "Romanian": "ro",
    "Russian": "ru",
    "Serbian": "sr",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Spanish": "es",
    "Swahili": "sw",
    "Swedish": "sv",
    "Tamil": "ta",
    "Telugu": "te",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Vietnamese": "vi"
}

# Load models
bart = pipeline("summarization", model="facebook/bart-large-cnn")
nlp = spacy.load("en_core_web_sm")

MODEL_NAME = "google/pegasus-xsum"

tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME)
model.eval()

def pegasus_summary(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="longest",
        return_tensors="pt"
    )

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=60,
            min_length=20,
            num_beams=4
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_text(text, model):
    if model == "Pegasus":
        return pegasus_summary(text)
    
    elif model == "BART":
        return bart(text, max_length=60)[0]['summary_text']
    else:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        return " ".join(str(s) for s in summarizer(parser.document, 2))

def translate_text(text, src, tgt):
    if not src:
        src = detect(text)

    translated = GoogleTranslator(
        source=src,
        target=tgt
    ).translate(text)

    return translated

def speech_to_text(audio):
    r = sr.Recognizer()
    with sr.AudioFile(audio) as source:
        audio = r.record(source)
    return r.recognize_google(audio)

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def ner_analysis(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def text_sentiment(text):
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0:
        return "Positive"
    elif blob.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def speech_sentiment():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)

        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        if polarity > 0:
            return f"Speech Text: {text}\nSentiment: Positive"
        elif polarity < 0:
            return f"Speech Text: {text}\nSentiment: Negative"
        else:
            return f"Speech Text: {text}\nSentiment: Neutral"

    except Exception as e:
        return "Could not understand audio"

def camera_emotion_detection():
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect_emotions(frame)

        for face in result:
            (x, y, w, h) = face["box"]
            emotions = face["emotions"]
            dominant_emotion = max(emotions, key=emotions.get)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(
                frame,
                dominant_emotion,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0,255,0),
                2
            )

        cv2.imshow("Live Emotion Detection (Press Q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def read_document(uploaded_file):

    if uploaded_file is None:
        return ""

    file_type = uploaded_file.name.split(".")[-1].lower()

    # TXT files
    if file_type == "txt":
        return uploaded_file.read().decode("utf-8", errors="ignore")

    # PDF files
    elif file_type == "pdf":
        text = ""
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    # DOCX files
    elif file_type == "docx":
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])

    else:
        raise ValueError("Unsupported file type. Please upload TXT, PDF, or DOCX.")

def mic_to_text():

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)

        try:
            audio = recognizer.listen(
                source,
                timeout=10,
                phrase_time_limit=None
            )
        except sr.WaitTimeoutError:
            return ""

    try:
        text = recognizer.recognize_google(audio)
        return text

    except sr.UnknownValueError:
        return ""

    except sr.RequestError:
        return ""