import io
import torch
import spacy
import pdfplumber
import docx

from langdetect import detect
from textblob import TextBlob
from deep_translator import GoogleTranslator
from transformers import pipeline, PegasusTokenizer, PegasusForConditionalGeneration


@torch.inference_mode()
def load_bart():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        framework="pt"
    )

@torch.inference_mode()
def load_pegasus():
    model_name = "google/pegasus-xsum"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

@torch.inference_mode()
def load_spacy():
    return spacy.load("en_core_web_sm")


def pegasus_summary(text, tokenizer, model):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="longest",
        return_tensors="pt"
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=60,
        min_length=20,
        num_beams=4
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_text(text, model_type, bart=None, pegasus=None):
    if model_type == "Pegasus":
        tokenizer, model = pegasus
        return pegasus_summary(text, tokenizer, model)

    elif model_type == "BART":
        return bart(text, max_length=60)[0]["summary_text"]

    else:
        return text

def translate_text(text, src, tgt):
    if not src:
        src = detect(text)

    return GoogleTranslator(source=src, target=tgt).translate(text)

def ner_analysis(text, nlp):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def text_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
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

def read_document(uploaded_file):
    if uploaded_file is None:
        return ""

    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "txt":
        return uploaded_file.read().decode("utf-8", errors="ignore")

    elif file_type == "pdf":
        text = ""
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
        return text

    elif file_type == "docx":
        doc = docx.Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)

    else:
        raise ValueError("Unsupported file type")

