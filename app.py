import streamlit as st
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
import torch.nn.functional as F
import requests
from streamlit_lottie import st_lottie
import os 
import gdown


MODEL_PATH = "bert_sentiment_epoch1.pth"

if not os.path.exists(MODEL_PATH):
    file_id = "1kGY_X_jSHeI1DJvZphenAjD_0eieJgL-"
    url = f"https://drive.google.com/uc?id={file_id}"
    print("Downloading model from Google Drive...")
    gdown.download(url, MODEL_PATH, quiet=False)

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Cache Model Loading
@st.cache_resource
def load_model():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.load_state_dict(torch.load("bert_sentiment_epoch1.pth", map_location=device))
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()


# Load Lottie Animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        st.error("Failed to load Lottie animation.")
        return None
    return r.json()

lottie_success = load_lottieurl("https://lottie.host/f0d45ec5-2d16-423a-8e0d-37a3025bcc73/tXL5sK1oOt.json")
lottie_loader = load_lottieurl("https://lottie.host/f5d86798-1208-4678-8ab8-2f46bb1bca95/jhqsdyd8dq.json")


# Streamlit Config
st.set_page_config(
    page_title="Movie Mood Analyzer",
    page_icon="🎥",
    layout="centered",
)


# Custom CSS for a clean, dark theme
st.markdown("""
    <style>
    textarea {
        background-color: #1e1e1e !important;   
        color: #ffffff !important;              
        border: 1px solid #ffffff !important;   
        border-radius: 8px !important;          
    }     
     
    textarea:focus {
        border: 1px solid #00ffff !important;
        outline: none !important;        

    button {
        background-color: #0A9396 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px;
    }

    button:hover {
        background-color: #005F73 !important;
    }

    h1, h2 {
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)


# App Header
col1, col2 = st.columns([1, 6])

with col1:
    st_lottie(lottie_loader, height=80, key="loader")

with col2:
    st.markdown("<h1 style='margin-bottom:0;'>Movie Mood Analyzer</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:18px; margin-top:0;'>Paste a movie review below to find out its sentiment.</p>",
        unsafe_allow_html=True
    )



# User Input
user_input = st.text_area(
    "Your Review:",
    height=200,
    placeholder="E.g., This movie was absolutely amazing!"
)


# Predict Button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        with st.spinner('Analyzing...'):
            inputs = tokenizer(
                user_input, return_tensors="pt",
                truncation=True, padding=True, max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                predicted = torch.argmax(probs).item()
                confidence = probs[0][predicted].item()

        label = "Positive " if predicted == 1 else "Negative 😠"
        st.subheader(f"Result: {label}")
        st.write(f"Confidence: {confidence:.2%}")

        if predicted == 1 and lottie_success:
            st_lottie(lottie_success, height=200, key="success")


# Footer
st.markdown(
    "<p style='font-size:12px; color:gray;'>© 2025 Movie Mood Analyzer | Built with Streamlit</p>",
    unsafe_allow_html=True
)
