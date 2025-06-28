## Movie Mood Analyzer
AI Sentiment Analysis App for Movie Reviews

A simple, stylish NLP-powered web app that predicts whether a movie review is positive or negative — built with BERT, PyTorch, Streamlit, and a bit of frontend magic.

--- 

## 🚀 Demo
Paste any movie review in the text box — get an instant sentiment prediction with a neat animation!
Try it live: https://movie-mood-analyzer.streamlit.app/

---

## Features
✅ Sentiment analysis using a fine-tuned BERT model
✅ Loads the model weights dynamically (hosted via Google Drive to keep the repo light)
✅ Stylish Streamlit UI with gradient background and Lottie animations
✅ Responsive design — works on desktop & mobile
✅ Fully containerized — easy to deploy on Streamlit Cloud

---

## Tech Stack
Language: Python 3.10+
NLP:  Hugging Face Transformers (BERT)
Backend: PyTorch
Frontend: Streamlit + Lottie animations
Deployment: Streamlit Cloud

---

## Model Weights
The trained model (bert_sentiment_epoch1.pth) is not tracked with Git to keep the repo small.
Instead, it loads dynamically from Google Drive:
```bash
# Example in app.py:
import gdown

url = "https://drive.google.com/uc?id=<your_file_id>"
output = "bert_sentiment_epoch1.pth"
gdown.download(url, output, quiet=False)
```
##  How to Run Locally
1. Clone this repo:
   ```bash
   git clone https://github.com/Wheesk/movie-mood-analyzer.git
   cd movie-mood-analyzer
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Lessons Learned
Fine-tuned BERT with IMDb movie reviews (started with 25k samples, scaled to 5k for speed)
Saved model weights to avoid retraining every time
Loaded model from Google Drive to bypass GitHub’s file size limits
Added Lottie animations for better UX
Deployed on Streamlit Cloud for easy sharing

---

## Future Ideas
Add a database to save user inputs & results
Visualize overall sentiment trends
Expand to multilingual sentiment analysis
Turn into a full REST API with a separate frontend

---

## Author
Made with ❤️ by Wheesk
---

Feel free to fork, star, and share! 
      
