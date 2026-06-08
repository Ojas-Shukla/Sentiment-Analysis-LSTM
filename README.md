# 🎬 IMDb Sentiment Analysis using LSTM

A Deep Learning-based Sentiment Analysis application that predicts whether a movie review is **Positive** or **Negative** using an LSTM (Long Short-Term Memory) neural network trained on the IMDb 50K Movie Reviews dataset.

## 🚀 Live Demo

🔗 https://sentiment-analysis-lstm-p6opvqwd68r8li5zuiappmb.streamlit.app/

---

## 📌 Features

- Predict sentiment of movie reviews
- Deep Learning model using LSTM
- Text preprocessing pipeline
- Interactive Streamlit UI
- Confidence score visualization
- Real-time inference

---

## 🛠️ Tech Stack

### Machine Learning & NLP
- Python
- TensorFlow / Keras
- NLTK
- NumPy
- Pandas
- Scikit-learn

### Frontend
- Streamlit

### Model Architecture
- Embedding Layer
- LSTM Layer
- Dense Layers
- Sigmoid Output Layer

---

## 📂 Dataset

Dataset used:

📊 IMDb Dataset of 50K Movie Reviews

- 50,000 movie reviews
- Binary sentiment labels
- Positive / Negative classification

Source:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---

## 🔄 Data Preprocessing

The following preprocessing steps were applied:

- Convert text to lowercase
- Remove HTML tags
- Remove URLs
- Remove punctuation
- Remove stopwords
- Lemmatization
- Tokenization
- Sequence Padding

## 🧠 Model Architecture

```text
Input Review
      ↓
Tokenizer
      ↓
Padding (maxlen = 200)
      ↓
Embedding Layer
      ↓
LSTM (128 Units)
      ↓
Dense Layer
      ↓
Dropout
      ↓
Output Layer (Sigmoid)
      ↓
Positive / Negative
```


## 📊 Results

**Accuracy | ~85-90%**


## 📁 Project Structure

```text
Sentiment-Analysis-LSTM/
│
├── app.py
├── sentiment_lstm.keras
├── tokenizer.pkl
├── requirements.txt
├── README.md
├── Sentiment.ipynb
```


## 📸 Application Preview

<img width="1919" height="932" alt="Screenshot 2026-06-04 201207" src="https://github.com/user-attachments/assets/2342a7d1-88ae-4456-ba7b-613d2301187a" />

<img width="1919" height="934" alt="Screenshot 2026-06-04 201236" src="https://github.com/user-attachments/assets/733dd796-8096-492c-9e79-83b95346b3be" />

---

## 🎯 Future Improvements

- Bidirectional LSTM
- GRU Comparison
- Attention Mechanism
- BERT Integration
- Explainable AI (SHAP/LIME)
- Docker Deployment
- REST API using FastAPI
- React Frontend
