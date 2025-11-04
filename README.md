# 🧠 Multi-Project NLP Suite

This repository contains three natural language processing (NLP) projects focused on **text mining**, **sentiment analysis**, and **fake news detection**. Each project explores a different aspect of text analytics — from TF-IDF-based query engines to machine learning-powered classification models.

---

## 📂 Projects Overview

### 1. **TF-IDF Powered Sentiment Query Engine for Product Reviews**

#### 🔍 Description
A query engine that allows users to **search and rank product reviews** based on sentiment and textual relevance using **TF-IDF (Term Frequency–Inverse Document Frequency)**.  
It helps users quickly find reviews that best match a given query while also factoring in sentiment polarity.

#### 🚀 Features
- Implements **TF-IDF vectorization** for relevance scoring  
- Integrates **sentiment polarity** (positive/negative/neutral) into query ranking  
- Supports **custom search queries** across product reviews  
- Built using **Python**, **scikit-learn**, and **NLTK**

#### 🧩 Key Components
- `vectorizer.py` – TF-IDF model implementation  
- `sentiment_model.py` – Sentiment scoring engine  
- `query_engine.py` – Query processor combining TF-IDF and sentiment scores  

---

### 2. **Sentiment Analysis on Product Reviews**

#### 🔍 Description
A supervised machine learning model that classifies product reviews as **positive, negative, or neutral**.  
This project explores multiple NLP preprocessing steps and ML algorithms for text classification.

#### 🚀 Features
- Extensive **text preprocessing** (stopword removal, stemming, tokenization)  
- Multiple ML models: **Logistic Regression**, **Naive Bayes**, **SVM**  
- Visualizations for sentiment distribution and model performance  
- Dataset: Custom or open-source (e.g., Amazon Reviews)

#### 🧩 Key Components
- `preprocessing.py` – Text cleaning and tokenization pipeline  
- `model_training.ipynb` – Model training and evaluation  
- `sentiment_visuals.py` – Performance and sentiment distribution plots  

---

### 3. **Fake News Classifier**

#### 🔍 Description
A machine learning classifier that detects **fake vs. real news articles** using NLP techniques and text vectorization.  
It applies a similar preprocessing pipeline but focuses on identifying deceptive writing patterns.

#### 🚀 Features
- Uses **TF-IDF** or **word embeddings** for feature extraction  
- Trained on **Fake News Dataset** (Kaggle or custom source)  
- Evaluates models using **confusion matrix**, **precision**, **recall**, and **F1-score**  
- Modular code for retraining on new datasets

#### 🧩 Key Components
- `data_cleaning.py` – Data preprocessing and normalization  
- `train_model.py` – Model training and evaluation  
- `predict.py` – Real-time prediction script  

---

## 🛠️ Installation & Setup

### Prerequisites
Make sure you have the following installed:
- Python 3.8+
- pip (Python package manager)

### Clone the Repository
```bash
git clone https://github.com/your-username/nlp-multi-project-suite.git
cd nlp-multi-project-suite
