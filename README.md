# Twitter-Sentiment-Analysis-Apple-vs.-Google

## 📌 Project Overview

This project aims to analyze Twitter sentiment about Apple and Google products using Natural Language Processing (NLP). The goal is to build a machine learning model that classifies tweets as positive, negative, or neutral, helping businesses understand public perception.

## 🎯 Business Problem

Tech companies like Apple and Google rely on customer feedback to improve products and marketing strategies. An automated sentiment analysis system can help them track public sentiment in real-time, identifying trends and potential concerns.

## 📊 Dataset

Source: CrowdFlower via Data.world

Size: 9,093 tweets

Labels: Positive, Negative, Neutral

Features:

tweet_text: The actual tweet content

emotion_in_tweet_is_directed_at: The brand or product mentioned

is_there_an_emotion_directed_at_a_brand_or_product: Whether the tweet expresses sentiment

Data File: judge_1377884607_tweet_product_company.csv (located in /data/ folder)

## 🔎 Data Preprocessing

Cleaning: Removing special characters, URLs, mentions, and hashtags

Tokenization: Splitting text into words

Stopword Removal: Eliminating common words (e.g., "the", "is")

Stemming/Lemmatization: Reducing words to their root form

Vectorization: Converting text into numerical format (TF-IDF, Word2Vec, or Transformer embeddings)

## 🧠 Modeling Approach

Baseline Models: Logistic Regression, Naive Bayes

Advanced Models: LSTM, BERT (transformer-based model)

Evaluation Metrics: Accuracy, Precision, Recall, F1-score

Validation Strategy: Train-test split, cross-validation

## 📌 Project Structure

📂 nlp-twitter-sentiment-analysis  
│── 📂 data/                # Raw & processed datasets  
│── 📂 notebooks/           # Jupyter Notebooks  
│── 📂 src/                 # Python scripts for data processing & modeling  
│── 📂 reports/             # Presentation slides & summary  
│── README.md               # Project overview & instructions  
│── requirements.txt        # Dependencies  
│── .gitignore              # Ignore unnecessary files  

## 🚀 How to Run the Project

Clone the Repository:

git clone https://github.com/yourusername/nlp-twitter-sentiment-analysis.git
cd nlp-twitter-sentiment-analysis

Install Dependencies:

pip install -r requirements.txt

Run Jupyter Notebook:

jupyter notebook

## 📌 Dependencies

Python 3.x

pandas, numpy, matplotlib, seaborn

scikit-learn, nltk, spaCy

transformers (for BERT-based models)

## 📜 License

MIT License

## 📞 Contact

For questions or collaborations, reach out via dafkiarie@gmail.com.