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

# **Project Workflow:**

## 🔎 Data Preprocessing

Text Cleaning: Lowercasing, punctuation removal, and stopword filtering.

Feature Engineering: Tokenization and TF-IDF vectorization.

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(df['cleaned_tweet'])

## 🧠 Modeling Approach

The following models were trained and evaluated:

Logistic Regression
Support Vector Machine (SVM)
XGBoost

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

## Model Evaluation

Metrics Used: Accuracy, Precision, Recall, F1-score

SHAP Analysis for feature importance and interpretability.

from sklearn.metrics import classification_report

y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred))


## Key Insights

SVM outperformed all models with 98% accuracy, offering the best balance.

Logistic Regression (96%) was valuable for interpretability.

XGBoost (92%) had a slight trade-off in precision and recall.

Misclassification occurred mainly between neutral and negative sentiment classes.

## Conclusion & Future Work
- Conclusion:

SVM is the best model for Twitter sentiment classification.
NLP techniques effectively distinguish consumer sentiment for Apple vs. Google.
- Future Improvements:

Real-time sentiment tracking for live tweets.
Deep learning models (LSTMs, Transformers) for enhanced predictions.
Fine-tuned handling of class imbalances.

## 🚀 How to Run the Project

Install dependencies:

pip install numpy pandas scikit-learn xgboost shap

pip install numpy pandas scikit-learn xgboost shap
Open Jupyter Notebook and run index.ipynb.
Follow the workflow to preprocess data, train models, and evaluate results.

## 📞 Contact

For questions or collaborations, reach out via dafkiarie@gmail.com.