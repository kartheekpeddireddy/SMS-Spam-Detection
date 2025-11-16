# 📨 Project 05: SMS Spam Detection (TF-IDF + Logistic Regression / SVM)

## 📘 Overview
This project builds a **machine learning model to classify SMS messages** as **Spam** or **Ham (Not Spam)** using **Natural Language Processing (NLP)**.  
It uses **TF-IDF vectorization** for text feature extraction and compares the performance of **Logistic Regression** and **Support Vector Machine (SVM)** classifiers.

---

## 🎯 Objectives
- Preprocess and clean the SMS Spam dataset.
- Convert text data into numerical form using **TF-IDF**.
- Train and evaluate two models:
  - Logistic Regression
  - Support Vector Machine (LinearSVC)
- Compare their performance based on accuracy, precision, recall, and F1-score.
- Visualize confusion matrices for both models.

---

## 📊 Dataset
**Source:** [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)  
**File:** `data/sms_spam.csv`

| Column | Description |
|---------|--------------|
| `label` | Message type (`ham` or `spam`) |
| `message` | The text content of the SMS message |

---

## 🧠 Methodology
1. **Data Loading & Cleaning**
   - Load dataset, remove duplicates, clean special characters, lowercase conversion.
2. **Exploratory Data Analysis (EDA)**
   - Analyze class distribution and message length patterns.
3. **Feature Engineering**
   - Convert messages to **TF-IDF features** (max 3000 features).
4. **Model Training**
   - Train **Logistic Regression** and **SVM** models.
5. **Model Evaluation**
   - Evaluate using accuracy, precision, recall, F1-score, and confusion matrix.
6. **Comparison**
   - Compare model results and visualize.

---


