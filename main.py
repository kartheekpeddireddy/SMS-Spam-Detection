import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ----------------------------------------
# LOAD DATA
# ----------------------------------------
def load_data(path):
    df = pd.read_csv(path, sep="\t", names=["label", "message"], encoding="utf-8")
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

# ----------------------------------------
# TF-IDF FEATURES
# ----------------------------------------
def get_tfidf_features(X_train):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_tf = vectorizer.fit_transform(X_train)
    return X_train_tf, vectorizer

# ----------------------------------------
# TRAIN MODELS
# ----------------------------------------
def train_models(X_train_tf, y_train):
    models = {}

    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train_tf, y_train)
    models["Logistic Regression"] = lr

    svm = LinearSVC()
    svm.fit(X_train_tf, y_train)
    models["SVM"] = svm

    return models

# ----------------------------------------
# EVALUATE MODELS
# ----------------------------------------
def evaluate_model(model, X_test_tf, y_test):
    y_pred = model.predict(X_test_tf)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm

# ----------------------------------------
# MAIN EXECUTION
# ----------------------------------------
def main():
    print("\nLoading dataset...")
    df = load_data("data/sms_spam.tsv")
    print("Dataset loaded:", df.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=0.2, random_state=42
    )

    print("\nExtracting TF-IDF features...")
    X_train_tf, vectorizer = get_tfidf_features(X_train)
    X_test_tf = vectorizer.transform(X_test)

    print("\nTraining models...")
    models = train_models(X_train_tf, y_train)

    print("\nEvaluating models...")
    for name, model in models.items():
        metrics, cm = evaluate_model(model, X_test_tf, y_test)

        print("\n==============================")
        print("MODEL:", name)
        print("==============================")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    main()
