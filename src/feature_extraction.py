from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_features(texts, max_features=3000):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X = vectorizer.fit_transform(texts)
    return X, vectorizer
