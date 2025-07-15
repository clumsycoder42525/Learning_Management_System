import pandas as pd
import numpy as np
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

class ResumeCategoryPredictor:
    def __init__(self, csv_path):
        # Load dataset
        self.df = pd.read_csv(csv_path)
        self.df.dropna(subset=["Resume", "Category"], inplace=True)

        # Clean resumes
        self.df["Resume"] = self.df["Resume"].apply(self.clean_resume)

        # Encode category labels
        self.label_encoder = LabelEncoder()
        self.df["Category_Encoded"] = self.label_encoder.fit_transform(self.df["Category"])

        # Vectorize
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.X = self.vectorizer.fit_transform(self.df["Resume"])
        self.y = self.df["Category_Encoded"]

        # Train classifier
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(self.X, self.y)

        # Save models
        joblib.dump(self.label_encoder, "models/label_encoder.pkl")
        joblib.dump(self.vectorizer, "models/vectorizer.pkl")
        joblib.dump(self.model, "models/knn_model.pkl")

    def clean_resume(self, text):
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"RT|cc", "", text)
        text = re.sub(r"@\S+", "", text)
        text = re.sub(r"#\S+", "", text)
        text = re.sub(r"[^A-Za-z\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip().lower()

    def predict_category(self, resume_text):
        cleaned = self.clean_resume(resume_text)
        vector = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(vector)[0]
        category = self.label_encoder.inverse_transform([prediction])[0]
        return category

    def rate_resume(self, resume_text):
        cleaned = self.clean_resume(resume_text)
        word_count = len(cleaned.split())

        # Define good keywords to check for strong resumes
        keywords = ['project', 'experience', 'skills', 'education', 'internship', 'certificate', 'achievement']
        keyword_hits = sum([1 for kw in keywords if kw in cleaned])

        # Scoring Logic
        score = 0

        if word_count > 100:
            score += 2
        elif word_count > 50:
            score += 1

        if keyword_hits >= 5:
            score += 3
        elif keyword_hits >= 3:
            score += 2
        elif keyword_hits >= 1:
            score += 1

        # Convert score to 1-5 stars
        rating = min(5, score)
        return rating
