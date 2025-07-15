from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class InternshipRecommender:

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.df.fillna('', inplace=True)

        self.df["internship_title"] = self.df["internship_title"].str.lower()
        self.df["company_name"] = self.df["company_name"].str.lower()
        self.df["location"] = self.df["location"].str.lower()

        self.df['skills'] = (
            self.df['internship_title'] + ' ' +
            self.df['company_name'] + ' ' +
            self.df['location']
        )

        self.warning_message = None

    def recommend(self, user_skills, top_n=5):
        user_input = ' '.join(user_skills).lower()
        combined = self.df["skills"].tolist() + [user_input]

        # Vectorize
        vc = CountVectorizer()
        vectors = vc.fit_transform(combined)

        # Cosine similarity
        sim = cosine_similarity(vectors[-1], vectors[:-1])

        # Get top N matches
        top_indices = sim[0].argsort()[-top_n:][::-1]

        return self.df.iloc[top_indices][['internship_title', 'company_name', 'location', 'stipend', 'skills']]

