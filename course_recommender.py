
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class CourseRecommender:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.df.fillna("", inplace=True)

        # Convert text to lowercase
        self.df['Title'] = self.df['Title'].str.lower()
        self.df['Gained Skills'] = self.df['Gained Skills'].str.lower()

        # Combine text for vectorization
        self.df['combined'] = self.df['Title'] + ' ' + self.df['Gained Skills']

    def recommend(self, user_skills, top_n=5):
        # Convert user input list to one lowercase string
        user_input = ' '.join(user_skills).lower()

        # Combine with dataset to vectorize together
        combined = self.df['combined'].tolist() + [user_input]

        # Vectorization
        vec = CountVectorizer()
        vectors = vec.fit_transform(combined)

        # Cosine similarity (user vector vs course vectors)
        sim = cosine_similarity(vectors[-1], vectors[:-1])

        # Get top matching indices
        top_indices = sim[0].argsort()[-top_n:][::-1]

        # Return the matching rows
        return self.df.iloc[top_indices][['Title', 'Institution', 'Gained Skills', 'Level', 'Duration', 'Rate', 'Reviews']]
