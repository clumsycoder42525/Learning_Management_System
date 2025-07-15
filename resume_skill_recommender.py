import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ResumeSkillRecommender:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.df.fillna("", inplace=True)

        # Lowercase conversion
        self.df['skills'] = self.df['skills'].str.lower()
        self.df['related_skils_in_job'] = self.df['related_skils_in_job'].str.lower()
        self.df['certification_skills'] = self.df['certification_skills'].str.lower()
        self.df['skills_required'] = self.df['skills_required'].str.lower()

        # Combined skill field
        self.df['all_skills'] = (
            self.df['skills'] + ' ' +
            self.df['related_skils_in_job'] + ' ' +
            self.df['certification_skills']
        )

        # ✅ Debug: Check columns available
        print("🧾 Columns available in dataset:", self.df.columns.tolist())

    def recommend_missing_skills(self, resume_index):
        known_skills = set(self.df.iloc[resume_index]["all_skills"].split())
        required_skills = set(self.df.iloc[resume_index]["skills_required"].split())
        missing_skills = required_skills - known_skills
        return list(missing_skills)

    def find_similar_resumes(self, resume_index, top_n=3):
        vec = CountVectorizer()
        skill_matrix = vec.fit_transform(self.df['all_skills'])

        similarity = cosine_similarity(skill_matrix[resume_index], skill_matrix)
        top_indices = similarity[0].argsort()[-top_n-1:][::-1]
        top_indices = [i for i in top_indices if i != resume_index][:top_n]

        # ✅ Handle missing columns gracefully
        possible_cols = ['job_position_name', 'skills_required', 'all_skills', 'matched_score']
        existing_cols = [col for col in possible_cols if col in self.df.columns]

        return self.df.iloc[top_indices][existing_cols]
