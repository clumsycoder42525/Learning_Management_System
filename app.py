# app.py
import streamlit as st
import os
import sys

# === Page Setup ===
st.set_page_config(page_title="Smart LMS", layout="wide")
st.title("🎓 Smart Learning Management System")
st.markdown("Empowering students with AI-driven internship, learning & resume insights.")

# === Imports ===
try:
    from scripts.internship_recommender import InternshipRecommender
    from scripts.course_recommender import CourseRecommender
    from scripts.resume_skill_recommender import ResumeSkillRecommender
    from scripts.resume_classifier import ResumeCategoryPredictor
    import PyPDF2
    st.success("✅ Modules loaded successfully!")
except ModuleNotFoundError as e:
    st.error(f"❌ Import Error: {e}")
    sys.exit(1)

# === Model Initialization ===
try:
    internship_model = InternshipRecommender("data/internship.csv")
    course_model = CourseRecommender("data/Coursera.csv")
    skill_model = ResumeSkillRecommender("data/final_resume_dataset.csv")
    category_model = ResumeCategoryPredictor("data/UpdatedResumeDataSet.csv")
except Exception as e:
    st.error(f"❌ Model Initialization Failed: {e}")
    sys.exit(1)

# === Tabs ===
tab1, tab2, tab3, tab4 = st.tabs([
    "🧑‍💼 Internship Recommender",
    "📘 Course Recommender",
    "📄 Resume Skill Analyzer",
    "📂 Resume Category Predictor"
])

# 🧑‍💼 Internship Recommender
with tab1:
    st.header("🧑‍💼 Internship Recommender")
    skills_input = st.text_input("Enter your skills (comma-separated):", placeholder="e.g. Python, SQL, Excel")
    if skills_input:
        skills = [s.strip() for s in skills_input.split(',') if s.strip()]
        try:
            result = internship_model.recommend(skills)
            if not result.empty:
                st.success("🎯 Recommended Internships:")
                st.dataframe(result[['internship_title', 'company_name', 'skills']])
            else:
                st.warning("No internships found for given skills.")
        except Exception as e:
            st.error(f"Error: {e}")

# 📘 Course Recommender
with tab2:
    st.header("📘 Course Recommender")
    course_input = st.text_input("Which skills do you want to learn?", placeholder="e.g. Data Science, Tableau")
    if course_input:
        skills = [s.strip() for s in course_input.split(',') if s.strip()]
        try:
            result = course_model.recommend(skills)
            if not result.empty:
                st.success("📚 Suggested Courses:")
                st.dataframe(result)
            else:
                st.warning("No relevant courses found.")
        except Exception as e:
            st.error(f"Error: {e}")

# 📄 Resume Skill Analyzer
with tab3:
    st.header("📄 Resume Skill Gap & Similarity Analyzer")
    idx = st.number_input("Enter Resume Index from dataset:", min_value=0, step=1)
    if st.button("🔍 Analyze Resume"):
        try:
            st.subheader("✅ Missing Skills")
            missing = skill_model.recommend_missing_skills(idx)
            st.write(missing if missing else "No missing skills found.")

            st.subheader("🔁 Similar Resumes")
            similar = skill_model.find_similar_resumes(idx)
            st.dataframe(similar)
        except Exception as e:
            st.error(f"Error during analysis: {e}")

# 📂 Resume Category Predictor
with tab4:
    st.header("📂 Resume Category Predictor (PDF / TXT / Paste)")

    uploaded_file = st.file_uploader("📁 Upload your resume (PDF or TXT):", type=["pdf", "txt"])
    manual_text = st.text_area("📎 Or paste your resume content below:")

    resume_text = ""

    if uploaded_file:
        try:
            if uploaded_file.type == "application/pdf":
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    resume_text += page.extract_text()
            elif uploaded_file.type == "text/plain":
                resume_text = uploaded_file.read().decode("utf-8")
        except Exception as e:
            st.error(f"❌ Could not read file: {e}")

    if not resume_text and manual_text.strip():
        resume_text = manual_text

    if resume_text:
        if st.button("📌 Predict Category & Rate Resume"):
            try:
                category = category_model.predict_category(resume_text)
                rating = category_model.rate_resume(resume_text)
                st.success(f"🎯 Predicted Resume Category: **{category}**")
                st.info(f"⭐ Resume Rating (1 to 5): **{rating} / 5**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.info("📄 Please upload a resume file or paste the resume text above.")

# Footer
st.markdown("---")
st.markdown("🔧 Built with ❤️ by **CodeX Guru** | Powered by ML & Streamlit 🚀")
