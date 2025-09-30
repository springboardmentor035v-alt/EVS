import streamlit as st
import base64
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="About EnviroScan AI",
    page_icon="ℹ️",
    layout="wide",
)

# --- HELPER FUNCTION TO READ & ENCODE IMAGE ---
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

# --- HEADER ---
st.title("ℹ️ About EnviroScan AI")
st.markdown("Understanding the technology and methodology behind our predictions.")
st.markdown("---")

# --- PROJECT OVERVIEW ---
st.header("Project Overview")
col1, col2 = st.columns([1.5, 2])
with col1:
    st.image("pages/image.png",
             caption="Fusing Data Science with Environmental Science",
             use_container_width=True)

with col2:
    st.markdown("""
    **EnviroScan AI** is a proof-of-concept platform designed to demonstrate the power of machine learning in environmental science. Our primary goal is to move beyond simply measuring pollution levels and instead provide actionable insights into the *sources* of that pollution.

    By analyzing a combination of real-time air quality data, weather patterns, and geospatial information, our AI model can attribute pollution events to specific categories, such as vehicular traffic, industrial activity, or agricultural burning.

    This tool is intended for:
    - **Urban Planners & Policymakers:** To make informed, data-driven decisions on city planning and environmental regulations.
    - **Environmental Agencies:** To better allocate resources for monitoring and enforcement.
    - **Researchers & Students:** To study the complex interplay of factors contributing to urban air pollution.
    """)

st.markdown("---")

# --- METHODOLOGY ---
st.header("🔬 Our Methodology")
st.markdown("Our approach is centered around a supervised machine learning model. Here's a breakdown of the process:")

# --- DATA PIPELINE ---
st.subheader("1. Data Pipeline")
st.markdown("""
The foundation of our model is a robust data pipeline that aggregates and processes information from multiple sources. This automated pipeline ensures our model is trained on clean, relevant, and well-structured data.
- **Ingestion:** We collect air quality metrics (PM₂.₅, NO₂, SO₂, etc.), meteorological data (temperature, humidity, wind speed), and geospatial features (proximity to roads, industrial zones).
- **Preprocessing:** Data is cleaned to handle missing values, scaled to a uniform range to ensure model stability, and engineered to create new features like time-of-day and day-of-week.
- **Labeling:** A rule-based engine assigns a preliminary `pollution_source` label to the data based on known correlations (e.g., high NO₂ during rush hour near a major road is likely 'Vehicular'). This creates our training dataset.
""")

# --- MODEL TRAINING & EVALUATION ---
st.subheader("2. Model Training and Evaluation")
st.markdown("""
We use a **Random Forest Classifier**, a powerful and interpretable ensemble learning method. It consists of numerous individual decision trees that operate as a collective. Each tree votes on a class prediction, and the class with the most votes becomes the model's final prediction.
""")

with st.expander("Why did we choose a Random Forest?"):
    st.markdown("""
    - **High Accuracy:** It is known for its strong performance on a wide range of classification problems.
    - **Robustness:** It handles non-linear relationships and is less prone to overfitting compared to single decision trees.
    - **Interpretability:** It allows us to calculate and visualize **Feature Importance**, showing which environmental factors are most influential in predicting pollution sources.
    - **Efficiency:** It can be parallelized and trained efficiently on large datasets.
    """)

st.markdown("""
**Evaluation:** After training, the model's performance is rigorously tested on a separate, unseen dataset. We use standard metrics to ensure its predictions are reliable:
- **Classification Report:** Provides a detailed breakdown of precision, recall, and F1-score for each pollution source class.
- **Confusion Matrix:** A visual table that shows the model's performance, highlighting where it makes correct and incorrect predictions.
""")

# --- TECHNOLOGY STACK ---
st.markdown("---")
st.header("💻 Technology Stack")
st.markdown("This platform is built using a modern, open-source technology stack:")
tech_cols = st.columns(4)
technologies = [
    ("Python", "The core programming language for all backend processing."),
    ("Streamlit", "Used to build this interactive web application and dashboard."),
    ("Scikit-learn", "The primary library for building and evaluating our machine learning models."),
    ("Pandas", "The essential tool for data manipulation, cleaning, and analysis."),
    ("Folium", "Powers the interactive geospatial maps on the dashboard."),
    ("Plotly", "Used for creating interactive charts and trend visualizations."),
    ("Joblib", "For serializing and saving our trained model and encoders."),
    ("Seaborn", "For generating static plots like the confusion matrix and feature importance charts.")
]

for i, col in enumerate(tech_cols):
    if i < len(technologies):
        with col:
            st.info(f"**{technologies[i][0]}**\n\n{technologies[i][1]}")

st.markdown("---")
st.write("Created with a focus on data-driven environmental stewardship.")

