import streamlit as st
import base64
from pathlib import Path

# Add interactive background
with open("background.html", "r") as f:
    background_html = f.read()
st.markdown(background_html, unsafe_allow_html=True)

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

# Create 5 columns with equal width for 13 technologies (2 or 3 per column)
tech_cols = st.columns([1, 1, 1, 1, 1])  # Equal width for all 5 columns

# List of technologies
technologies = [
    ("Python", "The core programming language for all backend processing."),
    ("Streamlit", "Used to build this interactive web application and dashboard."),
    ("Scikit-learn", "The primary library for building and evaluating our machine learning models."),
    ("Pandas", "The essential tool for data manipulation, cleaning, and analysis."),
    ("OSMX (OSMnx)", "Utilized for collecting geospatial data from OpenStreetMap."),
    ("OpenWeather", "Provides real-time weather and air quality data for analysis."),
    ("OpenAQ", "Offers open-access air quality data for comprehensive monitoring."),
    ("Folium", "Powers the interactive geospatial maps as output."),
    ("Plotly", "Used for creating interactive charts and trend visualizations."),
    ("Joblib", "For serializing and saving our trained model and encoders."),
    ("Seaborn", "For generating static plots like the confusion matrix and feature importance charts."),
    ("GeoPandas", "Enhances geospatial data handling and analysis."),
    ("Requests", "Facilitates API calls for data collection from external sources.")
]

# Distribute all technologies across 5 columns (2 or 3 per column)
for i in range(0, len(technologies), 2):  # Step by 2 to pair technologies
    col_idx = i // 2  # Map to column index (0 to 4)
    if col_idx < len(tech_cols):  # Ensure we don't exceed column count
        with tech_cols[col_idx]:
            # Use custom markdown for card with matching gradient color
            st.markdown(
                f"""
                <div style="color: #ffffff; border: 1px solid rgba(250, 250, 250, 0.2); border-radius: 5px; padding: 10px;">
                <h4>{technologies[i][0]}</h4>{technologies[i][1]}
                </div>
                """,
                unsafe_allow_html=True
            )
            if i + 1 < len(technologies):  # Display second technology if available
                st.markdown(
                    f"""
                    <div style="color: #ffffff; border: 1px solid rgba(250, 250, 250, 0.2); border-radius: 5px; padding: 10px;">
                    <h4>{technologies[i + 1][0]}</h4>{technologies[i + 1][1]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

st.markdown("---")
st.write("Created with a focus on data-driven environmental stewardship.")
