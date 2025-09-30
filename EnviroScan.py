import streamlit as st
import pandas as pd
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EnviroScan AI - Home",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- HEADER SECTION ---
with st.container():
    st.title("Welcome to EnviroScan AI 💨")
    st.subheader("AI-Powered Pollution Source Identification using Geospatial Analytics")
    st.markdown("---")

# --- INTRODUCTION & IMAGE ---
col1, col2 = st.columns([2, 1.5])

with col1:
    st.markdown("""
    ### Revolutionizing Environmental Monitoring
    
    EnviroScan AI leverages a sophisticated machine learning model to analyze complex environmental data. By identifying pollution sources in near real-time, we empower authorities, researchers, and urban planners to make faster, more effective decisions for a cleaner, healthier planet.
    """)
    st.success("**👈 Select a page from the sidebar to begin your analysis!**", icon="✅")
    st.markdown("""
    Our platform provides an intuitive interface to interact with powerful analytics, turning raw data into actionable intelligence. Explore pollution trends, simulate scenarios, and uncover insights that were previously hidden in the noise.
    """)

with col2:
    st.image(
        "https://image.pollinations.ai/prompt/An%20urban%20landscape%20with%20transparent%20data%20overlays%20and%20environmental%20monitoring%20elements%2C%20symbolizing%20AI-powered%20pollution%20source%20identification.%20Clean%2C%20modern%2C%20digital%20art%20style%20with%20a%20focus%20on%20data%20visualization%20and%20sustainability%20themes.",
        caption="AI-Powered Environmental Monitoring",
        use_container_width=True
    )

st.markdown("---")

# --- CORE FEATURES (in cards) ---
st.header("Explore the Platform's Core Features")
c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.markdown("#### 🌍 Pollution Dashboard")
        st.write("Visualize geospatial pollution hotspots, analyze trends over time, and filter data by city or predicted source. Get a high-level overview of the environmental landscape.")
        
with c2:
    with st.container(border=True):
        st.markdown("#### 🔬 Live Prediction")
        st.write("Interact directly with our AI model. Input custom environmental parameters to simulate a scenario and receive an instant prediction for the likely pollution source.")

with c3:
    with st.container(border=True):
        st.markdown("#### ℹ️ About")
        st.write("Discover the methodology behind EnviroScan AI. Learn about the data sources, the machine learning models, and the technologies that power this platform.")

st.markdown("---")

# --- HOW IT WORKS ---
st.header("How EnviroScan AI Works")
st.markdown("A simplified overview of our end-to-end data pipeline:")

flow_cols = st.columns(4)
steps = [
    ("Data Ingestion", "Real-time sensor and geospatial data is collected from various monitoring stations."),
    ("AI Modeling", "Data is preprocessed, cleaned, and fed into our trained classification model."),
    ("Source Prediction", "The model analyzes the inputs and accurately predicts the dominant pollution source."),
    ("Insight Visualization", "Results are displayed on an interactive dashboard for easy interpretation and analysis.")
]

for i, col in enumerate(flow_cols):
    with col:
        with st.container(border=True):
            st.metric(label=f"Step {i+1}", value=steps[i][0])
            st.caption(steps[i][1])

st.markdown("---")

# --- DATA SCOPE ---
st.header("Current Monitoring Scope")
try:
    locations_path = os.path.join("data", "locations.csv")
    df_locations = pd.read_csv(locations_path)
    num_cities = df_locations['name'].nunique()
    
    ds_col1, ds_col2 = st.columns([1, 3])
    with ds_col1:
        st.metric(label="Cities Currently Monitored", value=num_cities)
    with ds_col2:
        st.info(f"Our network currently covers **{num_cities} cities**. We are continuously expanding our reach to provide more comprehensive global coverage.", icon="🌐")
        
except FileNotFoundError:
    st.warning("Location data file ('data/locations.csv') not found. Scope metrics cannot be displayed.", icon="⚠️")
