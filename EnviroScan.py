import streamlit as st
import pandas as pd
import os

# Inject CSS for sidebar scrollbar
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        max-height: 80vh;  /* 80% of viewport height, adjust as needed */
        overflow-y: auto;  /* Enable vertical scrollbar */
        overflow-x: hidden; /* Hide horizontal scrollbar */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add interactive background
with open("background.html", "r") as f:
    background_html = f.read()
st.markdown(background_html, unsafe_allow_html=True)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EnviroScan AI - Home",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- HEADER SECTION ---
with st.container():
    st.title("EnviroScan AI 💨")
    st.subheader("AI-Powered Pollution Source Identification using Geospatial Analytics")
    st.markdown("---")

# --- INTRODUCTION & IMAGE ---
col1, col2 = st.columns([2, 1.5])

with col1:
    st.markdown("""
    ### Environmental Monitoring
    
    EnviroScan AI leverages a robust, enterprise-grade machine learning framework to efficiently process complex environmental datasets, delivering near-real-time pollution source identification.<br> 
    Adopt EnviroScan AI to enable IT-driven decision-making for authorities, researchers, and urban planners, optimizing resource allocation and enhancing environmental management with precision and scalability.
    """, unsafe_allow_html=True)
    st.success("**👈 Select a page from the sidebar to begin your analysis!**", icon="✅")
    st.markdown("""
    Our platform provides an intuitive interface to interact with powerful analytics, turning raw data into actionable intelligence. Explore pollution trends, simulate scenarios, and uncover insights that were previously hidden in the noise.
    """)

with col2:
    st.image(
            "Picsart_25-10-12_12-06-26-286.jpg",
            use_container_width=True,
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

st.header("How EnviroScan AI Works")
st.markdown("A streamlined overview of our comprehensive data-to-insight pipeline:")

flow_cols = st.columns(4)
steps = [
    ("Data Loader", "Gathers real-time and historical environmental data—air quality metrics, weather patterns, and geospatial information—from a network of monitoring stations and external sources, ensuring a robust data foundation."),
    ("Model Training", "Preprocesses and cleans the collected data, training an advanced machine learning model to accurately classify and predict pollution sources based on complex patterns and features."),
    ("Deployment", "Deploys the trained model into a scalable, real-time prediction system, enabling seamless integration and instant analysis for actionable environmental insights."),
    ("Dashboard", "Presents the results through an interactive, user-friendly dashboard, allowing users to visualize trends, simulate scenarios, and derive strategic decisions with ease.")
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
