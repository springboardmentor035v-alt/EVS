import streamlit as st
import pandas as pd
import os
import subprocess
import sys

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EnviroScan AI - Home",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    /* Main App Styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Custom Headers */
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .header h1 {
        font-size: 3rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #0072ff, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header p {
        font-size: 1.25rem;
        color: #4f4f4f;
    }

    /* Section Headers */
    .section-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1e1e1e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #0072ff;
        padding-bottom: 0.5rem;
    }

    /* Custom Cards for Features */
    .feature-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        border-color: #0072ff;
    }
    .feature-card h4 {
        margin-top: 0;
        color: #1e1e1e;
        font-size: 1.25rem;
        font-weight: 600;
    }
    .feature-card p {
        color: #4f4f4f;
        flex-grow: 1; /* Pushes status to the bottom */
    }
    .status-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        text-align: center;
        margin-top: 1rem;
    }
    .status-ready {
        background-color: #e6f7f0;
        color: #00874e;
    }
    .status-pending {
        background-color: #fff4e5;
        color: #ff9800;
    }
    .status-available {
        background-color: #e9f5ff;
        color: #0072ff;
    }

    /* Pipeline Status Card */
    .pipeline-card {
        background-color: #fffbe6;
        border-left: 5px solid #ffc107;
        padding: 1.5rem;
        border-radius: 8px;
    }
    
    /* Custom Button */
    .stButton>button {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }

</style>
""", unsafe_allow_html=True)


def check_pipeline_status():
    """Check if pipeline has been run and models are available"""
    status = {
        'models_ready': False,
        'output_files': [],
        'missing_files': []
    }
    
    expected_files = [
        "outputs/pollution_source_model.joblib",
        "outputs/label_encoder.joblib",
        "outputs/scaler.joblib",
        "outputs/processed_data_for_eda.csv"
    ]
    
    for file in expected_files:
        if os.path.exists(file):
            status['output_files'].append(file)
        else:
            status['missing_files'].append(file)
    
    status['models_ready'] = len(status['missing_files']) == 0
    return status

def run_pipeline():
    """Run the data pipeline"""
    try:
        with st.spinner('🚀 **Running Data & Modeling Pipeline...** This may take a few minutes. Please wait.'):
            result = subprocess.run(
                [sys.executable, "run_pipeline.py"],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                st.success("✅ **Pipeline executed successfully!** All models are trained and ready.")
                return True, result.stdout
            else:
                st.error("❌ **Pipeline failed with an error.** See details below.")
                return False, result.stderr
                
    except subprocess.TimeoutExpired:
        st.error("⏰ **Pipeline timed out after 10 minutes.** The process took too long. Please check the `run_pipeline.py` script for potential issues.")
        return False, "Pipeline execution timed out"
    except Exception as e:
        st.error(f"💥 **An unexpected error occurred while running the pipeline:** {e}")
        return False, str(e)

# --- HEADER SECTION ---
st.markdown("""
<div class="header">
    <h1>Welcome to EnviroScan AI 💨</h1>
    <p>AI-Powered Pollution Source Identification using Geospatial Analytics</p>
</div>
""", unsafe_allow_html=True)

# --- PIPELINE STATUS & SETUP ---
st.markdown('<h2 class="section-header">🚀 Quick Setup & Status</h2>', unsafe_allow_html=True)
pipeline_status = check_pipeline_status()

if not pipeline_status['models_ready']:
    st.markdown("""
    <div class="pipeline-card">
        <h3 style="margin-top:0;">⚠️ Models Not Trained Yet!</h3>
        <p>Before using the prediction features, you must run the data pipeline. This process will prepare the data, train the machine learning models, and save the necessary output files.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("▶️ Run Data Pipeline", type="primary", help="Click to start the model training process"):
        success, output = run_pipeline()
        
        if success:
            st.toast("Pipeline complete! Refreshing...")
            st.rerun()
        else:
            st.error("Pipeline failed. Please check the error log below for details:")
            st.code(output, language='bash')
            
    with st.expander("Show Pipeline File Status Details"):
        st.write("**Found files:**")
        if pipeline_status['output_files']:
            for file in pipeline_status['output_files']:
                st.markdown(f"✔️ `{file}`")
        else:
            st.write("None.")
            
        st.write("**Missing files:**")
        if pipeline_status['missing_files']:
            for file in pipeline_status['missing_files']:
                st.markdown(f"❌ `{file}`")
        else:
            st.write("None.")
else:
    st.success("✅ **System Ready!** All models are trained and available. You can now access all platform features from the sidebar.", icon="🎉")

st.markdown("---")

# --- INTRODUCTION & IMAGE ---
st.markdown('<h2 class="section-header">💡 About the Platform</h2>', unsafe_allow_html=True)
col1, col2 = st.columns([1.8, 1], gap="large")

with col1:
    st.markdown("""
    ### Revolutionizing Environmental Monitoring
    
    EnviroScan AI leverages a sophisticated machine learning model to analyze complex environmental data. By identifying pollution sources in near real-time, we empower authorities, researchers, and urban planners to make faster, more effective decisions for a cleaner, healthier planet.
    
    Our platform provides an intuitive interface to interact with powerful analytics, turning raw data into actionable intelligence. Explore pollution trends, simulate scenarios, and uncover insights that were previously hidden in the noise.
    """)
    
    if pipeline_status['models_ready']:
        st.info("**👈 Select a page from the sidebar to begin your analysis!**", icon="🧭")
    else:
        st.warning("**⬆️ Run the data pipeline first to unlock all features.**", icon="⚙️")
        
with col2:
    st.image(
        "/Users/hari/Downloads/SkcetAcc/2151929046.jpg",
        caption="AI-Powered Environmental Intelligence",
        use_container_width=True
    )

st.markdown("---")

# --- CORE FEATURES (in cards) ---
st.markdown('<h2 class="section-header">✨ Core Features</h2>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3, gap="large")

features = [
    {
        "col": c1, "title": "🌍 Pollution Dashboard",
        "desc": "Visualize geospatial pollution hotspots, analyze trends over time, and filter data by city or predicted source. Get a high-level overview of the environmental landscape.",
        "status": "ready" if pipeline_status['models_ready'] else "pending"
    },
    {
        "col": c2, "title": "🔬 Live Prediction",
        "desc": "Interact directly with our AI model. Input custom environmental parameters to simulate a scenario and receive an instant prediction for the likely pollution source.",
        "status": "ready" if pipeline_status['models_ready'] else "pending"
    },
    {
        "col": c3, "title": "ℹ️ About",
        "desc": "Discover the methodology behind EnviroScan AI. Learn about the data sources, the machine learning models, and the technologies that power this platform.",
        "status": "available"
    }
]

for feature in features:
    with feature["col"]:
        status_text = {
            "ready": "✅ Ready to Use",
            "pending": "⚠️ Run Pipeline First",
            "available": "📖 Always Available"
        }.get(feature["status"])
        
        st.markdown(f"""
        <div class="feature-card">
            <h4>{feature['title']}</h4>
            <p>{feature['desc']}</p>
            <div class="status-badge status-{feature['status']}">
                {status_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# --- DATA SCOPE & INSTRUCTIONS ---
st.markdown('<h2 class="section-header">📋 Setup & Data Scope</h2>', unsafe_allow_html=True)
sc1, sc2 = st.columns(2, gap="large")

with sc1:
    st.subheader("Setup Instructions")
    instructions = """
    1.  **Get OpenWeather API Key** (free tier):
        -   Sign up at [OpenWeatherMap API](https://openweathermap.org/api)
        -   Create a `.env` file in the project root.
        -   Add your key: `OPENWEATHER_API_KEY="your_key_here"`

    2.  **Run the Data Pipeline**:
        -   Click the **"Run Data Pipeline"** button at the top of this page.
        -   *Alternatively, run from your terminal:* `python run_pipeline.py`

    3.  **Start Exploring**:
        -   Use the sidebar to navigate to the platform's features.
    """
    st.info(instructions, icon="⚙️")

with sc2:
    st.subheader("Current Monitoring Scope")
    try:
        locations_path = os.path.join("data", "locations.csv")
        df_locations = pd.read_csv(locations_path)
        num_cities = df_locations['name'].nunique()
        
        st.metric(label="Cities Currently Monitored", value=f"{num_cities} Cities")
        st.success(f"Our network currently covers **{num_cities} cities**. We are continuously expanding our reach to provide more comprehensive global coverage.", icon="🌐")
        
        with st.expander("View Monitored Cities"):
            st.dataframe(df_locations, use_container_width=True)
            
    except FileNotFoundError:
        st.error("`data/locations.csv` not found. Please ensure the locations file exists to see monitoring scope.", icon="⚠️")
