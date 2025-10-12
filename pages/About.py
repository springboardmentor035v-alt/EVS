import streamlit as st
import base64
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="About EnviroScan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- PREMIUM CUSTOM STYLING ---
st.markdown("""
<style>
    /* Global Styles */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Premium Header */
    .premium-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .premium-subheader {
        font-size: 1.3rem;
        color: #495057;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* Premium Cards */
    .premium-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2.5rem;
        border-radius: 20px;
        border: none;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border-left: 6px solid #667eea;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .premium-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.12);
    }

    .overview-card-content h3 {
        color: #2c3e50;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }

    .overview-card-content p {
        color: #34495e;
        font-size: 1.1rem;
        line-height: 1.7;
        margin-bottom: 1.5rem;
    }

    .overview-card-content strong, .overview-card-content em {
        color: #667eea;
        font-weight: 600;
    }
    
    .tech-card-premium {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.07);
        border: 1px solid #e8e8e8;
        height: 100%;
        transition: all 0.3s ease;
        display: flex;
        flex-direction: column;
        text-align: center;
    }
    
    .tech-card-premium:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.12);
        border-color: #667eea;
    }

    .tech-card-premium ul {
        text-align: left;
        color: #495057;
        padding-left: 1.5rem;
        margin-top: 1rem;
        flex-grow: 1; /* Pushes content down if needed */
    }

    .tech-card-premium li {
        margin-bottom: 0.75rem;
    }
    
    /* Metric Cards */
    .metric-card-premium {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem 1rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card-premium:hover {
        transform: scale(1.05);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1rem;
        margin-top: 2rem;
        text-align: center;
        position: relative;
    }
    
    .section-subheader {
        text-align: center; 
        color: #495057; 
        font-size: 1.1rem; 
        margin-top: 0;
        margin-bottom: 3.5rem;
    }

    .section-header:after {
        content: '';
        position: absolute;
        bottom: -15px;
        left: 50%;
        transform: translateX(-50%);
        width: 80px;
        height: 4px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 2px;
    }
    
    /* Timeline */
    .timeline-premium {
        border-left: 3px solid #667eea;
        padding-left: 2.5rem;
        margin: 2rem 0;
    }
    
    .timeline-item-premium {
        margin-bottom: 2.5rem;
        position: relative;
        padding: 1.5rem;
        background: #fafcff;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    
    .timeline-item-premium:before {
        content: '';
        position: absolute;
        left: -2.75rem;
        top: 1.8rem;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: #667eea;
        border: 4px solid white;
        box-shadow: 0 0 0 3px #667eea;
    }
    
    /* Expander Elements */
    .stExpander {
        border: 1px solid #e0e0e0 !important;
        background: white !important;
        border-radius: 15px !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05) !important;
        margin: 1rem 0 !important;
        transition: box-shadow 0.3s ease;
    }
    
    .stExpander:hover {
        box-shadow: 0 10px 25px rgba(0,0,0,0.1) !important;
        border-color: #667eea !important;
    }
    
    /* Footer */
    .premium-footer {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-top: 3rem;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTION (NO CHANGE) ---
def img_to_bytes(img_path):
    try:
        img_bytes = Path(img_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded
    except:
        return None

# --- HEADER ---
st.markdown("""
<div style='text-align: center;'>
    <h1 class="premium-header">🔬 EnviroScan AI</h1>
    <p class="premium-subheader">
        Advanced Machine Learning for Intelligent Environmental Monitoring & Pollution Source Attribution
    </p>
</div>
""", unsafe_allow_html=True)

# --- PROJECT OVERVIEW ---
st.markdown('<div class="section-header">🎯 Project Overview</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 2], gap="large")

with col1:
    try:
        # Assuming the image is in a folder named 'assets' or similar
        st.image("/Users/hari/Downloads/SkcetAcc/3065.jpg",
                 caption="Fusing Data Science with Environmental Science",
                 use_container_width=True)
    except:
        st.markdown("""
        <div style='text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 20px; height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center;'>
            <div style='font-size: 4rem; margin-bottom: 1rem;'>🌍</div>
            <h3 style='color: #2c3e50;'>Environmental Intelligence Platform</h3>
            <p style='color: #495057;'>AI-Powered Pollution Source Analysis</p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="premium-card overview-card-content">
        <h3>🚀 Transforming Environmental Monitoring</h3>
        <p>
        <strong>EnviroScan AI</strong> represents a quantum leap in environmental analytics, harnessing the transformative power of machine learning to deliver unprecedented insights into pollution dynamics.
        </p>
        <p>
        We transcend traditional measurement approaches by providing <strong>real-time, actionable intelligence</strong> that identifies pollution <em>sources</em>, <em>patterns</em>, and <em>trends</em> with scientific precision.
        </p>
        <p>
        Through sophisticated analysis of multi-dimensional environmental data, our AI ecosystem accurately attributes pollution events to specific anthropogenic activities, enabling targeted interventions and sustainable urban planning.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1.5rem;'>
        <div class="metric-card-premium">
            <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🎯</div>
            <div style='font-size: 2.2rem; font-weight: bold;'>87%</div>
            <div style='font-size: 1rem; opacity: 0.9;'>Model Accuracy</div>
        </div>
        <div class="metric-card-premium">
            <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>📊</div>
            <div style='font-size: 2.2rem; font-weight: bold;'>18+</div>
            <div style='font-size: 1rem; opacity: 0.9;'>Data Features</div>
        </div>
        <div class="metric-card-premium">
            <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🌍</div>
            <div style='font-size: 2.2rem; font-weight: bold;'>10+</div>
            <div style='font-size: 1rem; opacity: 0.9;'>Cities Covered</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- STAKEHOLDERS ---
st.markdown("""
<div class="premium-card" style="margin-top: 3rem;">
    <h4 style='color: #2c3e50; font-size: 1.6rem; margin-bottom: 2rem; text-align: center;'>👥 Primary Stakeholders</h4>
    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem;'>
        <div style='display: flex; align-items: center;'>
            <span style='font-size: 2.5rem; margin-right: 1.5rem;'>🏛️</span>
            <div>
                <strong style='color: #2c3e50; font-size: 1.1rem;'>Government & Policy</strong>
                <p style='margin: 0.3rem 0 0 0; color: #495057; font-size: 0.95rem;'>Data-driven urban planning and environmental regulations</p>
            </div>
        </div>
        <div style='display: flex; align-items: center;'>
            <span style='font-size: 2.5rem; margin-right: 1.5rem;'>🔬</span>
            <div>
                <strong style='color: #2c3e50; font-size: 1.1rem;'>Research & Academia</strong>
                <p style='margin: 0.3rem 0 0 0; color: #495057; font-size: 0.95rem;'>Advanced environmental studies and ML applications</p>
            </div>
        </div>
        <div style='display: flex; align-items: center;'>
            <span style='font-size: 2.5rem; margin-right: 1.5rem;'>🌱</span>
            <div>
                <strong style='color: #2c3e50; font-size: 1.1rem;'>Environmental Agencies</strong>
                <p style='margin: 0.3rem 0 0 0; color: #495057; font-size: 0.95rem;'>Optimized monitoring and enforcement strategies</p>
            </div>
        </div>
        <div style='display: flex; align-items: center;'>
            <span style='font-size: 2.5rem; margin-right: 1.5rem;'>🏢</span>
            <div>
                <strong style='color: #2c3e50; font-size: 1.1rem;'>Industry Leaders</strong>
                <p style='margin: 0.3rem 0 0 0; color: #495057; font-size: 0.95rem;'>Environmental compliance and sustainability initiatives</p>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- METHODOLOGY ---
st.markdown('<div class="section-header">⚙️ Scientific Methodology</div>', unsafe_allow_html=True)
st.markdown('<p class="section-subheader">Our end-to-end AI pipeline transforms complex environmental data into actionable intelligence.</p>', unsafe_allow_html=True)

method_cols = st.columns(3, gap="large")
with method_cols[0]:
    st.markdown("""
    <div class="tech-card-premium">
        <div style='font-size: 3.5rem; margin-bottom: 1rem; color: #667eea;'>📡</div>
        <h4 style='color: #2c3e50; margin-bottom: 1rem; font-size: 1.3rem;'>Data Intelligence</h4>
        <ul>
            <li>Multi-source environmental data fusion</li>
            <li>Real-time air quality metrics</li>
            <li>Geospatial feature extraction</li>
            <li>Meteorological data integration</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with method_cols[1]:
    st.markdown("""
    <div class="tech-card-premium">
        <div style='font-size: 3.5rem; margin-bottom: 1rem; color: #667eea;'>🔧</div>
        <h4 style='color: #2c3e50; margin-bottom: 1rem; font-size: 1.3rem;'>Smart Processing</h4>
        <ul>
            <li>Advanced data validation & cleaning</li>
            <li>Intelligent feature engineering</li>
            <li>Temporal pattern analysis</li>
            <li>Spatial aggregation & normalization</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with method_cols[2]:
    st.markdown("""
    <div class="tech-card-premium">
        <div style='font-size: 3.5rem; margin-bottom: 1rem; color: #667eea;'>🤖</div>
        <h4 style='color: #2c3e50; margin-bottom: 1rem; font-size: 1.3rem;'>AI Analytics</h4>
        <ul>
            <li>Random Forest ensemble modeling</li>
            <li>Real-time pollution source prediction</li>
            <li>Feature importance analysis</li>
            <li>Model confidence scoring</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- DATA PIPELINE ---
st.markdown("""
<div class="premium-card" style='margin-top: 4rem;'>
    <h3 style='color: #2c3e50; font-size: 1.8rem; margin-bottom: 2rem; text-align: center;'>🏗️ Data Pipeline Architecture</h3>
    <div class="timeline-premium">
        <div class="timeline-item-premium">
            <h4 style='color: #2c3e50; margin-bottom: 0.5rem;'>📥 Intelligent Data Ingestion</h4>
            <p style='color: #495057; margin: 0;'>Multi-source environmental data collection from APIs, sensors, and geospatial databases with real-time streaming capabilities.</p>
        </div>
        <div class="timeline-item-premium">
            <h4 style='color: #2c3e50; margin-bottom: 0.5rem;'>🛠️ Advanced Preprocessing</h4>
            <p style='color: #495057; margin: 0;'>Sophisticated data cleaning, feature engineering, and quality assurance with automated validation pipelines.</p>
        </div>
        <div class="timeline-item-premium">
            <h4 style='color: #2c3e50; margin-bottom: 0.5rem;'>🏷️ Smart Source Labeling</h4>
            <p style='color: #495057; margin: 0;'>Rule-based pollution source attribution integrated with domain expertise for high-quality training data generation.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- MODEL TRAINING & EVALUATION ---
st.markdown('<div class="section-header" style="margin-top: 4rem;">🌳 AI Model Architecture</div>', unsafe_allow_html=True)

model_cols = st.columns(2, gap="large")
with model_cols[0]:
    with st.expander("🎯 **Why Random Forest?**", expanded=True):
        st.markdown("""
        <div style='padding: 1rem;'>
            <div style='display: grid; grid-template-columns: 1fr; gap: 1rem;'>
                <div style='background: #f0f3ff; border-left: 5px solid #667eea; padding: 1rem; border-radius: 10px;'>
                    <strong style='color:#2c3e50;'>High Accuracy</strong>
                    <p style='margin: 0.5rem 0 0 0; color: #495057;'>Superior performance on complex, multi-variate environmental classification tasks.</p>
                </div>
                <div style='background: #f0fff0; border-left: 5px solid #4caf50; padding: 1rem; border-radius: 10px;'>
                    <strong style='color:#2c3e50;'>Robustness</strong>
                    <p style='margin: 0.5rem 0 0 0; color: #495057;'>Effectively handles non-linear relationships and noisy environmental data without overfitting.</p>
                </div>
                <div style='background: #fff8e1; border-left: 5px solid #ffab00; padding: 1rem; border-radius: 10px;'>
                    <strong style='color:#2c3e50;'>Interpretability</strong>
                    <p style='margin: 0.5rem 0 0 0; color: #495057;'>Provides clear feature importance analysis for transparent, data-driven decision-making.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with model_cols[1]:
    with st.expander("📊 **Model Performance**", expanded=True):
        st.markdown("""
        <div style='padding: 1rem;'>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; text-align: center;'>
                <div style='background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                    <div style='font-size: 2.2rem; font-weight: bold; color: #667eea;'>87%</div>
                    <div style='color: #495057; font-size: 0.9rem; font-weight: 500;'>Accuracy</div>
                </div>
                <div style='background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                    <div style='font-size: 2.2rem; font-weight: bold; color: #667eea;'>85%</div>
                    <div style='color: #495057; font-size: 0.9rem; font-weight: 500;'>Precision</div>
                </div>
                <div style='background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                    <div style='font-size: 2.2rem; font-weight: bold; color: #667eea;'>83%</div>
                    <div style='color: #495057; font-size: 0.9rem; font-weight: 500;'>Recall</div>
                </div>
                <div style='background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                    <div style='font-size: 2.2rem; font-weight: bold; color: #667eea;'>84%</div>
                    <div style='color: #495057; font-size: 0.9rem; font-weight: 500;'>F1-Score</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- TECHNOLOGY STACK ---
st.markdown('<div class="section-header">💻 Technology Ecosystem</div>', unsafe_allow_html=True)

tech_cols = st.columns(4)
technologies = [
    ("🐍", "Python 3.9+", "Core programming ecosystem for data science and ML."),
    ("🎈", "Streamlit", "Interactive web application framework for rapid development."),
    ("🤖", "Scikit-learn", "Comprehensive machine learning library for classification."),
    ("🐼", "Pandas", "High-performance data manipulation and analysis toolkit."),
    ("🗺️", "Folium & GeoPandas", "Advanced geospatial mapping and spatial analysis."),
    ("📈", "Plotly", "Interactive, publication-quality data visualizations."),
    ("💾", "Joblib & Pickle", "Model serialization and efficient pipeline persistence."),
    ("🎨", "Matplotlib & Seaborn", "Statistical visualization and foundational plotting.")
]

for i, (icon, name, desc) in enumerate(technologies):
    with tech_cols[i % 4]:
        st.markdown(f"""
        <div class="tech-card-premium" style="margin-bottom: 1rem;">
            <div style='font-size: 2.5rem; margin-bottom: 1rem;'>{icon}</div>
            <strong style='color: #2c3e50; font-size: 1.1rem;'>{name}</strong>
            <p style='margin: 0.8rem 0 0 0; color: #495057; font-size: 0.95rem; flex-grow: 1;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div class="premium-footer">
    <h3 style='color: white; margin-bottom: 1.5rem; font-size: 2rem; font-weight: 700;'>🔬 EnviroScan AI</h3>
    <p style='margin-bottom: 1rem; font-size: 1.2rem; color: #ecf0f1;'>
        <strong>Transforming Environmental Intelligence Through Advanced AI</strong>
    </p>
    <p style='margin: 0; font-size: 1rem; color: #bdc3c7;'>
        Built with precision for a sustainable future | Data-driven environmental stewardship
    </p>
    <div style='margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid #4a627a;'>
        <p style='margin: 0; font-size: 0.9rem; color: #95a5a6;'>
            © 2025 EnviroScan AI | Advanced Environmental Analytics Platform
        </p>
    </div>
</div>
""", unsafe_allow_html=True)