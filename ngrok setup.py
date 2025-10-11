# -----------------------------
# Colab / ngrok Setup to run Streamlit
# -----------------------------
!pip install --quiet streamlit pyngrok folium plotly

import os, time, subprocess
from pyngrok import ngrok

# Kill any existing Streamlit processes
!pkill -f streamlit

# Set your ngrok auth token
NGROK_AUTH_TOKEN = "33QCHOtq4hHFWEXCB9kDmX8UdPk_6jF4d3HanwvLbsbkUS1H"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Run Streamlit app in background
streamlit_process = subprocess.Popen(["streamlit", "run", "app.py"])

# Give it a few seconds to start
time.sleep(10)

# Open ngrok tunnel
public_url = ngrok.connect(8501)
print(f"🌐 Your Streamlit Dashboard is live at: {public_url}")

try:
    streamlit_process.wait()
except KeyboardInterrupt:
    streamlit_process.terminate()
    ngrok.disconnect(public_url)
    print("Terminated Streamlit and ngrok tunnel.")
