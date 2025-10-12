# src/utils.py

import pandas as pd
import plotly.express as px
import pdfkit
from datetime import datetime

# --- Logic Function 1: Perform Predictions ---
def perform_prediction(df, model, scaler, features):
    """
    Takes a dataframe and returns it with a new 'predicted_source' column.
    """
    # Ensure all required feature columns are present
    if not all(col in df.columns for col in features):
        missing = [col for col in features if col not in df.columns]
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")
    
    # Ensure data is in the correct order for the model
    X_batch = df[features]
    X_batch_scaled = scaler.transform(X_batch)
    predictions = model.predict(X_batch_scaled)
    
    df['predicted_source'] = predictions
    return df

# --- Logic Function 2: Generate Report HTML ---
def generate_report_html(results_df):
    """
    Takes a dataframe with prediction results and generates an HTML report.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # For a single prediction, we can show more detail
    if len(results_df) == 1:
        prediction_row = results_df.iloc[0]
        pred = prediction_row['predicted_source']
        
        html = f"""
        <h1 style='color:#00FFFF;'>🌍 EnviroScan Prediction Report</h1>
        <p><b>Generated:</b> {timestamp}</p>
        <h2 style='color:#00FFFF;'>Predicted Source: {pred}</h2>
        <p><b>Input Values:</b></p>
        {prediction_row.to_frame().to_html()}
        """
    # For a batch report, we show a summary table
    else:
        source_counts = results_df['predicted_source'].value_counts().to_frame().to_html()
        html = f"""
        <h1 style='color:#00FFFF;'>🌍 EnviroScan Batch Prediction Report</h1>
        <p><b>Generated:</b> {timestamp}</p>
        <h2 style='color:#00FFFF;'>Prediction Summary</h2>
        {source_counts}
        <hr>
        <h2 style='color:#00FFFF;'>Full Results</h2>
        {results_df.to_html(index=False)}
        """
    return html

# --- Utility Function 3: Convert HTML to PDF ---
def convert_to_pdf(html_string):
    """Converts an HTML string to a PDF byte object."""
    try:
        
        path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
        config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
        # Update the pdfkit call to use the new configuration
        pdf = pdfkit.from_string(html_string, False, configuration=config)
        return pdf
    except OSError:
        # This error happens if the path is wrong or wkhtmltopdf is not installed
        return None

# --- Utility Function 4: Convert DataFrame to CSV ---
def convert_to_csv(df):
    """Converts a DataFrame to a CSV byte object for downloading."""
    return df.to_csv(index=False).encode('utf-8')