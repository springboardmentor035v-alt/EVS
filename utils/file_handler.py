import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from typing import Dict, Any
import base64

class FileHandler:
    """Handle file operations for the application"""
    
    @staticmethod
    def dataframe_to_csv(df: pd.DataFrame) -> bytes:
        """Convert DataFrame to CSV bytes"""
        return df.to_csv(index=False).encode('utf-8')
    
    @staticmethod
    def create_analysis_pdf(df: pd.DataFrame, prediction: Dict[str, Any]) -> bytes:
        """Create PDF report for single prediction"""
        from utils.visualization import VisualizationEngine
        
        buf = BytesIO()
        
        fig = VisualizationEngine.create_single_prediction_chart(prediction)
        plt.savefig(buf, format='pdf', bbox_inches='tight')
        plt.close(fig)
        
        buf.seek(0)
        return buf.getvalue()
    
    @staticmethod
    def create_batch_report_pdf(df: pd.DataFrame) -> bytes:
        """Create PDF report for batch analysis"""
        from utils.visualization import VisualizationEngine
        
        buf = BytesIO()
        
        charts = VisualizationEngine.create_pollution_dashboard_charts(df)
        
        if charts:
            # Create a multi-page PDF with all charts
            from matplotlib.backends.backend_pdf import PdfPages
            
            with PdfPages(buf) as pdf:
                for chart_name, fig in charts.items():
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
        
        buf.seek(0)
        return buf.getvalue()