import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

def create_statistical_summary(df):
    """Generate statistical summary of the dataset"""
    summary = []
    
    # Basic dataset information
    summary.append("## Dataset Overview")
    summary.append(f"* Number of Rows: {df.shape[0]}")
    summary.append(f"* Number of Columns: {df.shape[1]}")
    summary.append(f"* Missing Values: {df.isna().sum().sum()}")
    
    # Statistical summary
    summary.append("\n## Statistical Summary")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        stats_summary = numeric_df.describe().to_string()
        summary.append("```\n" + stats_summary + "\n```")
    
    # Data types information
    summary.append("\n## Column Data Types")
    dtypes_info = df.dtypes.to_string()
    summary.append("```\n" + dtypes_info + "\n```")
    
    return "\n".join(summary)

def create_correlation_heatmap(df):
    """Generate correlation heatmap for numerical columns"""
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title("Correlation Heatmap")
        return plt.gcf()
    return None

def generate_plots(df, target_variable=None):
    """Generate various plots based on the data"""
    plots = []
    
    # Distribution plots for numerical columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        fig = plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f"Distribution of {col}")
        plots.append(("distribution", col, fig))
        plt.close()
    
    # Bar plots for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].nunique() < 10:  # Only for columns with reasonable number of categories
            fig = plt.figure(figsize=(10, 6))
            df[col].value_counts().plot(kind='bar')
            plt.title(f"Distribution of {col}")
            plt.xticks(rotation=45)
            plots.append(("categorical", col, fig))
            plt.close()
    
    # Box plots if target variable is specified
    if target_variable and target_variable in df.columns:
        if df[target_variable].dtype in ['float64', 'int64']:
            for col in numeric_cols:
                if col != target_variable:
                    fig = plt.figure(figsize=(10, 6))
                    sns.boxplot(data=df, y=col)
                    plt.title(f"Box Plot of {col}")
                    plots.append(("box", col, fig))
                    plt.close()
    
    return plots

def generate_report(df, target_variable=None, trained_model=None):
    """
    Generate a comprehensive HTML report for download without previewing it.
    """
    # Generate the plots first
    plots = generate_plots(df, target_variable)
    
    # Generate the HTML report content
    html_report = generate_downloadable_report(df, plots, target_variable, trained_model)
    
    # Trigger a download button for the report
    st.download_button(
        label="📥 Download Report",
        data=html_report,
        file_name=f"data_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        mime="text/html"
    )

def generate_downloadable_report(df, plots, target_variable, trained_model):
    """Generate an HTML report that can be downloaded"""
    html_content = f"""
    <html>
    <head>
        <title>Data Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .container {{ max-width: 1200px; margin: auto; }}
            .stats-table {{ border-collapse: collapse; width: 100%; }}
            .stats-table td, .stats-table th {{ border: 1px solid #ddd; padding: 8px; }}
            .plot-container {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Data Analysis Report</h1>
            <p><em>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
            
            <h2>1. Dataset Overview</h2>
            {df.head().to_html(classes='stats-table')}
            
            <h2>2. Statistical Summary</h2>
            {df.describe().to_html(classes='stats-table')}
            
            <h2>3. Missing Values Analysis</h2>
            {df.isnull().sum().to_frame().to_html(classes='stats-table')}
    """
    
    # Add plots to the report
    for plot_type, col_name, fig in plots:
        # Save plot to base64 string
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        
        html_content += f"""
            <div class="plot-container">
                <h3>{plot_type.title()} Plot: {col_name}</h3>
                <img src="data:image/png;base64,{img_str}" alt="{plot_type} plot">
            </div>
        """
    
    # Add model performance if available
    if trained_model and 'evaluation_metrics' in st.session_state:
        html_content += f"""
            <h2>4. Model Performance</h2>
            <pre>{str(st.session_state['evaluation_metrics'])}</pre>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    return html_content