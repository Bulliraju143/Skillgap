import streamlit as st
import pandas as pd
import json
import os
import tempfile
from io import BytesIO
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import numpy as np

# Add your src directory to path if it exists
if os.path.exists('./src'):
    sys.path.append('./src')

# Import your modular functions with error handling
def safe_import():
    try:
        from src.section_normalizer import normalize_text as full_normalize
        from src.remove_personal import remove_personal
        from src.skill_extractor import extract_skills
        from src.file_readers_txt import read_txt
        from src.file_readers_docx import read_docx
        from src.file_readers_pdf import read_pdf
        return True, (full_normalize, remove_personal, extract_skills, read_txt, read_docx, read_pdf)
    except ImportError as e:
        return False, str(e)

# Check if modules are available
modules_available, modules_or_error = safe_import()

# Page configuration
st.set_page_config(
    page_title="AI Skill Gap Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS with modern dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        --warning-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --dark-bg: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        --glass-bg: rgba(255, 255, 255, 0.1);
        --glass-border: rgba(255, 255, 255, 0.2);
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.8);
        --shadow-primary: 0 8px 32px rgba(0, 0, 0, 0.3);
        --shadow-hover: 0 16px 48px rgba(0, 0, 0, 0.4);
    }
    
    * {
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stApp {
        background: var(--dark-bg);
        min-height: 100vh;
        position: relative;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 80%, rgba(102, 126, 234, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(240, 147, 251, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(79, 172, 254, 0.08) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
    }
    
    .main .block-container {
        padding: 1.5rem 2rem 3rem;
        max-width: none !important;
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        margin: 1rem;
        box-shadow: var(--shadow-primary);
        border: 1px solid var(--glass-border);
    }
    
    /* Enhanced Header */
    .main-header {
        background: var(--primary-gradient);
        padding: 4rem 2rem;
        border-radius: 25px;
        color: var(--text-primary);
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: var(--shadow-primary);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { transform: rotate(0deg); }
        50% { transform: rotate(180deg); }
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #fff, #e8f4f8, #fff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
        z-index: 2;
    }
    
    .main-header p {
        font-size: 1.4rem;
        opacity: 0.95;
        margin: 0;
        position: relative;
        z-index: 2;
        font-weight: 300;
    }
    
    /* Modern Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--glass-bg);
        border-radius: 25px;
        padding: 8px;
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        box-shadow: var(--shadow-primary);
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-secondary);
        font-weight: 600;
        border-radius: 20px;
        padding: 16px 32px;
        margin: 4px;
        font-size: 1.1rem;
        border: none;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.1);
        color: var(--text-primary);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-gradient);
        color: var(--text-primary) !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
    
    /* Enhanced Cards */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(25px);
        border: 1px solid var(--glass-border);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: var(--shadow-primary);
        margin: 1.5rem 0;
        color: var(--text-primary);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.05) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .glass-card:hover {
        transform: translateY(-8px);
        box-shadow: var(--shadow-hover);
        background: rgba(255,255,255,0.15);
    }
    
    .upload-card {
        background: var(--glass-bg);
        backdrop-filter: blur(25px);
        border: 1px solid var(--glass-border);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: var(--shadow-primary);
        margin: 1.5rem 0;
        color: var(--text-primary);
        position: relative;
        overflow: hidden;
    }
    
    .upload-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-hover);
    }
    
    .upload-card h3 {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .success-card {
        background: var(--success-gradient);
        color: var(--text-primary);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(17, 153, 142, 0.4);
        margin: 1.5rem 0;
        border: none;
        position: relative;
        overflow: hidden;
    }
    
    .warning-card {
        background: var(--warning-gradient);
        color: var(--text-primary);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(240, 147, 251, 0.4);
        margin: 1.5rem 0;
        border: none;
    }
    
    .error-card {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: var(--text-primary);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(255, 65, 108, 0.4);
        margin: 1.5rem 0;
        border: none;
    }
    
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: var(--text-primary);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
        margin: 1.5rem 0;
        border: none;
    }
    
    /* Enhanced Skills Container */
    .skill-container {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 2rem;
        padding: 2rem;
        background: var(--glass-bg);
        border-radius: 20px;
        backdrop-filter: blur(15px);
        border: 1px solid var(--glass-border);
        box-shadow: var(--shadow-primary);
    }
    
    .skill-tag {
        background: var(--accent-gradient);
        color: var(--text-primary);
        padding: 12px 20px;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        border: 1px solid rgba(255,255,255,0.2);
        cursor: default;
        white-space: nowrap;
    }
    
    .skill-tag:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.5);
        background: var(--secondary-gradient);
    }
    
    /* Section Headers */
    .section-header {
        color: var(--text-primary);
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 2rem;
        text-align: center;
        background: var(--glass-bg);
        padding: 2.5rem;
        border-radius: 25px;
        backdrop-filter: blur(15px);
        border: 1px solid var(--glass-border);
        box-shadow: var(--shadow-primary);
        position: relative;
    }
    
    .section-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: var(--primary-gradient);
        opacity: 0.1;
        border-radius: 25px;
        pointer-events: none;
    }
    
    /* Enhanced Metrics */
    [data-testid="metric-container"] {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border) !important;
        padding: 2rem !important;
        border-radius: 20px !important;
        box-shadow: var(--shadow-primary) !important;
        color: var(--text-primary);
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-hover) !important;
        background: rgba(255,255,255,0.15) !important;
    }
    
    [data-testid="metric-container"] label {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: var(--text-primary) !important;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
    }
    
    /* Enhanced Input Fields */
    .stTextInput input, .stTextArea textarea {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 15px !important;
        color: var(--text-primary) !important;
        backdrop-filter: blur(10px);
        font-size: 1rem !important;
        padding: 15px 20px !important;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: rgba(102, 126, 234, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
    }
    
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: var(--text-secondary) !important;
    }
    
    .stTextInput label, .stTextArea label {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    /* Enhanced File Uploader */
    [data-testid="stFileUploader"] {
        background: var(--glass-bg);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(15px);
        border: 2px dashed var(--glass-border);
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(102, 126, 234, 0.6);
        background: rgba(255,255,255,0.15);
        transform: translateY(-2px);
    }
    
    /* Enhanced Buttons */
    .stButton button {
        background: var(--primary-gradient) !important;
        color: var(--text-primary) !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 1rem 2.5rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6) !important;
        background: var(--secondary-gradient) !important;
    }
    
    .stDownloadButton button {
        background: var(--success-gradient) !important;
        border-radius: 20px !important;
        padding: 0.8rem 2rem !important;
        color: var(--text-primary) !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4) !important;
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(17, 153, 142, 0.6) !important;
    }
    
    /* Enhanced Progress Bar */
    .stProgress > div > div > div {
        background: var(--primary-gradient) !important;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    
    .stProgress > div > div {
        background: rgba(255,255,255,0.2) !important;
        border-radius: 10px;
        backdrop-filter: blur(5px);
    }
    
    /* Enhanced Expander */
    .streamlit-expanderHeader {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(20px);
        border-radius: 15px !important;
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        border: 1px solid var(--glass-border) !important;
        padding: 1.2rem 1.5rem !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255,255,255,0.15) !important;
        transform: translateY(-1px);
    }
    
    .streamlit-expanderContent {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(20px);
        border-radius: 0 0 15px 15px !important;
        border: 1px solid var(--glass-border) !important;
        border-top: none !important;
        color: var(--text-primary);
        padding: 2rem !important;
    }
    
    /* Enhanced Messages */
    .stSuccess, .stError, .stInfo, .stWarning {
        border-radius: 15px !important;
        backdrop-filter: blur(10px);
        border: none !important;
        box-shadow: var(--shadow-primary) !important;
    }
    
    /* Chart Styling */
    .js-plotly-plot {
        background: rgba(255,255,255,0.98) !important;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: var(--shadow-primary);
        border: 1px solid var(--glass-border);
    }
    
    /* Enhanced Dataframe */
    .stDataFrame {
        background: var(--glass-bg);
        border-radius: 15px;
        backdrop-filter: blur(15px);
        border: 1px solid var(--glass-border);
        overflow: hidden;
        box-shadow: var(--shadow-primary);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-gradient);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--secondary-gradient);
    }
    
    /* Radio Buttons */
    .stRadio > div {
        background: var(--glass-bg);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid var(--glass-border);
    }
    
    .stRadio label {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
</style>
""", unsafe_allow_html=True)

def render_header():
    """Render the main header with enhanced styling"""
    st.markdown("""
    <div class="main-header">
        <h1>🎯 AI Skill Gap Analyzer</h1>
        <p>Advanced Resume Analysis & Skill Extraction Platform</p>
    </div>
    """, unsafe_allow_html=True)

def create_skill_tags(skills, max_display=30):
    """Create modern skill tags with hover effects"""
    if not skills:
        return """
        <div class="warning-card">
            <h4>⚠️ No Skills Detected</h4>
            <p>No skills were extracted from this document. This could be due to:</p>
            <ul>
                <li>Document formatting issues</li>
                <li>Skills not in our recognition database</li>
                <li>Non-technical content</li>
            </ul>
        </div>
        """
    
    display_skills = skills[:max_display] if len(skills) > max_display else skills
    remaining = len(skills) - len(display_skills)
    
    tags_html = '<div class="skill-container">'
    
    for skill in sorted(display_skills):
        tags_html += f'<div class="skill-tag">{skill}</div>'
    
    if remaining > 0:
        tags_html += f'<div class="skill-tag" style="background: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%);">+{remaining} more</div>'
    
    tags_html += '</div>'
    return tags_html

def safe_convert_value(value):
    """Safely convert numpy/pandas values to Python native types"""
    if value is None:
        return 0
    
    # Handle numpy arrays and scalars
    if hasattr(value, 'item'):
        return value.item()
    elif hasattr(value, 'tolist'):
        return value.tolist()
    elif isinstance(value, (np.integer, np.floating)):
        return value.item()
    elif isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        else:
            return value.tolist()
    # Handle pandas data
    elif hasattr(value, 'iloc') or hasattr(value, 'values'):
        if hasattr(value, 'values'):
            return safe_convert_value(value.values)
        return value
    # Handle lists and other iterables
    elif isinstance(value, (list, tuple)):
        return [safe_convert_value(item) for item in value]
    
    # Return as-is for basic Python types
    return value

def create_analysis_charts(processed_docs):
    """Create improved interactive charts for analysis results"""
    if not processed_docs:
        return None
    
    successful_docs = [doc for doc in processed_docs if doc.get('success', False)]
    
    if not successful_docs:
        return None
    
    try:
        # Prepare data with safe conversion
        doc_names = []
        skill_counts = []
        doc_types = []
        word_counts = []
        
        for doc in successful_docs:
            # Safely convert all values
            name = str(doc.get('file_name', 'Unknown'))
            # Truncate long names for better display
            display_name = name[:12] + '...' if len(name) > 15 else name
            doc_names.append(display_name)
            
            skills = safe_convert_value(doc.get('extracted_skills', []))
            skill_count = len(skills) if isinstance(skills, list) else 0
            skill_counts.append(skill_count)
            
            doc_types.append(str(doc.get('document_type', 'unknown')))
            
            words = safe_convert_value(doc.get('word_count', 0))
            word_counts.append(int(words) if isinstance(words, (int, float)) else 0)
        
        # Create a more readable chart layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Skills Extracted per Document', 
                'Skills vs Document Length', 
                'Document Type Distribution', 
                'Document Processing Summary'
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "pie"}, {"type": "bar"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Color palette
        colors = ['#667eea', '#f093fb', '#4facfe', '#43e97b', '#fa709a', '#764ba2', '#f093fb']
        
        # 1. Skills by Document - Horizontal bar for better readability
        fig.add_trace(
            go.Bar(
                y=doc_names,
                x=skill_counts,
                name="Skills Count",
                marker_color=[colors[i % len(colors)] for i in range(len(doc_names))],
                showlegend=False,
                text=[f"{count} skills" for count in skill_counts],
                textposition='auto',
                orientation='h'
            ),
            row=1, col=1
        )
        
        # 2. Scatter plot with better labels
        hover_text = [f"{name}<br>Skills: {skills}<br>Words: {words}" 
                     for name, skills, words in zip(doc_names, skill_counts, word_counts)]
        
        fig.add_trace(
            go.Scatter(
                x=word_counts,
                y=skill_counts,
                mode='markers+text',
                text=doc_names,
                textposition="top center",
                hovertext=hover_text,
                hoverinfo='text',
                marker=dict(
                    size=[max(8, min(20, count/2)) for count in skill_counts],
                    color=skill_counts,
                    colorscale='Viridis',
                    opacity=0.8,
                    line=dict(width=2, color='white')
                ),
                name="Skills vs Length",
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Document types pie chart with better colors
        type_counts = {}
        for doc_type in doc_types:
            clean_type = doc_type.replace('_', ' ').title()
            type_counts[clean_type] = type_counts.get(clean_type, 0) + 1
        
        fig.add_trace(
            go.Pie(
                labels=list(type_counts.keys()),
                values=list(type_counts.values()),
                showlegend=True,
                hole=0.4,
                marker_colors=colors[:len(type_counts)],
                textinfo='label+percent',
                textfont_size=12
            ),
            row=2, col=1
        )
        
        # 4. Word counts with better formatting
        fig.add_trace(
            go.Bar(
                x=doc_names,
                y=word_counts,
                name="Word Count",
                marker_color='#43e97b',
                showlegend=False,
                text=[f"{count:,}" for count in word_counts],
                textposition='outside',
                textangle=0
            ),
            row=2, col=2
        )
        
        # Update layout for better readability
        fig.update_layout(
            height=800,
            title={
                'text': "📊 Document Analysis Dashboard",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'family': 'Poppins, sans-serif'}
            },
            showlegend=False,
            template="plotly_white",
            plot_bgcolor='rgba(248,249,250,0.98)',
            paper_bgcolor='rgba(255,255,255,0.98)',
            font=dict(family="Poppins, sans-serif", size=11),
            margin=dict(t=80, b=60, l=60, r=60)
        )
        
        # Update individual subplot layouts
        fig.update_xaxes(title_text="Skills Count", row=1, col=1, title_font_size=12)
        fig.update_yaxes(title_text="Documents", row=1, col=1, title_font_size=12)
        
        fig.update_xaxes(title_text="Word Count", row=1, col=2, title_font_size=12)
        fig.update_yaxes(title_text="Skills Extracted", row=1, col=2, title_font_size=12)
        
        fig.update_xaxes(title_text="Documents", row=2, col=2, title_font_size=12, tickangle=45)
        fig.update_yaxes(title_text="Word Count", row=2, col=2, title_font_size=12)
        
        # Add annotations for better understanding
        fig.add_annotation(
            text="Larger bubbles = more skills",
            xref="x2", yref="y2",
            x=max(word_counts) * 0.7, y=max(skill_counts) * 0.9,
            showarrow=False,
            font=dict(size=10, color="gray")
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating charts: {str(e)}")
        # Return a simple fallback chart
        try:
            simple_fig = go.Figure()
            simple_fig.add_bar(
                x=doc_names,
                y=skill_counts,
                name="Skills Extracted",
                marker_color='#667eea'
            )
            simple_fig.update_layout(
                title="Skills Extracted by Document",
                xaxis_title="Documents",
                yaxis_title="Skills Count",
                template="plotly_white",
                font=dict(family="Poppins, sans-serif")
            )
            return simple_fig
        except:
            return None

def mock_text_processing(text, doc_name, doc_type):
    """Mock processing function when modules are not available"""
    # Simple skill extraction using keywords
    common_skills = [
        'Python', 'Java', 'JavaScript', 'HTML', 'CSS', 'React', 'Node.js', 'SQL',
        'Machine Learning', 'Data Analysis', 'Project Management', 'Communication',
        'Leadership', 'Problem Solving', 'Team Work', 'Git', 'Docker', 'AWS',
        'MongoDB', 'PostgreSQL', 'Excel', 'PowerBI', 'Tableau', 'Agile', 'Scrum'
    ]
    
    found_skills = []
    text_lower = text.lower()
    
    for skill in common_skills:
        if skill.lower() in text_lower:
            found_skills.append(skill)
    
    # Mock text cleaning (simple version)
    cleaned_text = text.replace('\n\n', '\n').strip()
    
    return {
        'file_name': doc_name,
        'document_type': doc_type,
        'success': True,
        'error': None,
        'original_text': text,
        'cleaned_text': cleaned_text,
        'extracted_skills': found_skills,
        'original_length': len(text),
        'final_length': len(cleaned_text),
        'reduction_percentage': ((len(text) - len(cleaned_text)) / len(text) * 100) if len(text) > 0 else 0,
        'word_count': len(cleaned_text.split()),
    }

def process_text_input(text, doc_name, doc_type):
    """Process text input through pipeline or mock processing"""
    if not text or not text.strip():
        return {
            'file_name': doc_name,
            'document_type': doc_type,
            'success': False,
            'error': 'No text content provided',
            'extracted_skills': [],
        }
    
    try:
        if modules_available:
            full_normalize, remove_personal, extract_skills, _, _, _ = modules_or_error
            
            # Use real modules
            cleaned_text = remove_personal(text)
            cleaned_text = full_normalize(cleaned_text)
            
            skills_file_path = "skills_list.txt"
            if os.path.exists(skills_file_path):
                extracted_skills = extract_skills(cleaned_text, skills_file_path)
            else:
                # Fallback to mock if skills file not found
                return mock_text_processing(text, doc_name, doc_type)
            
            original_length = len(text)
            final_length = len(cleaned_text)
            reduction = ((original_length - final_length) / original_length * 100) if original_length > 0 else 0
            
            return {
                'file_name': doc_name,
                'document_type': doc_type,
                'success': True,
                'error': None,
                'original_text': text,
                'cleaned_text': cleaned_text,
                'extracted_skills': list(extracted_skills) if extracted_skills else [],
                'original_length': original_length,
                'final_length': final_length,
                'reduction_percentage': reduction,
                'word_count': len(cleaned_text.split()),
            }
        else:
            # Use mock processing
            return mock_text_processing(text, doc_name, doc_type)
            
    except Exception as e:
        return {
            'file_name': doc_name,
            'document_type': doc_type,
            'success': False,
            'error': f"Processing error: {str(e)}",
            'extracted_skills': [],
        }

def process_uploaded_file(uploaded_file, doc_type):
    """Process uploaded file with error handling"""
    try:
        file_content = uploaded_file.getvalue()
        file_name = uploaded_file.name
        file_extension = file_name.split('.')[-1].lower()
        
        # Try to read file content
        if file_extension == 'txt':
            text = file_content.decode('utf-8')
        elif modules_available and file_extension in ['pdf', 'docx']:
            # Use real file readers if available
            _, _, _, read_txt, read_docx, read_pdf = modules_or_error
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            
            try:
                if file_extension == 'pdf':
                    text = read_pdf(tmp_path)
                elif file_extension == 'docx':
                    text = read_docx(tmp_path)
                else:
                    text = "Unsupported file format"
                    
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            # Mock file reading for unsupported formats or when modules unavailable
            text = f"Mock content extracted from {file_name}. This is a sample text with skills like Python, JavaScript, Machine Learning, and Project Management."
            
        return process_text_input(text, file_name, doc_type)
        
    except Exception as e:
        return {
            'file_name': uploaded_file.name,
            'document_type': doc_type,
            'success': False,
            'error': f"File processing error: {str(e)}",
            'extracted_skills': [],
        }

def display_summary_stats(processed_docs):
    """Display comprehensive summary statistics"""
    successful_docs = [doc for doc in processed_docs if doc.get('success', False)]
    failed_docs = [doc for doc in processed_docs if not doc.get('success', False)]
    
    # Safely calculate total skills
    total_skills = 0
    unique_skills = set()
    for doc in successful_docs:
        skills = safe_convert_value(doc.get('extracted_skills', []))
        if isinstance(skills, list):
            total_skills += len(skills)
            unique_skills.update(skills)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Documents Processed",
            value=len(successful_docs),
            delta=f"{len(failed_docs)} failed" if failed_docs else "All successful",
            delta_color="inverse" if failed_docs else "normal"
        )
    
    with col2:
        st.metric(
            label="🎯 Total Skills Found",
            value=total_skills,
            delta=f"Avg: {total_skills/len(successful_docs):.1f} per doc" if successful_docs else "0"
        )
    
    with col3:
        st.metric(
            label="🔍 Unique Skills",
            value=len(unique_skills),
            delta=f"{(len(unique_skills)/total_skills*100):.1f}% diversity" if total_skills > 0 else "0%"
        )
    
    with col4:
        avg_reduction = 0
        if successful_docs:
            reductions = [safe_convert_value(doc.get('reduction_percentage', 0)) for doc in successful_docs]
            avg_reduction = sum(reductions) / len(successful_docs)
        
        st.metric(
            label="🧹 Avg Text Reduction",
            value=f"{avg_reduction:.1f}%",
            delta="Processing efficiency"
        )

def main():
    render_header()
    
    # Show module status
    if not modules_available:
        st.warning(f"⚠️ Some modules are not available: {modules_or_error}. Using mock processing for demonstration.")
    
    # Create enhanced tabs
    tab1, tab2, tab3 = st.tabs(["📤 Document Upload", "📊 Analysis Results", "📈 Advanced Analytics"])
    
    with tab1:
        st.markdown('<h2 class="section-header">Document Processing Center</h2>', unsafe_allow_html=True)
        
        # File upload section
        upload_col, jd_col = st.columns([1, 1])
        
        with upload_col:
            st.markdown("""
            <div class="upload-card">
                <h3>📄 Resume Upload</h3>
                <p>Upload resume files for comprehensive skill analysis. Our AI extracts and categorizes technical and professional skills.</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_resume_files = st.file_uploader(
                "Select Resume Files",
                type=['pdf', 'docx', 'txt'],
                accept_multiple_files=True,
                key="resume_uploader",
                help="Supported formats: PDF, DOCX, TXT"
            )

        with jd_col:
            st.markdown("""
            <div class="upload-card">
                <h3>💼 Job Description</h3>
                <p>Provide job description to identify skill gaps and matching opportunities.</p>
            </div>
            """, unsafe_allow_html=True)
            
            jd_input_mode = st.radio(
                "Choose input method:",
                ('Paste Text', 'Upload File'),
                index=0,
                horizontal=True,
                key="jd_mode_radio"
            )
            
            jd_name = st.text_input(
                "Job Description Name",
                "Target Position Analysis",
                key="jd_name_input"
            )
            
            if jd_input_mode == 'Paste Text':
                jd_text = st.text_area(
                    "Job Description Content",
                    height=250,
                    key="jd_text_area",
                    placeholder="Paste the complete job description here..."
                )
                jd_file = None
            else:
                jd_file = st.file_uploader(
                    "Upload Job Description File",
                    type=['pdf', 'docx', 'txt'],
                    accept_multiple_files=False,
                    key="jd_file_uploader"
                )
                jd_text = None

        # Process documents
        st.markdown("---")
        
        # Collect documents to process
        all_documents = []
        
        # Add job description
        if jd_file:
            all_documents.append(('file', jd_file, jd_name, 'job_description'))
        elif jd_text and jd_text.strip():
            all_documents.append(('text', jd_text, jd_name, 'job_description'))

        # Add resumes
        if uploaded_resume_files:
            for file in uploaded_resume_files:
                all_documents.append(('file', file, file.name, 'resume'))
        
        if all_documents:
            num_resumes = len([d for d in all_documents if d[3] == 'resume'])
            num_jd = len([d for d in all_documents if d[3] == 'job_description'])
            
            st.markdown(f'''
            <div class="success-card">
                <h4>🎯 Ready for Processing</h4>
                <p><strong>{num_resumes}</strong> Resume(s) and <strong>{num_jd}</strong> Job Description(s) loaded</p>
                <p>Click below to start AI-powered analysis</p>
            </div>
            ''', unsafe_allow_html=True)
            
            if st.button("🚀 Start AI Analysis", type="primary", key="process_btn"):
                st.session_state.processed_docs = []
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (doc_type, doc_content, doc_name, category) in enumerate(all_documents):
                    progress = (i + 1) / len(all_documents)
                    progress_bar.progress(progress)
                    status_text.info(f"🔄 Processing: {doc_name} ({category})")
                    
                    if doc_type == 'text':
                        result = process_text_input(doc_content, doc_name, category)
                    else:  # file
                        result = process_uploaded_file(doc_content, category)
                    
                    st.session_state.processed_docs.append(result)
                
                progress_bar.empty()
                status_text.success("✅ Analysis completed successfully!")
                st.rerun()
        else:
            st.markdown('''
            <div class="info-card">
                <h4>📋 Getting Started</h4>
                <p>Upload resume files and/or job description to begin analysis.</p>
                <ul>
                    <li>Multiple resume formats supported</li>
                    <li>Batch processing capabilities</li>
                    <li>AI-powered skill extraction</li>
                    <li>Comprehensive gap analysis</li>
                </ul>
            </div>
            ''', unsafe_allow_html=True)

    with tab2:
        if 'processed_docs' in st.session_state and st.session_state.processed_docs:
            st.markdown('<h2 class="section-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
            
            display_summary_stats(st.session_state.processed_docs)
            st.markdown("---")
            
            # Display results
            successful_docs = [doc for doc in st.session_state.processed_docs if doc.get('success', False)]
            failed_docs = [doc for doc in st.session_state.processed_docs if not doc.get('success', False)]
            
            if successful_docs:
                for i, doc in enumerate(successful_docs):
                    doc_icon = "💼" if doc.get('document_type') == 'job_description' else "📄"
                    
                    with st.expander(f"{doc_icon} {doc['file_name']} - Analysis Complete", expanded=i==0):
                        
                        # Safely convert metrics
                        original_len = safe_convert_value(doc.get('original_length', 0))
                        final_len = safe_convert_value(doc.get('final_length', 0))
                        reduction_pct = safe_convert_value(doc.get('reduction_percentage', 0))
                        word_count = safe_convert_value(doc.get('word_count', 0))
                        
                        # Metrics
                        st.markdown("#### 📊 Processing Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        col1.metric("Original Size", f"{int(original_len):,} chars")
                        col2.metric("Processed Size", f"{int(final_len):,} chars")
                        col3.metric("Reduction", f"{float(reduction_pct):.1f}%")
                        col4.metric("Words", f"{int(word_count):,}")
                        
                        st.markdown("---")
                        
                        # Skills
                        st.markdown("#### 🎯 Extracted Skills")
                        skills = safe_convert_value(doc.get('extracted_skills', []))
                        if skills and isinstance(skills, list) and len(skills) > 0:
                            st.metric("Skills Identified", len(skills))
                            st.markdown(create_skill_tags(skills), unsafe_allow_html=True)
                        else:
                            st.markdown('''
                            <div class="warning-card">
                                <h4>⚠️ No Skills Detected</h4>
                                <p>No specific skills were identified in this document.</p>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Text comparison
                        st.markdown("#### 🔍 Text Comparison")
                        text_col1, text_col2 = st.columns(2)
                        
                        with text_col1:
                            st.markdown("**📄 Original Text (Preview)**")
                            original_text = str(doc.get('original_text', ''))
                            original_preview = original_text[:500] + "..." if len(original_text) > 500 else original_text
                            st.code(original_preview, language='text')
                        
                        with text_col2:
                            st.markdown("**✨ Processed Text (Preview)**")
                            cleaned_text = str(doc.get('cleaned_text', ''))
                            cleaned_preview = cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text
                            st.code(cleaned_preview, language='text')

                # Export results
                st.markdown("---")
                st.markdown("#### 📥 Export Results")
                
                export_data = []
                for doc in successful_docs:
                    # Safely convert all data
                    skills = safe_convert_value(doc.get('extracted_skills', []))
                    if not isinstance(skills, list):
                        skills = []
                    
                    filename = str(doc.get('file_name', 'Unknown'))
                    doc_type = str(doc.get('document_type', 'unknown'))
                    word_count = int(safe_convert_value(doc.get('word_count', 0)))
                    
                    export_data.append({
                        'filename': filename,
                        'document_type': doc_type,
                        'skills_extracted': skills,
                        'total_skills': len(skills),
                        'word_count': word_count,
                        'analysis_timestamp': datetime.now().isoformat()
                    })
                
                if export_data:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        try:
                            json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
                            st.download_button(
                                label="📋 Download Detailed Results (JSON)",
                                data=json_data,
                                file_name=f"skill_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        except Exception as e:
                            st.error(f"Error creating JSON export: {str(e)}")
                    
                    with col2:
                        try:
                            # CSV summary with safe data conversion
                            df_data = []
                            for item in export_data:
                                skills_list = item.get('skills_extracted', [])
                                if isinstance(skills_list, np.ndarray):
                                    skills_list = skills_list.tolist()
                                skills_list = [str(skill.item() if hasattr(skill, "item") else skill) for skill in skills_list]
                                top_skills = ', '.join(str(skill) for skill in skills_list[:5]) if skills_list else 'None'
                                
                                df_data.append({
                                    'Document': str(item.get('filename', 'Unknown')),
                                    'Type': str(item.get('document_type', 'Unknown')),
                                    'Skills Found': int(item.get('total_skills', 0)),
                                    'Word Count': int(item.get('word_count', 0)),
                                    'Top Skills': top_skills
                                })
                            
                            if df_data:
                                df = pd.DataFrame(df_data)
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    label="📊 Download Summary (CSV)",
                                    data=csv,
                                    file_name=f"skill_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                        except Exception as e:
                            st.error(f"Error creating CSV: {str(e)}")

            # Show failed documents
            if failed_docs:
                st.markdown("#### ❌ Processing Issues")
                for doc in failed_docs:
                    st.markdown(f'''
                    <div class="error-card">
                        <h4>🚫 Processing Failed</h4>
                        <p><strong>Document:</strong> {doc.get("file_name", "Unknown")}</p>
                        <p><strong>Error:</strong> {doc.get("error", "Unknown error")}</p>
                    </div>
                    ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="info-card">
                <h4>📊 Results Dashboard</h4>
                <p>Analysis results will appear here after processing documents.</p>
                <p>Features include:</p>
                <ul>
                    <li>📈 Processing metrics and statistics</li>
                    <li>🎯 Visual skill extraction results</li>
                    <li>🔍 Text comparison views</li>
                    <li>📥 Export capabilities</li>
                </ul>
            </div>
            ''', unsafe_allow_html=True)

    with tab3:
        if 'processed_docs' in st.session_state and st.session_state.processed_docs:
            st.markdown('<h2 class="section-header">📈 Advanced Analytics</h2>', unsafe_allow_html=True)
            
            # Create charts
            chart_fig = create_analysis_charts(st.session_state.processed_docs)
            if chart_fig:
                st.plotly_chart(chart_fig, use_container_width=True, config={'displayModeBar': True})
            else:
                st.warning("Unable to generate charts. Please check your data.")
            
            # Advanced analysis
            successful_docs = [doc for doc in st.session_state.processed_docs if doc.get('success', False)]
            
            if len(successful_docs) > 1:
                st.markdown("---")
                st.markdown("#### 🔗 Cross-Document Analysis")
                
                # Skill overlap analysis with safe conversion
                all_skills_by_doc = {}
                for doc in successful_docs:
                    skills = safe_convert_value(doc.get('extracted_skills', []))
                    if isinstance(skills, list):
                        all_skills_by_doc[doc['file_name']] = set(skills)
                    else:
                        all_skills_by_doc[doc['file_name']] = set()
                
                if len(all_skills_by_doc) >= 2:
                    # Common skills
                    skill_sets = [skills for skills in all_skills_by_doc.values() if skills]
                    common_skills = set.intersection(*skill_sets) if skill_sets else set()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('''
                        <div class="glass-card">
                            <h4>🎯 Universal Skills</h4>
                            <p>Skills found across all documents</p>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        st.metric("Common Skills", len(common_skills))
                        if common_skills:
                            st.markdown(create_skill_tags(list(common_skills)), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('''
                        <div class="glass-card">
                            <h4>📊 Skill Distribution</h4>
                            <p>Analysis breakdown</p>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        total_skills = sum(len(skills) for skills in all_skills_by_doc.values())
                        unique_skills = len(set().union(*all_skills_by_doc.values())) if all_skills_by_doc else 0
                        
                        st.metric("Unique Skills Pool", unique_skills)
                        if successful_docs:
                            st.metric("Average per Document", f"{total_skills/len(successful_docs):.1f}")
                
                # Gap analysis for JD vs Resume
                jd_docs = [doc for doc in successful_docs if doc.get('document_type') == 'job_description']
                resume_docs = [doc for doc in successful_docs if doc.get('document_type') == 'resume']
                
                if jd_docs and resume_docs:
                    jd_skills = set()
                    resume_skills = set()
                    
                    for doc in jd_docs:
                        skills = safe_convert_value(doc.get('extracted_skills', []))
                        if isinstance(skills, list):
                            jd_skills.update(skills)
                    
                    for doc in resume_docs:
                        skills = safe_convert_value(doc.get('extracted_skills', []))
                        if isinstance(skills, list):
                            resume_skills.update(skills)
                    
                    overlap = len(jd_skills.intersection(resume_skills))
                    jd_only = len(jd_skills - resume_skills)
                    resume_only = len(resume_skills - jd_skills)
                    
                    st.markdown("---")
                    st.markdown("#### 🎯 Skill Gap Analysis")
                    
                    gap_col1, gap_col2, gap_col3 = st.columns(3)
                    gap_col1.metric("Matching Skills", overlap, delta="Found in both")
                    gap_col2.metric("Missing Skills", jd_only, delta="Gap identified")
                    gap_col3.metric("Additional Skills", resume_only, delta="Bonus skills")
                    
                    if jd_only > 0:
                        missing_skills = list(jd_skills - resume_skills)
                        st.markdown("##### 🔍 Skills to Develop:")
                        st.markdown(create_skill_tags(missing_skills), unsafe_allow_html=True)
            
            # Processing statistics with safe data handling
            if successful_docs:
                st.markdown("---")
                st.markdown("#### 📊 Processing Statistics")
                
                try:
                    stats_data = []
                    for doc in successful_docs:
                        # Safely convert all values
                        skills = safe_convert_value(doc.get('extracted_skills', []))
                        skills_count = len(skills) if isinstance(skills, list) else 0
                        word_count = safe_convert_value(doc.get('word_count', 0))
                        reduction_pct = safe_convert_value(doc.get('reduction_percentage', 0))
                        
                        # Ensure all values are basic Python types
                        skills_count = int(skills_count)
                        word_count = int(word_count) if word_count else 0
                        reduction_pct = float(reduction_pct) if reduction_pct else 0.0
                        
                        skills_per_1000 = (skills_count / max(word_count, 1) * 1000) if word_count > 0 else 0
                        
                        stats_data.append({
                            'Document': str(doc.get('file_name', 'Unknown')),
                            'Type': str(doc.get('document_type', 'unknown')).replace('_', ' ').title(),
                            'Skills Count': skills_count,
                            'Word Count': word_count,
                            'Reduction %': f"{reduction_pct:.1f}%",
                            'Skills per 1000 words': f"{skills_per_1000:.1f}"
                        })
                    
                    if stats_data:
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                        
                except Exception as e:
                    st.error(f"Error creating statistics table: {str(e)}")
                    # Fallback simple display
                    st.markdown("**Document Summary:**")
                    for doc in successful_docs:
                        skills = safe_convert_value(doc.get('extracted_skills', []))
                        skills_count = len(skills) if isinstance(skills, list) else 0
                        word_count = safe_convert_value(doc.get('word_count', 0))
                        st.write(f"• **{doc.get('file_name', 'Unknown')}**: {skills_count} skills, {int(word_count) if word_count else 0} words")
                    
        else:
            st.markdown('''
            <div class="info-card">
                <h4>📈 Advanced Analytics</h4>
                <p>Comprehensive insights will be available after document processing:</p>
                <ul>
                    <li>📊 Interactive charts and visualizations</li>
                    <li>🔗 Cross-document skill analysis</li>
                    <li>🎯 Gap identification and recommendations</li>
                    <li>📈 Processing efficiency metrics</li>
                    <li>🔍 Statistical breakdowns</li>
                </ul>
            </div>
            ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()