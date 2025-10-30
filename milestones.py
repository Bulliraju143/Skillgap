""" 
Professional Skill Gap Analysis System
AI-Powered NLP Analysis with Advanced BERT Embeddings

Student Name: Miriyala Bulli Raju
Student ID: 21  

Complete System Including:
✅ Document Parsing & Text Extraction
✅ Multi-Method Skill Extraction (NER, POS, Context, Keyword, Noun Chunks)
✅ BERT Contextual Embeddings
✅ Custom NER Training
✅ Semantic Similarity Analysis
✅ Advanced BERT-based Gap Analysis
✅ Cosine Similarity Matrices
✅ Learning Path Recommendations
✅ Interactive Dashboard & Analytics
✅ Multiple Export Formats
"""

import streamlit as st

# CRITICAL: Page Config MUST BE FIRST
st.set_page_config(
    page_title="Skill Gap Analysis System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modules AFTER set_page_config
import torch
import time
import json
import random
from pathlib import Path
from typing import Optional, Tuple, Set, Dict, List
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Import custom modules
# from pipeline import DocumentParser, get_parser
# from txt_cleaner import TextCleaner
from parser_pipeline import DocumentParser, get_parser
from file_readers_txt import read_txt
from file_readers_docx import read_docx
from file_readers_pdf import read_pdf
from txt_cleaner import normalize_text
from remove_personal import remove_personal

# Import milestone2 components
from skillextraction_helpers import (
    AdvancedSkillExtractor,
    SkillGapAnalyzer as M2SkillGapAnalyzer,
    SkillDatabase,
    BERTSkillExtractor,
    SkillAnnotator,
    CustomNERTrainer,
    create_skill_visualization,
    create_skill_comparison_chart,
    create_category_breakdown_chart,
    create_extraction_method_chart,
    load_spacy_model,
    load_sentence_transformer,
    load_bert_model,
    export_analysis_report,
    export_to_json,
    export_to_csv
)

# Import milestone3 components
from sentenceBERT import (
    SentenceBERTEncoder,
    SimilarityCalculator,
    SkillGapAnalyzer as M3SkillGapAnalyzer,
    SkillRanker,
    GapVisualizer,
    ReportGenerator,
    LearningPathGenerator,
    GapAnalysisResult,
    SkillMatch
)

# Custom CSS
# def load_custom_css():
#     """Load enhanced custom CSS with premium animations"""
#     st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@300;400;500;600;700;800&display=swap');

#     :root {
#         --primary: #667eea;
#         --primary-dark: #5568d3;
#         --secondary: #764ba2;
#         --accent: #f093fb;
#         --success: #38ef7d;
#         --danger: #ff6b6b;
#         --warning: #feca57;
#         --glass-bg: rgba(255, 255, 255, 0.08);
#         --glass-border: rgba(255, 255, 255, 0.18);
#     }

#     .main {
#         font-family: 'Inter', sans-serif;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
#         min-height: 100vh;
#         padding: 1rem;
#     }

#     .custom-header {
#         background: var(--glass-bg);
#         backdrop-filter: blur(30px);
#         padding: 3rem 2rem;
#         border-radius: 24px;
#         margin-bottom: 2rem;
#         color: white;
#         text-align: center;
#         box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
#         border: 1px solid var(--glass-border);
#         position: relative;
#         overflow: hidden;
#     }

#     .custom-header::before {
#         content: '';
#         position: absolute;
#         top: -50%;
#         left: -50%;
#         width: 200%;
#         height: 200%;
#         background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
#         animation: shimmer 3s infinite;
#     }

#     @keyframes shimmer {
#         0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
#         100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
#     }

#     .glass-card {
#         background: var(--glass-bg);
#         backdrop-filter: blur(20px);
#         border: 1px solid var(--glass-border);
#         border-radius: 20px;
#         padding: 2rem;
#         margin: 1rem 0;
#         box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
#         transition: all 0.4s cubic-bezier(0.23, 1, 0.320, 1);
#     }

#     .glass-card:hover {
#         transform: translateY(-8px);
#         box-shadow: 0 20px 50px rgba(102, 126, 234, 0.4);
#     }

#     .skill-badge {
#         display: inline-block;
#         background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
#         color: white;
#         padding: 12px 24px;
#         margin: 8px 6px;
#         border-radius: 25px;
#         font-size: 14px;
#         font-weight: 600;
#         box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
#         border: 1px solid rgba(255, 255, 255, 0.2);
#         transition: all 0.3s ease;
#         cursor: pointer;
#         position: relative;
#         overflow: hidden;
#     }

#     .skill-badge::before {
#         content: '';
#         position: absolute;
#         top: 50%;
#         left: 50%;
#         width: 0;
#         height: 0;
#         border-radius: 50%;
#         background: rgba(255, 255, 255, 0.3);
#         transform: translate(-50%, -50%);
#         transition: width 0.6s, height 0.6s;
#     }

#     .skill-badge:hover::before {
#         width: 300px;
#         height: 300px;
#     }

#     .skill-badge:hover {
#         transform: translateY(-4px) scale(1.05);
#         box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
#     }

#     .skill-badge-matched {
#         background: linear-gradient(135deg, #38ef7d 0%, #11998e 100%);
#         box-shadow: 0 4px 15px rgba(56, 239, 125, 0.3);
#     }

#     .skill-badge-missing {
#         background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
#         box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
#     }

#     .skill-badge-extra {
#         background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
#         box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
#     }

#     .skill-category {
#         background: var(--glass-bg);
#         backdrop-filter: blur(15px);
#         border-radius: 16px;
#         padding: 1.5rem;
#         margin: 1rem 0;
#         border: 1px solid var(--glass-border);
#         animation: slideInUp 0.6s ease-out;
#     }

#     .skill-category-header {
#         display: flex;
#         align-items: center;
#         margin-bottom: 1rem;
#         font-size: 1.3rem;
#         font-weight: 700;
#         color: white;
#     }

#     .skill-category-icon {
#         font-size: 1.8rem;
#         margin-right: 12px;
#         animation: bounce 2s infinite;
#     }

#     @keyframes bounce {
#         0%, 100% { transform: translateY(0); }
#         50% { transform: translateY(-10px); }
#     }

#     @keyframes slideInUp {
#         from {
#             opacity: 0;
#             transform: translateY(30px);
#         }
#         to {
#             opacity: 1;
#             transform: translateY(0);
#         }
#     }

#     .metric-card {
#         background: var(--glass-bg);
#         backdrop-filter: blur(20px);
#         border-radius: 16px;
#         padding: 1.5rem;
#         text-align: center;
#         border: 1px solid var(--glass-border);
#         transition: all 0.3s ease;
#         position: relative;
#         overflow: hidden;
#     }

#     .metric-card::before {
#         content: '';
#         position: absolute;
#         top: 0;
#         left: 0;
#         right: 0;
#         height: 3px;
#         background: linear-gradient(90deg, var(--primary), var(--accent));
#     }

#     .metric-card:hover {
#         transform: translateY(-5px) scale(1.02);
#         box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
#     }

#     .stButton > button {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border: none;
#         border-radius: 12px;
#         padding: 1rem 2.5rem;
#         font-weight: 600;
#         font-size: 16px;
#         transition: all 0.3s ease;
#         box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
#         text-transform: uppercase;
#         letter-spacing: 1px;
#         position: relative;
#         overflow: hidden;
#     }

#     .stButton > button::before {
#         content: '';
#         position: absolute;
#         top: 0;
#         left: -100%;
#         width: 100%;
#         height: 100%;
#         background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
#         transition: left 0.5s;
#     }

#     .stButton > button:hover::before {
#         left: 100%;
#     }

#     .stButton > button:hover {
#         transform: translateY(-3px);
#         box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6);
#     }

#     .stTabs [data-baseweb="tab-list"] {
#         gap: 8px;
#         background: rgba(0, 0, 0, 0.2);
#         padding: 8px;
#         border-radius: 16px;
#         backdrop-filter: blur(15px);
#     }

#     .stTabs [data-baseweb="tab"] {
#         background: transparent;
#         border-radius: 12px;
#         color: rgba(255, 255, 255, 0.7);
#         font-weight: 600;
#         padding: 12px 24px;
#         transition: all 0.3s ease;
#     }

#     .stTabs [aria-selected="true"] {
#         background: rgba(255, 255, 255, 0.15);
#         color: white;
#         box-shadow: 0 4px 20px rgba(255, 255, 255, 0.15);
#     }

#     .stTextArea textarea {
#         background: rgba(0, 0, 0, 0.3);
#         border: 1px solid var(--glass-border);
#         border-radius: 16px;
#         color: white;
#         font-family: 'JetBrains Mono', monospace;
#         font-size: 14px;
#         line-height: 1.6;
#         backdrop-filter: blur(10px);
#         transition: all 0.4s ease;
#     }

#     .stTextArea textarea:focus {
#         border-color: rgba(102, 126, 234, 0.8);
#         box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
#         outline: none;
#     }

#     ::-webkit-scrollbar {
#         width: 10px;
#     }

#     ::-webkit-scrollbar-track {
#         background: rgba(255, 255, 255, 0.1);
#         border-radius: 10px;
#     }

#     ::-webkit-scrollbar-thumb {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         border-radius: 10px;
#     }

#     ::-webkit-scrollbar-thumb:hover {
#         background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%);
#     }

#     .dashboard-card {
#         background: var(--glass-bg);
#         backdrop-filter: blur(20px);
#         border: 1px solid var(--glass-border);
#         border-radius: 16px;
#         padding: 1.5rem;
#         margin: 1rem 0;
#         box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
#         transition: all 0.3s ease;
#     }

#     .dashboard-card:hover {
#         transform: translateY(-4px);
#         box-shadow: 0 12px 40px rgba(102, 126, 234, 0.5);
#     }

#     .stat-box {
#         background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
#         border: 1px solid rgba(102, 126, 234, 0.3);
#         border-radius: 12px;
#         padding: 1.5rem;
#         text-align: center;
#         transition: all 0.3s ease;
#     }

#     .stat-box:hover {
#         transform: scale(1.05);
#         border-color: rgba(102, 126, 234, 0.6);
#     }

#     .stat-number {
#         font-size: 2.5rem;
#         font-weight: 700;
#         background: linear-gradient(135deg, #667eea 0%, #f093fb 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         background-clip: text;
#     }

#     .stat-label {
#         color: rgba(255, 255, 255, 0.8);
#         font-size: 0.9rem;
#         margin-top: 0.5rem;
#     }
#     </style>
#      """, unsafe_allow_html=True)
# def load_custom_css():
#     """Load enhanced custom CSS with perfect text visibility"""
#     st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@300;400;500;600;700;800&display=swap');

#     :root {
#         --primary: #667eea;
#         --primary-dark: #5568d3;
#         --secondary: #764ba2;
#         --accent: #f093fb;
#         --success: #38ef7d;
#         --danger: #ff6b6b;
#         --warning: #feca57;
#         --glass-bg: rgba(255, 255, 255, 0.15);
#         --glass-border: rgba(255, 255, 255, 0.3);
#     }

#     /* Universal text rendering optimization */
#     * {
#         text-rendering: optimizeLegibility;
#         -webkit-font-smoothing: antialiased;
#         -moz-osx-font-smoothing: grayscale;
#     }

#     /* Main container with gradient background */
#     .main {
#         font-family: 'Inter', sans-serif;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
#         min-height: 100vh;
#         padding: 1rem;
#     }

#     /* Ensure all text elements have good contrast */
#     body, div, span, p, a, li, td, th, label, .stMarkdown {
#         color: rgba(255, 255, 255, 0.95) !important;
#     }

#     /* All headings with better visibility */
#     h1, h2, h3, h4, h5, h6 {
#         color: white !important;
#         text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.4);
#         font-weight: 700;
#     }

#     /* Paragraph text visibility */
#     p {
#         color: rgba(255, 255, 255, 0.95) !important;
#         text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
#         line-height: 1.7;
#     }

#     /* Bold and strong text */
#     strong, b {
#         color: white !important;
#         font-weight: 700;
#         text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
#     }

#     /* List items */
#     li {
#         color: rgba(255, 255, 255, 0.95) !important;
#         text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
#         line-height: 1.8;
#     }

#     /* Custom header section */
#     .custom-header {
#         background: rgba(255, 255, 255, 0.15);
#         backdrop-filter: blur(30px);
#         padding: 3rem 2rem;
#         border-radius: 24px;
#         margin-bottom: 2rem;
#         color: white;
#         text-align: center;
#         box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
#         border: 1px solid rgba(255, 255, 255, 0.3);
#         position: relative;
#         overflow: hidden;
#     }

#     .custom-header::before {
#         content: '';
#         position: absolute;
#         top: -50%;
#         left: -50%;
#         width: 200%;
#         height: 200%;
#         background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
#         animation: shimmer 3s infinite;
#     }

#     @keyframes shimmer {
#         0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
#         100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
#     }

#     .custom-header h1 {
#         text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.5);
#         margin: 0 !important;
#         position: relative;
#         z-index: 1;
#     }

#     .custom-header p {
#         text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.4);
#         position: relative;
#         z-index: 1;
#         color: white !important;
#     }

#     /* Glass card effect with better visibility */
#     .glass-card {
#         background: rgba(255, 255, 255, 0.15);
#         backdrop-filter: blur(20px);
#         border: 1px solid rgba(255, 255, 255, 0.3);
#         border-radius: 20px;
#         padding: 2rem;
#         margin: 1rem 0;
#         box-shadow: 0 8px 32px rgba(31, 38, 135, 0.4);
#         transition: all 0.4s cubic-bezier(0.23, 1, 0.320, 1);
#     }

#     .glass-card:hover {
#         transform: translateY(-8px);
#         box-shadow: 0 20px 50px rgba(102, 126, 234, 0.5);
#         border-color: rgba(255, 255, 255, 0.4);
#     }

#     .glass-card h2, .glass-card h3, .glass-card h4 {
#         color: white !important;
#         text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.4);
#     }

#     .glass-card p, .glass-card span, .glass-card div {
#         color: rgba(255, 255, 255, 0.95) !important;
#         text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
#     }

#     /* Skill badges with perfect contrast */
#     .skill-badge {
#         display: inline-block;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white !important;
#         padding: 12px 24px;
#         margin: 8px 6px;
#         border-radius: 25px;
#         font-size: 14px;
#         font-weight: 600;
#         box-shadow: 0 4px 15px rgba(102, 126, 234, 0.5);
#         border: 1px solid rgba(255, 255, 255, 0.3);
#         transition: all 0.3s ease;
#         cursor: pointer;
#         position: relative;
#         overflow: hidden;
#         text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
#     }

#     .skill-badge::before {
#         content: '';
#         position: absolute;
#         top: 50%;
#         left: 50%;
#         width: 0;
#         height: 0;
#         border-radius: 50%;
#         background: rgba(255, 255, 255, 0.3);
#         transform: translate(-50%, -50%);
#         transition: width 0.6s, height 0.6s;
#     }

#     .skill-badge:hover::before {
#         width: 300px;
#         height: 300px;
#     }

#     .skill-badge:hover {
#         transform: translateY(-4px) scale(1.05);
#         box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
#     }

#     .skill-badge-matched {
#         background: linear-gradient(135deg, #38ef7d 0%, #11998e 100%);
#         box-shadow: 0 4px 15px rgba(56, 239, 125, 0.5);
#     }

#     .skill-badge-missing {
#         background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
#         box-shadow: 0 4px 15px rgba(255, 107, 107, 0.5);
#     }

#     .skill-badge-extra {
#         background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
#         box-shadow: 0 4px 15px rgba(79, 172, 254, 0.5);
#     }

#     /* Skill category sections */
#     .skill-category {
#         background: rgba(255, 255, 255, 0.15);
#         backdrop-filter: blur(15px);
#         border-radius: 16px;
#         padding: 1.5rem;
#         margin: 1rem 0;
#         border: 1px solid rgba(255, 255, 255, 0.3);
#         animation: slideInUp 0.6s ease-out;
#     }

#     .skill-category-header {
#         display: flex;
#         align-items: center;
#         margin-bottom: 1rem;
#         font-size: 1.3rem;
#         font-weight: 700;
#         color: white !important;
#         text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.4);
#     }

#     .skill-category-icon {
#         font-size: 1.8rem;
#         margin-right: 12px;
#         animation: bounce 2s infinite;
#         filter: drop-shadow(2px 2px 4px rgba(0, 0, 0, 0.3));
#     }

#     @keyframes bounce {
#         0%, 100% { transform: translateY(0); }
#         50% { transform: translateY(-10px); }
#     }

#     @keyframes slideInUp {
#         from {
#             opacity: 0;
#             transform: translateY(30px);
#         }
#         to {
#             opacity: 1;
#             transform: translateY(0);
#         }
#     }

#     /* Metric cards */
#     .metric-card {
#         background: rgba(255, 255, 255, 0.18);
#         backdrop-filter: blur(20px);
#         border-radius: 16px;
#         padding: 1.5rem;
#         text-align: center;
#         border: 1px solid rgba(255, 255, 255, 0.3);
#         transition: all 0.3s ease;
#         position: relative;
#         overflow: hidden;
#     }

#     .metric-card::before {
#         content: '';
#         position: absolute;
#         top: 0;
#         left: 0;
#         right: 0;
#         height: 3px;
#         background: linear-gradient(90deg, var(--primary), var(--accent));
#     }

#     .metric-card:hover {
#         transform: translateY(-5px) scale(1.02);
#         box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5);
#         border-color: rgba(255, 255, 255, 0.4);
#     }

#     .metric-card div {
#         color: white !important;
#         text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
#     }

#     /* Streamlit buttons */
#     .stButton > button {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white !important;
#         border: none;
#         border-radius: 12px;
#         padding: 1rem 2.5rem;
#         font-weight: 600;
#         font-size: 16px;
#         transition: all 0.3s ease;
#         box-shadow: 0 8px 20px rgba(102, 126, 234, 0.5);
#         text-transform: uppercase;
#         letter-spacing: 1px;
#         position: relative;
#         overflow: hidden;
#         text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
#     }

#     .stButton > button::before {
#         content: '';
#         position: absolute;
#         top: 0;
#         left: -100%;
#         width: 100%;
#         height: 100%;
#         background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
#         transition: left 0.5s;
#     }

#     .stButton > button:hover::before {
#         left: 100%;
#     }

#     .stButton > button:hover {
#         transform: translateY(-3px);
#         box-shadow: 0 12px 30px rgba(102, 126, 234, 0.7);
#     }

#     /* Tabs styling */
#     .stTabs [data-baseweb="tab-list"] {
#         gap: 8px;
#         background: rgba(0, 0, 0, 0.3);
#         padding: 8px;
#         border-radius: 16px;
#         backdrop-filter: blur(15px);
#     }

#     .stTabs [data-baseweb="tab"] {
#         background: transparent;
#         border-radius: 12px;
#         color: rgba(255, 255, 255, 0.9) !important;
#         font-weight: 600;
#         padding: 12px 24px;
#         transition: all 0.3s ease;
#         text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
#     }

#     .stTabs [aria-selected="true"] {
#         background: rgba(255, 255, 255, 0.25);
#         color: white !important;
#         box-shadow: 0 4px 20px rgba(255, 255, 255, 0.25);
#         text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4);
#     }

#     /* Text area styling */
#     .stTextArea textarea {
#         background: rgba(0, 0, 0, 0.4);
#         border: 1px solid rgba(255, 255, 255, 0.3);
#         border-radius: 16px;
#         color: white !important;
#         font-family: 'JetBrains Mono', monospace;
#         font-size: 14px;
#         line-height: 1.6;
#         backdrop-filter: blur(10px);
#         transition: all 0.4s ease;
#     }

#     .stTextArea textarea::placeholder {
#         color: rgba(255, 255, 255, 0.5);
#     }

#     .stTextArea textarea:focus {
#         border-color: rgba(102, 126, 234, 0.8);
#         box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3);
#         outline: none;
#         background: rgba(0, 0, 0, 0.5);
#     }

#     /* Scrollbar styling */
#     ::-webkit-scrollbar {
#         width: 10px;
#     }

#     ::-webkit-scrollbar-track {
#         background: rgba(255, 255, 255, 0.1);
#         border-radius: 10px;
#     }

#     ::-webkit-scrollbar-thumb {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         border-radius: 10px;
#     }

#     ::-webkit-scrollbar-thumb:hover {
#         background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%);
#     }

#     /* Dashboard cards */
#     .dashboard-card {
#         background: rgba(255, 255, 255, 0.15);
#         backdrop-filter: blur(20px);
#         border: 1px solid rgba(255, 255, 255, 0.3);
#         border-radius: 16px;
#         padding: 1.5rem;
#         margin: 1rem 0;
#         box-shadow: 0 8px 32px rgba(31, 38, 135, 0.4);
#         transition: all 0.3s ease;
#     }

#     .dashboard-card:hover {
#         transform: translateY(-4px);
#         box-shadow: 0 12px 40px rgba(102, 126, 234, 0.6);
#         border-color: rgba(255, 255, 255, 0.4);
#     }

#     .dashboard-card h3, .dashboard-card h4 {
#         color: white !important;
#         text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.4);
#     }

#     .dashboard-card ul, .dashboard-card li {
#         color: rgba(255, 255, 255, 0.95) !important;
#         text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
#     }

#     /* Stat boxes */
#     .stat-box {
#         background: linear-gradient(135deg, rgba(102, 126, 234, 0.25) 0%, rgba(118, 75, 162, 0.25) 100%);
#         border: 1px solid rgba(102, 126, 234, 0.5);
#         border-radius: 12px;
#         padding: 1.5rem;
#         text-align: center;
#         transition: all 0.3s ease;
#         backdrop-filter: blur(10px);
#     }

#     .stat-box:hover {
#         transform: scale(1.05);
#         border-color: rgba(102, 126, 234, 0.7);
#         box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
#     }

#     .stat-number {
#         font-size: 2.5rem;
#         font-weight: 700;
#         color: white !important;
#         filter: drop-shadow(2px 2px 4px rgba(0, 0, 0, 0.4));
#     }

#     .stat-label {
#         color: rgba(255, 255, 255, 0.95) !important;
#         font-size: 0.9rem;
#         margin-top: 0.5rem;
#         text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
#     }

#     /* Sidebar styling */
#     [data-testid="stSidebar"] {
#         background: linear-gradient(180deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
#         backdrop-filter: blur(20px);
#     }

#     [data-testid="stSidebar"] h1,
#     [data-testid="stSidebar"] h2,
#     [data-testid="stSidebar"] h3 {
#         color: white !important;
#         text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4);
#     }

#     [data-testid="stSidebar"] p,
#     [data-testid="stSidebar"] div {
#         color: rgba(255, 255, 255, 0.95) !important;
#         text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
#     }

#     /* Alert boxes */
#     .stAlert {
#         background: rgba(255, 255, 255, 0.15) !important;
#         backdrop-filter: blur(10px);
#         border: 1px solid rgba(255, 255, 255, 0.3) !important;
#         border-radius: 12px;
#     }

#     .stAlert > div {
#         color: white !important;
#         text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
#     }

#     /* Info boxes */
#     .stInfo {
#         background: rgba(79, 172, 254, 0.2) !important;
#         border-left: 4px solid #4facfe !important;
#     }

#     /* Success boxes */
#     .stSuccess {
#         background: rgba(56, 239, 125, 0.2) !important;
#         border-left: 4px solid #38ef7d !important;
#     }

#     /* Warning boxes */
#     .stWarning {
#         background: rgba(254, 202, 87, 0.2) !important;
#         border-left: 4px solid #feca57 !important;
#     }

#     /* Error boxes */
#     .stError {
#         background: rgba(255, 107, 107, 0.2) !important;
#         border-left: 4px solid #ff6b6b !important;
#     }

#     /* Dataframe styling */
#     .stDataFrame {
#         background: rgba(255, 255, 255, 0.1);
#         border-radius: 12px;
#         padding: 1rem;
#         backdrop-filter: blur(10px);
#     }

#     .stDataFrame table {
#         color: white !important;
#     }

#     .stDataFrame th {
#         background: rgba(102, 126, 234, 0.3) !important;
#         color: white !important;
#         font-weight: 600;
#         text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
#     }

#     .stDataFrame td {
#         color: rgba(255, 255, 255, 0.95) !important;
#         text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
#     }

#     /* Expander styling */
#     .streamlit-expanderHeader {
#         background: rgba(255, 255, 255, 0.15) !important;
#         border-radius: 12px;
#         color: white !important;
#         font-weight: 600;
#         text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
#     }

#     .streamlit-expanderHeader:hover {
#         background: rgba(255, 255, 255, 0.2) !important;
#     }

#     /* Progress bar */
#     .stProgress > div > div {
#         background-color: rgba(255, 255, 255, 0.2);
#     }

#     .stProgress > div > div > div {
#         background: linear-gradient(90deg, #667eea 0%, #f093fb 100%);
#     }

#     /* Download button */
#     .stDownloadButton > button {
#         background: linear-gradient(135deg, #38ef7d 0%, #11998e 100%);
#         color: white !important;
#         border: none;
#         border-radius: 12px;
#         padding: 0.75rem 1.5rem;
#         font-weight: 600;
#         box-shadow: 0 4px 15px rgba(56, 239, 125, 0.4);
#         text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
#         transition: all 0.3s ease;
#     }

#     .stDownloadButton > button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 6px 20px rgba(56, 239, 125, 0.6);
#     }

#     /* Metric widgets */
#     [data-testid="stMetricValue"] {
#         color: white !important;
#         font-size: 2rem !important;
#         font-weight: 700 !important;
#         text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4);
#     }

#     [data-testid="stMetricLabel"] {
#         color: rgba(255, 255, 255, 0.9) !important;
#         text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
#     }

#     /* File uploader */
#     .stFileUploader {
#         background: rgba(255, 255, 255, 0.1);
#         border-radius: 12px;
#         padding: 1rem;
#         border: 2px dashed rgba(255, 255, 255, 0.3);
#     }

#     .stFileUploader label {
#         color: white !important;
#         font-weight: 600;
#         text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
#     }

#     /* Slider */
#     .stSlider label {
#         color: white !important;
#         font-weight: 600;
#         text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
#     }

#     /* Checkbox */
#     .stCheckbox label {
#         color: rgba(255, 255, 255, 0.95) !important;
#         text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
#     }

#     /* Select box */
#     .stSelectbox label {
#         color: white !important;
#         font-weight: 600;
#         text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
#     }

#     /* Text input */
#     .stTextInput label {
#         color: white !important;
#         font-weight: 600;
#         text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
#     }

#     .stTextInput input {
#         background: rgba(0, 0, 0, 0.4);
#         color: white !important;
#         border: 1px solid rgba(255, 255, 255, 0.3);
#         border-radius: 8px;
#     }

#     /* Spinner */
#     .stSpinner > div {
#         border-top-color: #667eea !important;
#     }

#     /* Make sure all markdown text is visible */
#     .stMarkdown {
#         color: rgba(255, 255, 255, 0.95) !important;
#     }

#     /* Code blocks */
#     code {
#         background: rgba(0, 0, 0, 0.4) !important;
#         color: #f093fb !important;
#         padding: 2px 6px;
#         border-radius: 4px;
#         font-family: 'JetBrains Mono', monospace;
#     }

#     pre {
#         background: rgba(0, 0, 0, 0.4) !important;
#         border: 1px solid rgba(255, 255, 255, 0.2);
#         border-radius: 8px;
#         padding: 1rem;
#     }

#     /* Links */
#     a {
#         color: #4facfe !important;
#         text-decoration: none;
#         font-weight: 600;
#         text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
#         transition: all 0.3s ease;
#     }

#     a:hover {
#         color: #00f2fe !important;
#         text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4);
#     }

#     /* Horizontal rule */
#     hr {
#         border: none;
#         height: 1px;
#         background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
#         margin: 2rem 0;
#     }
#     </style>
#     """, unsafe_allow_html=True)
def load_custom_css():
    """Load professional CSS with blue gradient backgrounds everywhere and maximum text visibility"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --primary-blue: #4A90E2;
        --secondary-blue: #357ABD;
        --dark-blue: #2C5F8D;
        --light-blue: #7DB3E8;
        --accent-cyan: #00D4FF;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --text-primary: #FFFFFF;
        --text-secondary: #F0F4F8;
        --text-muted: #D1E0F0;
        --card-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    /* Global text rendering optimization */
    * {
        text-rendering: optimizeLegibility;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* Main container with enhanced gradient */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: linear-gradient(135deg, #1a2a6c 0%, #2d4a8c 25%, #4A90E2 50%, #3d7ab8 75%, #2e5f9d 100%);
        background-attachment: fixed;
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        min-height: 100vh;
        padding: 1rem;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Universal text visibility - ENHANCED */
    body, div, span, p, a, li, td, th, label, .stMarkdown, .stMarkdown * {
        color: var(--text-primary) !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8), 0 0 8px rgba(0, 0, 0, 0.6);
    }

    /* Headings with maximum visibility */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #FFFFFF !important;
        text-shadow: 0 3px 10px rgba(0, 0, 0, 0.9), 0 0 20px rgba(0, 0, 0, 0.7);
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    h1 { font-size: 2.5rem; }
    h2 { font-size: 2rem; }
    h3 { font-size: 1.5rem; }

    /* Paragraph text - high visibility */
    p, .stMarkdown p {
        color: #F0F4F8 !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8), 0 0 6px rgba(0, 0, 0, 0.6);
        line-height: 1.7;
        font-size: 0.95rem;
    }

    /* Strong emphasis */
    strong, b {
        color: #FFFFFF !important;
        font-weight: 700;
        text-shadow: 0 2px 6px rgba(0, 0, 0, 0.9);
    }

    /* List items */
    ul, ol, li {
        color: #F0F4F8 !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8), 0 0 6px rgba(0, 0, 0, 0.6);
        line-height: 1.8;
    }

    /* Custom header with blue gradient */
    .custom-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 35%, #4A90E2 65%, #357ABD 100%);
        backdrop-filter: blur(30px) saturate(180%);
        -webkit-backdrop-filter: blur(30px) saturate(180%);
        padding: 3rem 2rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), 
                    inset 0 1px 0 rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(74, 144, 226, 0.5);
        position: relative;
        overflow: hidden;
    }

    .custom-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.08), transparent);
        animation: shimmer 3s infinite;
    }

    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }

    .custom-header h1 {
        text-shadow: 0 4px 12px rgba(0, 0, 0, 0.9), 0 0 30px rgba(0, 0, 0, 0.7);
        margin: 0 !important;
        position: relative;
        z-index: 1;
        color: #FFFFFF !important;
    }

    .custom-header p {
        text-shadow: 0 3px 8px rgba(0, 0, 0, 0.9);
        position: relative;
        z-index: 1;
        color: #F0F4F8 !important;
        font-size: 1.1rem;
    }

    /* Glass cards with blue gradient */
    .glass-card {
        background: linear-gradient(135deg, #2a4d7c 0%, #2d5a9b 50%, #3668ab 100%);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(74, 144, 226, 0.4);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: var(--card-shadow), inset 0 1px 0 rgba(255, 255, 255, 0.15);
        transition: all 0.4s cubic-bezier(0.23, 1, 0.320, 1);
    }

    .glass-card:hover {
        transform: translateY(-6px);
        background: linear-gradient(135deg, #3558a0 0%, #3d6bbf 50%, #4579d4 100%);
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.5), 
                    0 0 0 1px rgba(74, 144, 226, 0.6),
                    inset 0 1px 0 rgba(255, 255, 255, 0.2);
        border-color: rgba(74, 144, 226, 0.6);
    }

    .glass-card h2, .glass-card h3, .glass-card h4 {
        color: #FFFFFF !important;
        text-shadow: 0 3px 8px rgba(0, 0, 0, 0.9);
    }

    /* Skill badges with blue gradient */
    .skill-badge {
        display: inline-block;
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 50%, #2C5F8D 100%);
        color: #FFFFFF !important;
        padding: 10px 20px;
        margin: 6px 4px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5), 
                    inset 0 1px 0 rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(74, 144, 226, 0.4);
        transition: all 0.3s ease;
        cursor: pointer;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
        letter-spacing: 0.3px;
    }

    .skill-badge:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.6),
                    inset 0 1px 0 rgba(255, 255, 255, 0.25);
        background: linear-gradient(135deg, #5BA3F5 0%, #4A90E2 50%, #357ABD 100%);
    }

    .skill-badge-matched {
        background: linear-gradient(135deg, #10B981 0%, #059669 50%, #047857 100%);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.5);
    }

    .skill-badge-missing {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 50%, #B91C1C 100%);
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.5);
    }

    .skill-badge-extra {
        background: linear-gradient(135deg, #00D4FF 0%, #0EA5E9 50%, #0284C7 100%);
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.5);
    }

    /* Skill categories with blue gradient */
    .skill-category {
        background: linear-gradient(135deg, #2d4a8c 0%, #2f5199 50%, #3558a0 100%);
        backdrop-filter: blur(15px);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(74, 144, 226, 0.4);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
        animation: slideInUp 0.6s ease-out;
    }

    .skill-category-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        font-size: 1.3rem;
        font-weight: 700;
        color: #FFFFFF !important;
        text-shadow: 0 3px 8px rgba(0, 0, 0, 0.9);
    }

    .skill-category-icon {
        font-size: 1.8rem;
        margin-right: 12px;
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.8));
        animation: bounce 2s infinite;
    }

    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }

    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Metric cards with blue gradient */
    .metric-card {
        background: linear-gradient(135deg, #2a4d7c 0%, #3558a0 50%, #3d6bbf 100%);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(74, 144, 226, 0.5);
        box-shadow: var(--card-shadow);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #4A90E2, #00D4FF, #4A90E2);
        background-size: 200% 100%;
        animation: gradientMove 3s linear infinite;
    }

    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }

    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        background: linear-gradient(135deg, #3558a0 0%, #3d6bbf 50%, #4579d4 100%);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5), 
                    0 0 0 1px rgba(74, 144, 226, 0.6);
    }

    .metric-card div {
        color: #FFFFFF !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
    }

    /* Buttons with blue gradient */
    .stButton > button {
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 50%, #2C5F8D 100%);
        background-size: 200% 100%;
        color: #FFFFFF !important;
        border: none;
        border-radius: 12px;
        padding: 0.9rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5), 
                    inset 0 1px 0 rgba(255, 255, 255, 0.2);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        position: relative;
        overflow: hidden;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        background-position: 100% 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6),
                    inset 0 1px 0 rgba(255, 255, 255, 0.25);
    }

    /* Tabs styling with blue gradient */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, #1e3c72 0%, #2a4d7c 100%);
        padding: 10px;
        border-radius: 16px;
        backdrop-filter: blur(15px);
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.4);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: #D1E0F0 !important;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s ease;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3558a0 0%, #4A90E2 100%);
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4), 
                    inset 0 1px 0 rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(74, 144, 226, 0.5);
    }

    /* Text area with blue gradient background */
    .stTextArea textarea {
        background: linear-gradient(135deg, #1e3c72 0%, #2a4d7c 100%) !important;
        border: 1px solid rgba(74, 144, 226, 0.4) !important;
        border-radius: 12px;
        color: #FFFFFF !important;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 0.9rem;
        line-height: 1.6;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        padding: 1rem;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.6);
    }

    .stTextArea textarea::placeholder {
        color: #D1E0F0 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
    }

    .stTextArea textarea:focus {
        border-color: rgba(74, 144, 226, 0.8) !important;
        background: linear-gradient(135deg, #2a4d7c 0%, #2d5a9b 100%) !important;
        box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.3), 
                    0 4px 12px rgba(0, 0, 0, 0.4);
        outline: none;
    }

    /* Scrollbar styling with blue gradient */
    ::-webkit-scrollbar {
        width: 12px;
    }

    ::-webkit-scrollbar-track {
        background: linear-gradient(135deg, #1e3c72 0%, #2a4d7c 100%);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 50%, #2C5F8D 100%);
        border-radius: 10px;
        border: 2px solid rgba(74, 144, 226, 0.3);
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5BA3F5 0%, #4A90E2 50%, #357ABD 100%);
    }

    /* Dashboard cards with blue gradient */
    .dashboard-card {
        background: linear-gradient(135deg, #2a4d7c 0%, #2d5a9b 50%, #3668ab 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(74, 144, 226, 0.4);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--card-shadow);
        transition: all 0.3s ease;
    }

    .dashboard-card:hover {
        transform: translateY(-4px);
        background: linear-gradient(135deg, #3558a0 0%, #3d6bbf 50%, #4579d4 100%);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5),
                    0 0 0 1px rgba(74, 144, 226, 0.6);
        border-color: rgba(74, 144, 226, 0.6);
    }

    .dashboard-card h3, .dashboard-card h4 {
        color: #FFFFFF !important;
        text-shadow: 0 3px 8px rgba(0, 0, 0, 0.9);
    }

    .dashboard-card ul, .dashboard-card li {
        color: #F0F4F8 !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
    }

    /* Stat boxes with blue gradient */
    .stat-box {
        background: linear-gradient(135deg, #3558a0 0%, #4A90E2 100%);
        border: 1px solid rgba(74, 144, 226, 0.5);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }

    .stat-box:hover {
        transform: scale(1.05);
        background: linear-gradient(135deg, #4A90E2 0%, #5BA3F5 100%);
        border-color: rgba(74, 144, 226, 0.7);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.5);
    }

    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFFFFF !important;
        text-shadow: 0 3px 10px rgba(0, 0, 0, 0.9), 0 0 20px rgba(0, 0, 0, 0.7);
    }

    .stat-label {
        color: #F0F4F8 !important;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
        font-weight: 500;
    }

    /* Sidebar styling with blue gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a4d7c 50%, #357ABD 100%);
        backdrop-filter: blur(20px);
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.4);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
        text-shadow: 0 3px 8px rgba(0, 0, 0, 0.9);
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label {
        color: #F0F4F8 !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
    }

    /* Alert boxes with blue gradient */
    .stAlert {
        background: linear-gradient(135deg, #2d5a9b 0%, #3668ab 100%) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(74, 144, 226, 0.5) !important;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }

    .stAlert > div {
        color: #FFFFFF !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
        font-weight: 500;
    }

    .stInfo {
        background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%) !important;
        border-left: 4px solid #00D4FF !important;
    }

    .stSuccess {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        border-left: 4px solid #10B981 !important;
    }

    .stWarning {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%) !important;
        border-left: 4px solid #F59E0B !important;
    }

    .stError {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%) !important;
        border-left: 4px solid #EF4444 !important;
    }

    /* Dataframe styling with blue gradient */
    .stDataFrame {
        background: linear-gradient(135deg, #2a4d7c 0%, #2d5a9b 100%);
        border-radius: 12px;
        padding: 1rem;
        backdrop-filter: blur(10px);
        box-shadow: var(--card-shadow);
    }

    .stDataFrame table {
        color: #FFFFFF !important;
    }

    .stDataFrame th {
        background: linear-gradient(135deg, #3558a0 0%, #4A90E2 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
        border-bottom: 2px solid rgba(74, 144, 226, 0.6);
    }

    .stDataFrame td {
        color: #F0F4F8 !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.6);
        border-bottom: 1px solid rgba(74, 144, 226, 0.3);
    }

    /* Download button with gradient */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10B981 0%, #059669 50%, #047857 100%);
        background-size: 200% 100%;
        color: #FFFFFF !important;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
        transition: all 0.3s ease;
    }

    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        background-position: 100% 0;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.6);
    }

    /* Input fields with blue gradient */
    .stTextInput input {
        background: linear-gradient(135deg, #1e3c72 0%, #2a4d7c 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(74, 144, 226, 0.4) !important;
        border-radius: 8px;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.6);
    }

    .stTextInput input:focus {
        background: linear-gradient(135deg, #2a4d7c 0%, #2d5a9b 100%) !important;
        border-color: rgba(74, 144, 226, 0.8) !important;
        box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.3);
    }

    /* File uploader with blue gradient */
    .stFileUploader {
        background: linear-gradient(135deg, #2d5a9b 0%, #3668ab 100%);
        border-radius: 12px;
        padding: 1rem;
        border: 2px dashed rgba(74, 144, 226, 0.5);
        transition: all 0.3s ease;
    }

    .stFileUploader:hover {
        border-color: rgba(74, 144, 226, 0.8);
        background: linear-gradient(135deg, #3668ab 0%, #3d6bbf 100%);
    }

    .stFileUploader label {
        color: #FFFFFF !important;
        font-weight: 600;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
    }

    /* Input field labels */
    .stTextInput label,
    .stSelectbox label,
    .stSlider label,
    .stCheckbox label {
        color: #FFFFFF !important;
        font-weight: 600;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
    }

    /* Progress bar with blue gradient */
    .stProgress > div > div {
        background: linear-gradient(135deg, #1e3c72 0%, #2a4d7c 100%);
        border-radius: 10px;
    }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4A90E2 0%, #00D4FF 50%, #4A90E2 100%);
        background-size: 200% 100%;
        border-radius: 10px;
        animation: progressShine 2s linear infinite;
    }

    @keyframes progressShine {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }

    /* Expander with blue gradient */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #2d5a9b 0%, #3668ab 100%) !important;
        border-radius: 12px;
        color: #FFFFFF !important;
        font-weight: 600;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
        border: 1px solid rgba(74, 144, 226, 0.4);
    }

    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #3668ab 0%, #3d6bbf 100%) !important;
        border-color: rgba(74, 144, 226, 0.6);
    }

    /* Code blocks with blue gradient */
    code {
        background: linear-gradient(135deg, #1e3c72 0%, #2a4d7c 100%) !important;
        color: #00D4FF !important;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        border: 1px solid rgba(0, 212, 255, 0.4);
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.8);
    }

    pre {
        background: linear-gradient(135deg, #1e3c72 0%, #2a4d7c 100%) !important;
        border: 1px solid rgba(74, 144, 226, 0.4);
        border-radius: 8px;
        padding: 1rem;
    }

    pre code {
        background: transparent !important;
        border: none;
    }

    /* Links */
    a {
        color: #00D4FF !important;
        text-decoration: none;
        font-weight: 600;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
        transition: all 0.3s ease;
    }

    a:hover {
        color: #7DB3E8 !important;
        text-shadow: 0 2px 6px rgba(0, 0, 0, 0.9), 0 0 15px rgba(0, 212, 255, 0.6);
    }

    /* Horizontal rule with blue gradient */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #4A90E2, #00D4FF, transparent);
        margin: 2rem 0;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.4);
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #4A90E2 !important;
        border-right-color: #00D4FF !important;
    }

    /* Radio buttons */
    .stRadio label {
        color: #F0F4F8 !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
    }

    /* Selectbox dropdown with blue gradient */
    .stSelectbox div[data-baseweb="select"] {
        background: linear-gradient(135deg, #1e3c72 0%, #2a4d7c 100%) !important;
        border-color: rgba(74, 144, 226, 0.4) !important;
    }

    .stSelectbox div[data-baseweb="select"] span {
        color: #FFFFFF !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.6);
    }

    /* Multiselect with blue gradient */
    .stMultiSelect div[data-baseweb="select"] {
        background: linear-gradient(135deg, #1e3c72 0%, #2a4d7c 100%) !important;
        border-color: rgba(74, 144, 226, 0.4) !important;
    }

    .stMultiSelect div[data-baseweb="select"] span {
        color: #FFFFFF !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.6);
    }

    /* Date input with blue gradient */
    .stDateInput input {
        background: linear-gradient(135deg, #1e3c72 0%, #2a4d7c 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(74, 144, 226, 0.4) !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.6);
    }

    /* Time input with blue gradient */
    .stTimeInput input {
        background: linear-gradient(135deg, #1e3c72 0%, #2a4d7c 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(74, 144, 226, 0.4) !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.6);
    }

    /* Number input with blue gradient */
    .stNumberInput input {
        background: linear-gradient(135deg, #1e3c72 0%, #2a4d7c 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(74, 144, 226, 0.4) !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.6);
    }

    /* Slider track with blue gradient */
    .stSlider [role="slider"] {
        background: linear-gradient(135deg, #4A90E2 0%, #00D4FF 100%);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
    }

    .stSlider [data-baseweb="slider"] {
        background: linear-gradient(135deg, #1e3c72 0%, #2a4d7c 100%);
    }

    /* Toast notifications with blue gradient */
    .stToast {
        background: linear-gradient(135deg, #2d5a9b 0%, #3668ab 100%) !important;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(74, 144, 226, 0.5);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }

    .stToast div {
        color: #FFFFFF !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
    }

    /* Plotly charts with blue gradient background */
    .js-plotly-plot .plotly {
        background: linear-gradient(135deg, #2a4d7c 0%, #2d5a9b 100%) !important;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }

    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }

    /* Status messages with blue gradient */
    .stStatus {
        background: linear-gradient(135deg, #2d5a9b 0%, #3668ab 100%) !important;
        border: 1px solid rgba(74, 144, 226, 0.5);
        border-radius: 12px;
    }

    .stStatus div {
        color: #FFFFFF !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
    }

    /* Tooltip styling with blue gradient */
    [data-baseweb="tooltip"] {
        background: linear-gradient(135deg, #1e3c72 0%, #2a4d7c 100%) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(74, 144, 226, 0.6);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.6);
    }

    [data-baseweb="tooltip"] div {
        color: #FFFFFF !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.8);
    }

    /* Modal/Dialog styling with blue gradient */
    [data-baseweb="modal"] {
        background: linear-gradient(135deg, #1a2a6c 0%, #2a4d7c 50%, #2d5a9b 100%) !important;
        backdrop-filter: blur(10px);
    }

    /* Popover styling with blue gradient */
    [data-baseweb="popover"] {
        background: linear-gradient(135deg, #2a4d7c 0%, #2d5a9b 50%, #357ABD 100%) !important;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(74, 144, 226, 0.5);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
    }

    [data-baseweb="popover"] div {
        color: #FFFFFF !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
    }

    /* Metric widgets */
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        text-shadow: 0 3px 10px rgba(0, 0, 0, 0.9), 0 0 20px rgba(0, 0, 0, 0.7);
    }

    [data-testid="stMetricLabel"] {
        color: #F0F4F8 !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
        font-weight: 500;
    }

    [data-testid="stMetricDelta"] {
        color: #F0F4F8 !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.8);
    }

    /* Image captions */
    .stImage figcaption {
        color: #F0F4F8 !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
        font-weight: 500;
    }

    /* Video player controls */
    video {
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(74, 144, 226, 0.3);
    }

    /* Audio player */
    audio {
        border-radius: 12px;
        filter: brightness(1.2) contrast(1.1);
    }

    /* Iframe styling */
    iframe {
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(74, 144, 226, 0.3);
    }

    /* Color picker labels */
    .stColorPicker label {
        color: #FFFFFF !important;
        font-weight: 600;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
    }

    /* Camera input labels */
    .stCameraInput label {
        color: #FFFFFF !important;
        font-weight: 600;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
    }

    /* Form styling with blue gradient */
    [data-testid="stForm"] {
        background: linear-gradient(135deg, #2a4d7c 0%, #2d5a9b 50%, #3668ab 100%);
        border: 1px solid rgba(74, 144, 226, 0.4);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    /* Form submit button with gradient */
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #10B981 0%, #059669 50%, #047857 100%);
        background-size: 200% 100%;
        color: #FFFFFF !important;
        font-weight: 600;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
    }

    [data-testid="stFormSubmitButton"] > button:hover {
        background-position: 100% 0;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.6);
    }

    /* Markdown tables with blue gradient */
    .stMarkdown table {
        background: linear-gradient(135deg, #2a4d7c 0%, #2d5a9b 100%);
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }

    .stMarkdown th {
        background: linear-gradient(135deg, #3558a0 0%, #4A90E2 100%);
        color: #FFFFFF !important;
        font-weight: 600;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
        padding: 12px;
        border-bottom: 2px solid rgba(74, 144, 226, 0.6);
    }

    .stMarkdown td {
        color: #F0F4F8 !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.6);
        padding: 10px 12px;
        border-bottom: 1px solid rgba(74, 144, 226, 0.3);
    }

    .stMarkdown tr:hover {
        background: linear-gradient(135deg, #3558a0 0%, #3d6bbf 100%);
    }

    /* Blockquote styling with blue gradient */
    blockquote {
        border-left: 4px solid #4A90E2;
        padding-left: 1.5rem;
        margin: 1.5rem 0;
        background: linear-gradient(135deg, #2d5a9b 0%, #3668ab 100%);
        padding: 1rem 1.5rem;
        border-radius: 8px;
        color: #F0F4F8 !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
    }

    /* Keyboard key styling with blue gradient */
    kbd {
        background: linear-gradient(135deg, #1e3c72 0%, #2a4d7c 100%);
        border: 1px solid rgba(74, 144, 226, 0.5);
        border-radius: 4px;
        padding: 2px 6px;
        color: #FFFFFF !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9em;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.4);
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
    }

    /* Mark/highlight styling with gradient */
    mark {
        background: linear-gradient(135deg, #F59E0B 0%, #FBBF24 100%);
        color: #FFFFFF !important;
        padding: 2px 4px;
        border-radius: 3px;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.6);
    }

    /* Selection styling with blue gradient */
    ::selection {
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
        color: #FFFFFF;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.8);
    }

    ::-moz-selection {
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
        color: #FFFFFF;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.8);
    }

    /* Focus visible for accessibility */
    *:focus-visible {
        outline: 2px solid #00D4FF;
        outline-offset: 2px;
    }

    /* Checkbox styling */
    .stCheckbox div {
        color: #F0F4F8 !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.6);
    }

    /* All dropdown menus */
    [data-baseweb="menu"] {
        background: linear-gradient(135deg, #2a4d7c 0%, #2d5a9b 100%) !important;
        border: 1px solid rgba(74, 144, 226, 0.4);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }

    [data-baseweb="menu"] li {
        color: #F0F4F8 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
    }

    [data-baseweb="menu"] li:hover {
        background: linear-gradient(135deg, #3558a0 0%, #3d6bbf 100%) !important;
        color: #FFFFFF !important;
    }

    /* All input placeholders */
    ::placeholder {
        color: #D1E0F0 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
        opacity: 0.8;
    }

    /* Container backgrounds */
    .element-container {
        color: #FFFFFF !important;
    }

    /* Ensure all child elements inherit proper text color */
    .stApp * {
        color: inherit;
    }

    /* Override any remaining white backgrounds with blue gradients */
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    .block-container {
        background: linear-gradient(135deg, #1a2a6c 0%, #2d4a8c 50%, #4A90E2 100%) !important;
    }

    /* Column containers with blue gradient */
    [data-testid="column"] {
        background: linear-gradient(135deg, rgba(42, 77, 124, 0.3) 0%, rgba(45, 90, 155, 0.3) 100%);
        border-radius: 12px;
        padding: 0.5rem;
    }

    /* App container with blue gradient */
    .stApp {
        background: linear-gradient(135deg, #1a2a6c 0%, #2d4a8c 25%, #4A90E2 50%, #3d7ab8 75%, #2e5f9d 100%) !important;
        background-attachment: fixed;
    }

    /* Widget containers with blue gradient */
    .row-widget,
    .stMarkdown,
    [class*="css"] {
        background: transparent;
    }

    /* Expander content with blue gradient */
    .streamlit-expanderContent {
        background: linear-gradient(135deg, #2a4d7c 0%, #2d5a9b 100%);
        border-radius: 0 0 12px 12px;
        padding: 1rem;
        border: 1px solid rgba(74, 144, 226, 0.4);
        border-top: none;
    }

    /* Tab content with blue gradient */
    [data-baseweb="tab-panel"] {
        background: linear-gradient(135deg, rgba(42, 77, 124, 0.4) 0%, rgba(45, 90, 155, 0.4) 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    /* Markdown container backgrounds */
    .stMarkdown > div {
        background: transparent;
    }

    /* Widget backgrounds */
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"] {
        background: transparent;
    }

    /* Remove any white from metric containers */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(74, 144, 226, 0.25) 0%, rgba(53, 122, 189, 0.2) 100%);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(74, 144, 226, 0.3);
    }

    /* File uploader drop area */
    [data-testid="stFileUploadDropzone"] {
        background: linear-gradient(135deg, #2d5a9b 0%, #3668ab 100%) !important;
        border: 2px dashed rgba(74, 144, 226, 0.5) !important;
    }

    /* Selectbox and dropdown containers */
    [data-baseweb="popover"] > div {
        background: linear-gradient(135deg, #2a4d7c 0%, #2d5a9b 100%) !important;
    }

    /* Calendar/date picker popup */
    [data-baseweb="calendar"] {
        background: linear-gradient(135deg, #2a4d7c 0%, #2d5a9b 100%) !important;
        border: 1px solid rgba(74, 144, 226, 0.5);
    }

    [data-baseweb="calendar"] button {
        color: #FFFFFF !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
    }

    [data-baseweb="calendar"] [aria-selected="true"] {
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%) !important;
    }

    /* Number input buttons */
    [data-baseweb="input"] button {
        background: linear-gradient(135deg, #3558a0 0%, #4A90E2 100%) !important;
        color: #FFFFFF !important;
    }

    /* Chip/tag backgrounds */
    [data-baseweb="tag"] {
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%) !important;
        color: #FFFFFF !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
    }

    /* Empty state backgrounds */
    .stEmpty {
        background: linear-gradient(135deg, rgba(42, 77, 124, 0.3) 0%, rgba(45, 90, 155, 0.3) 100%);
        border-radius: 12px;
        padding: 2rem;
    }

    /* Spinner container */
    .stSpinner {
        background: transparent;
    }

    /* Caption text */
    .caption, small {
        color: #D1E0F0 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
    }

    /* Reduce motion for accessibility */
    @media (prefers-reduced-motion: reduce) {
        *,
        *::before,
        *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }

    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .custom-header h1 {
            font-size: 1.8rem;
        }
        
        .custom-header p {
            font-size: 0.95rem;
        }
        
        .glass-card {
            padding: 1.5rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
        
        .stat-number {
            font-size: 2rem;
        }
        
        .skill-badge {
            padding: 8px 16px;
            font-size: 0.85rem;
        }
    }

    /* Print styles */
    @media print {
        .main {
            background: white;
        }
        
        .glass-card,
        .dashboard-card {
            background: white;
            border: 1px solid #ddd;
            box-shadow: none;
        }
        
        body, div, span, p, a, li, td, th, label {
            color: black !important;
            text-shadow: none;
        }
    }
    </style>
    """, unsafe_allow_html=True)
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'current_page': 'upload',
        'parsed_data': {},
        'processing_complete': False,
        'extraction_complete': False,
        'analysis_complete': False,
        'resume_skills': set(),
        'job_skills': set(),
        'exact_analysis': {},
        'semantic_analysis': {},
        'resume_extraction_methods': {},
        'job_extraction_methods': {},
        'bert_enabled': False,
        'custom_ner_trained': False,
        'training_annotations': [],
        'extraction_statistics': {},
        'm3_analysis_result': None,
        'm3_encoder': None,
        'm3_strong_threshold': 0.80,
        'm3_partial_threshold': 0.50
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def sidebar_navigation():
    """Enhanced sidebar navigation"""
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: rgba(255,255,255,0.1); 
                    border-radius: 16px; margin-bottom: 1.5rem; border: 1px solid rgba(255,255,255,0.2);'>
            <h2 style='color: white; margin: 0; font-size: 1.8rem;'>🎯</h2>
            <h3 style='color: white; margin: 0.5rem 0 0 0;'>Skill Gap Analysis</h3>
            <p style='color: rgba(255,255,255,0.7); margin: 0.5rem 0 0 0; font-size: 13px;'>
                AI-Powered NLP System
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🔍 Quick Navigation")
        
        nav_options = [
            ("📤 Upload Documents", "upload", "📄"),
            ("🎯 Extract Skills", "extraction", "🔍"),
            ("🔬 Analyze Gaps", "analysis", "📊"),
            ("📊 Dashboard", "dashboard", "📈")
        ]
        
        for label, page_key, icon in nav_options:
            button_type = "primary" if st.session_state.current_page == page_key else "secondary"
            
            disabled = False
            if page_key == "extraction" and not st.session_state.processing_complete:
                disabled = True
            elif page_key == "analysis" and not st.session_state.extraction_complete:
                disabled = True
            elif page_key == "dashboard" and not st.session_state.analysis_complete:
                disabled = True
            
            if st.button(f"{icon} {label.split(' ', 1)[1]}", use_container_width=True, 
                        disabled=disabled, type=button_type, key=f"nav_{page_key}"):
                st.session_state.current_page = page_key
                st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 📊 Progress")
        
        # Calculate progress
        processing_done = st.session_state.processing_complete
        extraction_done = st.session_state.extraction_complete
        analysis_done = st.session_state.analysis_complete
        
        completed_count = sum([processing_done, extraction_done, analysis_done])
        progress_percentage = (completed_count / 3) * 100
        
        st.progress(progress_percentage / 100)
        
        # Display progress items using a simpler approach
        st.markdown(
            f"""
            <div style='background: rgba(255,255,255,0.08); padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.15); margin-top: 1rem;'>
                <div style='margin-bottom: 0.8rem; display: flex; align-items: center;'>
                    <span style='font-size: 1.3rem; margin-right: 8px;'>{'✅' if processing_done else '⭕'}</span>
                    <span style='color: {'#38ef7d' if processing_done else '#888'}; font-weight: 600; font-size: 0.85rem;'>Documents</span>
                </div>
                <div style='margin-bottom: 0.8rem; display: flex; align-items: center;'>
                    <span style='font-size: 1.3rem; margin-right: 8px;'>{'✅' if extraction_done else '⭕'}</span>
                    <span style='color: {'#38ef7d' if extraction_done else '#888'}; font-weight: 600; font-size: 0.85rem;'>Extraction</span>
                </div>
                <div style='margin-bottom: 0.8rem; display: flex; align-items: center;'>
                    <span style='font-size: 1.3rem; margin-right: 8px;'>{'✅' if analysis_done else '⭕'}</span>
                    <span style='color: {'#38ef7d' if analysis_done else '#888'}; font-weight: 600; font-size: 0.85rem;'>Analysis</span>
                </div>
                <div style='margin-top: 1rem; text-align: center; color: rgba(255,255,255,0.9); font-size: 1.2rem; font-weight: 700;'>{progress_percentage:.0f}% Complete</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        
        if st.session_state.analysis_complete and st.session_state.m3_analysis_result:
            st.markdown("### 📈 Quick Stats")
            stats = st.session_state.m3_analysis_result.get_statistics()
            stats_html = f"""
            <div style='background: rgba(56, 239, 125, 0.1); padding: 1rem; border-radius: 12px; 
                        border: 1px solid rgba(56, 239, 125, 0.3); margin-bottom: 1rem;'>
                <div style='color: white; text-align: center;'>
                    <div style='font-size: 2rem; font-weight: 700; color: #38ef7d;'>
                        {stats['overall_score']:.0f}%
                    </div>
                    <div style='font-size: 0.9rem; opacity: 0.8;'>Overall Score</div>
                </div>
            </div>
            """
            st.markdown(stats_html, unsafe_allow_html=True)
        
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🧹 Clear Cache", use_container_width=True, key="clear_cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ Cache cleared!", icon="✅")
        
        if st.button("🔄 Reset Session", use_container_width=True, key="reset_session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.rerun()

def show_navigation_buttons(prev_page=None, next_page=None, next_disabled=False):
    """Show previous and next navigation buttons"""
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if prev_page:
            if st.button("⬅️ Previous", use_container_width=True, key=f"prev_{prev_page}"):
                st.session_state.current_page = prev_page
                st.rerun()
    
    with col3:
        if next_page:
            if st.button("Next ➡️", use_container_width=True, type="primary", 
                        disabled=next_disabled, key=f"next_{next_page}"):
                st.session_state.current_page = next_page
                st.rerun()

# PAGE 1: UPLOAD
def document_upload_section():
    """Enhanced document upload section"""
    st.markdown("""
    <div class='custom-header'>
        <h1 style='margin: 0; font-size: 2.5rem; position: relative; z-index: 1;'>📄 Document Processing</h1>
        <p style='font-size: 1.2em; margin: 1rem 0 0 0; opacity: 0.9; position: relative; z-index: 1;'>
            Upload your resume and job description for intelligent analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""<div class='glass-card'><h3 style='color: white; text-align: center; margin-bottom: 1rem;'>📄 Upload Resume</h3></div>""", unsafe_allow_html=True)
        resume_file = st.file_uploader("Choose resume file", type=['pdf', 'docx', 'txt'], key="resume_uploader", label_visibility="collapsed")
        if resume_file:
            st.success(f"✅ {resume_file.name}", icon="✅")

    with col2:
        st.markdown("""<div class='glass-card'><h3 style='color: white; text-align: center; margin-bottom: 1rem;'>💼 Upload Job Description</h3></div>""", unsafe_allow_html=True)
        job_file = st.file_uploader("Choose job description", type=['pdf', 'docx', 'txt'], key="job_uploader", label_visibility="collapsed")
        if job_file:
            st.success(f"✅ {job_file.name}", icon="✅")

    st.markdown("## ✏️ Or Enter Text Manually")
    col3, col4 = st.columns(2, gap="large")

    resume_text = col3.text_area("📝 Resume Text", height=200, key="resume_text", placeholder="Paste your resume here...")
    job_text = col4.text_area("📋 Job Description", height=200, key="job_text", placeholder="Paste job description here...")

    return resume_file, job_file, resume_text, job_text

def process_documents(parser, resume_file, job_file, resume_text, job_text):
    """Process documents with enhanced feedback"""
    if not any([resume_file, job_file, resume_text, job_text]):
        st.warning("⚠️ Please upload files or enter text before processing.", icon="⚠️")
        return None

    with st.spinner("🔄 Processing documents with advanced NLP..."):
        progress_bar = st.progress(0)
        status = st.empty()
        
        results = {'resume': {'raw': '', 'cleaned': ''}, 'job': {'raw': '', 'cleaned': ''}}

        status.info("📄 Parsing resume...")
        progress_bar.progress(25)
        time.sleep(0.3)
        
        if resume_file:
            file_path = parser.save_uploaded_file(resume_file)
            raw, cleaned, _ = parser.extract_text_auto(file_path)
            results['resume'] = {'raw': raw, 'cleaned': cleaned}
        elif resume_text:
            raw, cleaned, _ = parser.process_text_input(resume_text, "resume")
            results['resume'] = {'raw': raw, 'cleaned': cleaned}

        progress_bar.progress(50)
        
        status.info("💼 Parsing job description...")
        progress_bar.progress(75)
        time.sleep(0.3)
        
        if job_file:
            file_path = parser.save_uploaded_file(job_file)
            raw, cleaned, _ = parser.extract_text_auto(file_path)
            results['job'] = {'raw': raw, 'cleaned': cleaned}
        elif job_text:
            raw, cleaned, _ = parser.process_text_input(job_text, "job_description")
            results['job'] = {'raw': raw, 'cleaned': cleaned}

        progress_bar.progress(100)
        status.success("✅ Processing complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status.empty()
        
    return results

def data_review_section(results):
    """Data review section"""
    if not results:
        return
    
    st.markdown("""
    <div class='glass-card' style='margin-top: 2rem;'>
        <h2 style='color: white; text-align: center; margin-bottom: 1.5rem;'>📋 Document Content Review</h2>
        <p style='color: rgba(255,255,255,0.8); text-align: center; font-size: 1.1rem;'>Review the raw and cleaned document content</p>
    </div>
    """, unsafe_allow_html=True)
    
    doc_tabs = st.tabs(["📄 Resume Content", "💼 Job Description Content"])
    
    with doc_tabs[0]:
        if results['resume']['raw']:
            review_single_document("Resume", results['resume']['raw'], results['resume']['cleaned'])
        else:
            st.info("No resume data available for review.")
    
    with doc_tabs[1]:
        if results['job']['raw']:
            review_single_document("Job Description", results['job']['raw'], results['job']['cleaned'])
        else:
            st.info("No job description data available for review.")

def review_single_document(doc_name, raw_text, cleaned_text):
    """Review a single document"""
    content_tabs = st.tabs(["🔍 Raw Text", "✨ Cleaned Text"])
    
    with content_tabs[0]:
        st.markdown(f"#### Raw {doc_name} Content")
        st.text_area(
            f"Original {doc_name.lower()} text:", 
            value=raw_text, 
            height=400, 
            key=f"raw_{doc_name.lower().replace(' ', '_')}", 
            help="This is the original extracted text before any cleaning"
        )
        st.download_button(
            f"💾 Download Raw {doc_name}", 
            data=raw_text, 
            file_name=f"raw_{doc_name.lower().replace(' ', '_')}.txt", 
            mime="text/plain",
            key=f"download_raw_{doc_name.lower().replace(' ', '_')}"
        )
    
    with content_tabs[1]:
        st.markdown(f"#### Cleaned {doc_name} Content")
        st.text_area(
            f"Processed {doc_name.lower()} text:", 
            value=cleaned_text, 
            height=400, 
            key=f"cleaned_{doc_name.lower().replace(' ', '_')}", 
            help="This is the cleaned and processed text ready for analysis"
        )
        st.download_button(
            f"💾 Download Cleaned {doc_name}", 
            data=cleaned_text, 
            file_name=f"cleaned_{doc_name.lower().replace(' ', '_')}.txt", 
            mime="text/plain",
            key=f"download_cleaned_{doc_name.lower().replace(' ', '_')}"
        )

def page_upload_display():
    """Document upload page display"""
    parser = get_parser()
    resume_file, job_file, resume_text, job_text = document_upload_section()
    
    st.markdown("## 🚀 Process Documents")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📊 Parse Documents", use_container_width=True, type="primary", key="parse_btn"):
            results = process_documents(parser, resume_file, job_file, resume_text, job_text)
            if results:
                st.session_state.parsed_data = results
                st.session_state.processing_complete = True
                st.success("✅ Documents processed successfully!", icon="✅")
                st.balloons()
                time.sleep(0.5)
                st.rerun()
    
    if st.session_state.processing_complete:
        st.markdown("""
        <div class='glass-card' style='margin-top: 2rem;'>
            <h2 style='color: white; text-align: center;'>✅ Processing Complete!</h2>
            <p style='color: rgba(255,255,255,0.8); text-align: center; font-size: 1.1rem;'>Your documents have been successfully parsed and cleaned.</p>
        </div>
        """, unsafe_allow_html=True)
        
        data_review_section(st.session_state.parsed_data)
        
        show_navigation_buttons(prev_page=None, next_page="extraction", next_disabled=False)

# PAGE 2: EXTRACTION
def display_enhanced_skill_details(skills: Set[str], category: str, color: str, icon: str):
    """Display skills with enhanced UI"""
    if not skills:
        st.info(f"No {category.lower()} found.", icon="ℹ️")
        return
    
    analyzer = M2SkillGapAnalyzer(sentence_model=None)
    categorized = analyzer.categorize_skills(skills)
    
    if categorized['technical']:
        st.markdown(f"""<div class='skill-category'><div class='skill-category-header'>
            <span class='skill-category-icon'>💻</span><span>Technical Skills ({len(categorized['technical'])})</span></div>""", unsafe_allow_html=True)
        skills_html = "<div style='margin-top: 1rem;'>"
        for skill in sorted(categorized['technical']):
            badge_class = f"skill-badge-{category.lower().split()[0]}" if category in ["Matched", "Missing", "Additional"] else "skill-badge"
            skills_html += f"<span class='skill-badge {badge_class}'>{skill}</span>"
        skills_html += "</div></div>"
        st.markdown(skills_html, unsafe_allow_html=True)
    
    if categorized['soft']:
        st.markdown(f"""<div class='skill-category'><div class='skill-category-header'>
            <span class='skill-category-icon'>🤝</span><span>Soft Skills ({len(categorized['soft'])})</span></div>""", unsafe_allow_html=True)
        skills_html = "<div style='margin-top: 1rem;'>"
        for skill in sorted(categorized['soft']):
            badge_class = f"skill-badge-{category.lower().split()[0]}" if category in ["Matched", "Missing", "Additional"] else "skill-badge"
            skills_html += f"<span class='skill-badge {badge_class}'>{skill}</span>"
        skills_html += "</div></div>"
        st.markdown(skills_html, unsafe_allow_html=True)
    
    if categorized['other']:
        st.markdown(f"""<div class='skill-category'><div class='skill-category-header'>
            <span class='skill-category-icon'>📌</span><span>Other Skills ({len(categorized['other'])})</span></div>""", unsafe_allow_html=True)
        skills_html = "<div style='margin-top: 1rem;'>"
        for skill in sorted(categorized['other']):
            badge_class = f"skill-badge-{category.lower().split()[0]}" if category in ["Matched", "Missing", "Additional"] else "skill-badge"
            skills_html += f"<span class='skill-badge {badge_class}'>{skill}</span>"
        skills_html += "</div></div>"
        st.markdown(skills_html, unsafe_allow_html=True)

def page_extraction_display():
    """Skill extraction page display"""
    if not st.session_state.processing_complete:
        st.warning("⚠️ Please complete document upload first.", icon="⚠️")
        show_navigation_buttons(prev_page="upload", next_page=None)
        return

    st.markdown("""
    <div class='custom-header'>
        <h1 style='margin: 0; font-size: 2.5rem; position: relative; z-index: 1;'>🎯 Advanced Skill Extraction</h1>
        <p style='font-size: 1.2em; margin: 1rem 0 0 0; opacity: 0.9; position: relative; z-index: 1;'>
            AI-powered multi-method NLP analysis with BERT embeddings
        </p>
    </div>
    """, unsafe_allow_html=True)

    nlp = load_spacy_model()
    sentence_model = load_sentence_transformer()
    bert_tokenizer, bert_model = load_bert_model()
    
    if not nlp:
        st.error("❌ spaCy model not loaded.", icon="❌")
        return

    parsed_data = st.session_state.parsed_data
    resume_text = parsed_data['resume']['cleaned']
    job_text = parsed_data['job']['cleaned']

    skill_db = SkillDatabase()
    
    with st.expander("⚙️ Advanced NLP Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            use_bert = st.checkbox("🧠 Enable BERT", value=True if bert_model else False, disabled=not bert_model)
            use_custom_ner = st.checkbox("🎯 Train Custom NER", value=False)
        
        with col2:
            show_method_stats = st.checkbox("📊 Show Statistics", value=True)
            export_training_data = st.checkbox("💾 Export Training Data", value=False)
    
    bert_extractor = None
    if use_bert and bert_tokenizer and bert_model:
        bert_extractor = BERTSkillExtractor(bert_tokenizer, bert_model, skill_db)
        st.session_state.bert_enabled = True
    
    if use_custom_ner:
        st.markdown("### 🎯 Custom NER Training")
        with st.spinner("📚 Preparing training data..."):
            annotator = SkillAnnotator(skill_db)
            resume_annotations = annotator.auto_annotate(resume_text)
            job_annotations = annotator.auto_annotate(job_text)
            all_training_data = resume_annotations + job_annotations
            st.success(f"✅ Generated {len(all_training_data)} annotated samples")
        
        if st.button("🚀 Train Custom NER Model", type="primary", key="train_ner_btn"):
            with st.spinner("🔄 Training..."):
                trainer = CustomNERTrainer(nlp)
                trained_nlp = trainer.train_ner(all_training_data, n_iter=10)
                nlp = trained_nlp
                st.session_state.custom_ner_trained = True
                st.success("✅ Model trained!", icon="✅")
                st.balloons()
    
    if export_training_data:
        annotator = SkillAnnotator(skill_db)
        resume_annotations = annotator.auto_annotate(resume_text)
        job_annotations = annotator.auto_annotate(job_text)
        all_training_data = resume_annotations + job_annotations
        training_json = annotator.export_annotations(all_training_data)
        st.download_button("📥 Download Training Data", data=training_json, 
                          file_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")

    extractor = AdvancedSkillExtractor(nlp, bert_extractor)
    analyzer = M2SkillGapAnalyzer(sentence_model)

    st.markdown("## 🚀 Extract & Analyze Skills")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 Analyze Skills", use_container_width=True, type="primary", key="analyze_btn"):
            with st.spinner("🔄 Extracting skills..."):
                progress_bar = st.progress(0)
                
                progress_bar.progress(25)
                resume_skills = extractor.get_combined_skills(resume_text)
                resume_stats = extractor.get_extraction_statistics()
                
                progress_bar.progress(50)
                job_skills = extractor.get_combined_skills(job_text)
                job_stats = extractor.get_extraction_statistics()
                
                progress_bar.progress(75)
                exact_analysis = analyzer.calculate_exact_match(resume_skills, job_skills)
                semantic_analysis = analyzer.calculate_semantic_similarity(resume_skills, job_skills)
                
                st.session_state.extraction_complete = True
                st.session_state.resume_skills = resume_skills
                st.session_state.job_skills = job_skills
                st.session_state.exact_analysis = exact_analysis
                st.session_state.semantic_analysis = semantic_analysis
                st.session_state.extraction_statistics = {'resume': resume_stats, 'job': job_stats}
                
                progress_bar.progress(100)
                progress_bar.empty()
                
            st.success("✅ Analysis complete!", icon="✅")
            st.balloons()
            time.sleep(0.5)
            st.rerun()

    if st.session_state.extraction_complete:
        resume_skills = st.session_state.resume_skills
        job_skills = st.session_state.job_skills
        exact_analysis = st.session_state.exact_analysis

        st.markdown("""<div class='glass-card' style='margin-top: 2rem;'>
            <h2 style='color: white; text-align: center; margin-bottom: 2rem;'>📊 Extraction Results</h2>
        </div>""", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        metrics_data = [
            ("📄", "Resume Skills", len(resume_skills), "#667eea"),
            ("💼", "Job Requirements", len(job_skills), "#764ba2"),
            ("✅", "Matched", exact_analysis['matched_count'], "#38ef7d"),
            ("📈", "Match Rate", f"{exact_analysis['match_percentage']:.1f}%", "#feca57")
        ]
        
        for col, (icon, label, value, color) in zip([col1, col2, col3, col4], metrics_data):
            with col:
                st.markdown(f"""<div class='metric-card'>
                    <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>{icon}</div>
                    <div style='font-size: 2rem; font-weight: 700; color: {color}; margin-bottom: 0.5rem;'>{value}</div>
                    <div style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("### 📈 Visual Analysis")
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            fig1 = create_skill_visualization(exact_analysis, "Match Distribution")
            st.plotly_chart(fig1, use_container_width=True)
        with viz_col2:
            fig2 = create_skill_comparison_chart(resume_skills, job_skills)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 🔍 Detailed Analysis")
        skill_tabs = st.tabs(["✅ Matched", "❌ Missing", "➕ Additional"])
        
        with skill_tabs[0]:
            st.markdown("""<div style='background: linear-gradient(135deg, rgba(56, 239, 125, 0.1) 0%, rgba(17, 153, 142, 0.1) 100%);
                padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; border: 1px solid rgba(56, 239, 125, 0.3);'>
                <h3 style='color: #38ef7d; margin: 0;'>✅ Matched Skills</h3></div>""", unsafe_allow_html=True)
            display_enhanced_skill_details(exact_analysis['matched'], "Matched", "#38ef7d", "✅")
        
        with skill_tabs[1]:
            st.markdown("""<div style='background: linear-gradient(135deg, rgba(255, 107, 107, 0.1) 0%, rgba(238, 90, 82, 0.1) 100%);
                padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; border: 1px solid rgba(255, 107, 107, 0.3);'>
                <h3 style='color: #ff6b6b; margin: 0;'>❌ Missing Skills</h3></div>""", unsafe_allow_html=True)
            display_enhanced_skill_details(exact_analysis['missing'], "Missing", "#ff6b6b", "❌")
        
        with skill_tabs[2]:
            st.markdown("""<div style='background: linear-gradient(135deg, rgba(79, 172, 254, 0.1) 0%, rgba(0, 242, 254, 0.1) 100%);
                padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; border: 1px solid rgba(79, 172, 254, 0.3);'>
                <h3 style='color: #4facfe; margin: 0;'>➕ Additional Skills</h3></div>""", unsafe_allow_html=True)
            display_enhanced_skill_details(exact_analysis['extra'], "Additional", "#4facfe", "➕")

        st.markdown("### 💾 Export Results")
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            csv_data = export_to_csv(exact_analysis)
            st.download_button("📊 CSV", data=csv_data, file_name=f"summary_{datetime.now().strftime('%Y%m%d')}.csv", 
                             mime="text/csv", use_container_width=True)
        
        with export_col2:
            detailed_report = export_analysis_report(resume_skills, job_skills, exact_analysis, {}, False, False)
            st.download_button("📄 TXT", data=detailed_report, file_name=f"report_{datetime.now().strftime('%Y%m%d')}.txt", 
                             mime="text/plain", use_container_width=True)
        
        with export_col3:
            json_data = export_to_json(resume_skills, job_skills, exact_analysis, {}, {}, {})
            st.download_button("📋 JSON", data=json_data, file_name=f"data_{datetime.now().strftime('%Y%m%d')}.json", 
                             mime="application/json", use_container_width=True)

        show_navigation_buttons(prev_page="upload", next_page="analysis", next_disabled=False)

# PAGE 3: ANALYSIS  
def page_analysis_display():
    """Gap analysis page"""
    if not st.session_state.extraction_complete:
        st.warning("⚠️ Please complete extraction first.", icon="⚠️")
        show_navigation_buttons(prev_page="extraction", next_page=None)
        return

    st.markdown("""<div class='custom-header'>
        <h1 style='margin: 0; font-size: 2.5rem; position: relative; z-index: 1;'>🔬 Advanced Gap Analysis</h1>
        <p style='font-size: 1.2em; margin: 1rem 0 0 0; opacity: 0.9; position: relative; z-index: 1;'>
            BERT-based semantic similarity matching</p></div>""", unsafe_allow_html=True)

    resume_skills = list(st.session_state.resume_skills)
    job_skills = list(st.session_state.job_skills)

    with st.expander("⚙️ Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            strong_threshold = st.slider("Strong Match", 0.0, 1.0, st.session_state.m3_strong_threshold, 0.05)
            st.session_state.m3_strong_threshold = strong_threshold
        with col2:
            partial_threshold = st.slider("Partial Match", 0.0, 1.0, st.session_state.m3_partial_threshold, 0.05)
            st.session_state.m3_partial_threshold = partial_threshold

    st.markdown("## 🚀 Perform Analysis")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔬 Analyze with BERT", use_container_width=True, type="primary", key="bert_analyze"):
            with st.spinner("🔄 Analyzing..."):
                try:
                    if not st.session_state.m3_encoder:
                        encoder = SentenceBERTEncoder()
                        st.session_state.m3_encoder = encoder
                    else:
                        encoder = st.session_state.m3_encoder
                    
                    calculator = SimilarityCalculator()
                    analyzer = M3SkillGapAnalyzer(encoder, calculator, strong_threshold, partial_threshold)
                    result = analyzer.analyze(resume_skills, job_skills)
                    
                    st.session_state.m3_analysis_result = result
                    st.session_state.analysis_complete = True
                    
                    st.success("✅ Analysis complete!", icon="✅")
                    st.balloons()
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed: {str(e)}", icon="❌")

    if st.session_state.analysis_complete and st.session_state.m3_analysis_result:
        result = st.session_state.m3_analysis_result
        stats = result.get_statistics()

        st.markdown("""<div class='glass-card' style='margin-top: 2rem;'>
            <h2 style='color: white; text-align: center; margin-bottom: 2rem;'>📊 Analysis Results</h2></div>""", unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)
        metrics = [
            ("🎯", "Score", f"{stats['overall_score']:.1f}%", "#667eea"),
            ("📋", "Required", stats['total_required_skills'], "#764ba2"),
            ("✅", "Matched", stats['matched_count'], "#38ef7d"),
            ("⚠️", "Partial", stats['partial_count'], "#feca57"),
            ("❌", "Missing", stats['missing_count'], "#ff6b6b")
        ]
        
        for col, (icon, label, value, color) in zip([col1, col2, col3, col4, col5], metrics):
            with col:
                st.markdown(f"""<div class='metric-card'>
                    <div style='font-size: 2rem; margin-bottom: 0.5rem;'>{icon}</div>
                    <div style='font-size: 1.8rem; font-weight: 700; color: {color};'>{value}</div>
                    <div style='color: rgba(255,255,255,0.8); font-size: 0.85rem;'>{label}</div></div>""", unsafe_allow_html=True)

        tabs = st.tabs(["📊 Visualizations", "✅ Matched", "⚠️ Partial", "❌ Missing", "📈 Matrix", "🎓 Learning", "💾 Export"])
        
        with tabs[0]:
            viz_col1, viz_col2 = st.columns(2)
            visualizer = GapVisualizer()
            with viz_col1:
                fig = visualizer.create_overall_score_gauge(result.overall_score)
                st.plotly_chart(fig, use_container_width=True)
            with viz_col2:
                fig = visualizer.create_match_distribution_pie(result)
                st.plotly_chart(fig, use_container_width=True)
            
            fig = visualizer.create_skill_comparison_bar(result, 15)
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:
            if result.matched_skills:
                for match in result.matched_skills:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f'**{match.jd_skill}** ↔ {match.resume_skill}')
                    with col2:
                        st.metric("", f"{match.similarity*100:.1f}%")
            else:
                st.info("No strong matches")
        
        with tabs[2]:
            if result.partial_matches:
                for match in result.partial_matches:
                    st.markdown(f'**{match.jd_skill}** ↔ {match.resume_skill} ({match.similarity*100:.1f}%)')
            else:
                st.info("No partial matches")
        
        with tabs[3]:
            if result.missing_skills:
                ranker = SkillRanker()
                categorized = ranker.categorize_by_urgency(result.missing_skills)
                if categorized['critical']:
                    st.markdown("**🔴 Critical**")
                    for m in categorized['critical']:
                        st.markdown(f'• {m.jd_skill} - {m.priority}')
            else:
                st.success("No gaps!")
        
        with tabs[4]:
            fig = visualizer.create_similarity_heatmap(result.similarity_matrix, result.resume_skills, result.jd_skills)
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[5]:
            if result.missing_skills:
                path_gen = LearningPathGenerator()
                plan = path_gen.generate_path(result.missing_skills, result.resume_skills)
                for i, item in enumerate(plan[:5], 1):
                    with st.expander(f"{i}. {item['skill']}", expanded=(i<=3)):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Similarity", f"{item['current_similarity']*100:.1f}%")
                        col2.metric("Difficulty", item['difficulty'])
                        col3.metric("Time", item['time_estimate'])
            else:
                st.success("All skills matched!")
        
        with tabs[6]:
            report_gen = ReportGenerator()
            col1, col2, col3 = st.columns(3)
            with col1:
                txt = report_gen.generate_text_report(result)
                st.download_button("📄 TXT", txt, f"report_{datetime.now().strftime('%Y%m%d')}.txt", use_container_width=True)
            with col2:
                csv = report_gen.generate_csv_report(result)
                st.download_button("📊 CSV", csv, f"analysis_{datetime.now().strftime('%Y%m%d')}.csv", use_container_width=True)
            with col3:
                jsn = report_gen.generate_json_report(result)
                st.download_button("📋 JSON", jsn, f"data_{datetime.now().strftime('%Y%m%d')}.json", use_container_width=True)

        show_navigation_buttons(prev_page="extraction", next_page="dashboard", next_disabled=False)

# PAGE 4: DASHBOARD
def page_dashboard_display():
    """Interactive professional dashboard with comprehensive analytics"""
    if not st.session_state.analysis_complete:
        st.warning("⚠️ Complete analysis first.", icon="⚠️")
        show_navigation_buttons(prev_page="analysis", next_page=None)
        return

    st.markdown("""<div class='custom-header'>
        <h1 style='margin: 0; font-size: 2.5rem; position: relative; z-index: 1;'>📊 Comprehensive Analytics Dashboard</h1>
        <p style='font-size: 1.2em; margin: 1rem 0 0 0; opacity: 0.9; position: relative; z-index: 1;'>
            </p></div>""", unsafe_allow_html=True)

    result = st.session_state.m3_analysis_result
    stats = result.get_statistics()
    exact_analysis = st.session_state.exact_analysis
    resume_skills = st.session_state.resume_skills
    job_skills = st.session_state.job_skills

    # Hero Metrics Section
    st.markdown("## 🎯 Performance Overview")
    
    hero_col1, hero_col2, hero_col3, hero_col4, hero_col5 = st.columns(5)
    
    metrics = [
        ("🎯", "Overall Match", f"{stats['overall_score']:.1f}%", 
         "Primary success indicator", "#667eea"),
        ("✅", "Strong Matches", stats['matched_count'], 
         "Skills aligned perfectly", "#38ef7d"),
        ("⚠️", "Partial Matches", stats['partial_count'], 
         "Skills need strengthening", "#feca57"),
        ("❌", "Skill Gaps", stats['missing_count'], 
         "Skills to acquire", "#ff6b6b"),
        ("➕", "Bonus Skills", exact_analysis['extra_count'], 
         "Additional advantages", "#4facfe")
    ]
    
    for col, (icon, label, value, desc, color) in zip([hero_col1, hero_col2, hero_col3, hero_col4, hero_col5], metrics):
        with col:
            st.markdown(f"""
            <div class='stat-box' style='min-height: 160px;'>
                <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>{icon}</div>
                <div class='stat-number' style='font-size: 2rem;'>{value}</div>
                <div style='color: white; font-weight: 600; font-size: 0.9rem; margin: 0.5rem 0;'>{label}</div>
                <div style='color: rgba(255,255,255,0.6); font-size: 0.75rem;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Recommendation Banner
    st.markdown("## 💡 AI-Powered Recommendations")
    
    recommendation_col1, recommendation_col2 = st.columns([2, 1])
    
    with recommendation_col1:
        if stats['overall_score'] >= 80:
            recommendation_color = "#38ef7d"
            recommendation_icon = "🎉"
            recommendation_title = "Excellent Candidate Match!"
            recommendation_text = f"""You demonstrate strong alignment with the job requirements. 
            Your profile shows {stats['matched_count']} strong matches and an overall compatibility of {stats['overall_score']:.1f}%. 
            <strong>Action:</strong> Highlight your matching skills prominently in your application and prepare 
            to discuss your experience with these technologies in detail."""
        elif stats['overall_score'] >= 60:
            recommendation_color = "#feca57"
            recommendation_icon = "💪"
            recommendation_title = "Good Match with Growth Potential"
            recommendation_text = f"""You have a solid foundation with {stats['matched_count']} strong matches ({stats['overall_score']:.1f}% compatibility). 
            Focus on strengthening your {stats['partial_count']} partial matches and consider upskilling in the {stats['missing_count']} missing areas. 
            <strong>Action:</strong> Create a 60-90 day learning plan for critical gaps."""
        else:
            recommendation_color = "#ff6b6b"
            recommendation_icon = "📚"
            recommendation_title = "Significant Skill Development Needed"
            recommendation_text = f"""Your current match rate is {stats['overall_score']:.1f}% with {stats['missing_count']} skill gaps identified. 
            This role may require 3-6 months of focused learning. <strong>Action:</strong> Prioritize the {min(stats['missing_count'], 5)} critical 
            skills and consider entry-level positions or internships to build experience."""
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
                    padding: 2rem; border-radius: 16px; border-left: 6px solid {recommendation_color};
                    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);'>
            <h3 style='color: {recommendation_color}; margin: 0 0 1rem 0; font-size: 1.4rem;'>
                {recommendation_icon} {recommendation_title}
            </h3>
            <p style='color: rgba(255,255,255,0.9); line-height: 1.8; margin: 0; font-size: 0.95rem;'>
                {recommendation_text}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with recommendation_col2:
        # Quick Stats Card
        readiness_level = 'Ready to Apply' if stats['overall_score'] >= 70 else 'Nearly Ready' if stats['overall_score'] >= 50 else 'Build Experience'
        readiness_icon = '🎯' if stats['overall_score'] >= 70 else '📈' if stats['overall_score'] >= 50 else '📚'
        readiness_color = '#38ef7d' if stats['overall_score'] >= 70 else '#feca57' if stats['overall_score'] >= 50 else '#ff6b6b'
        
        st.markdown(f"""
        <div class='dashboard-card' style='text-align: center; min-height: 200px; display: flex; 
                    flex-direction: column; justify-content: center;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>{readiness_icon}</div>
            <div style='color: white; font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;'>
                Readiness Level
            </div>
            <div style='color: {readiness_color}; font-size: 1.5rem; font-weight: 700;'>
                {readiness_level}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Charts Section
    st.markdown("## 📈 Visual Analytics")
    
    chart_tabs = st.tabs([
        "📊 Performance Gauges", 
        "🎯 Skill Distribution", 
        "📉 Gap Analysis",
        "🔥 Skill Heatmap",
        "📅 Learning Roadmap"
    ])
    
    visualizer = GapVisualizer()
    
    with chart_tabs[0]:
        gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
        
        with gauge_col1:
            st.markdown("#### Overall Match Score")
            fig_gauge = visualizer.create_overall_score_gauge(result.overall_score)
            st.plotly_chart(fig_gauge, use_container_width=True, key="dash_main_gauge")
        
        with gauge_col2:
            st.markdown("#### Match Distribution")
            fig_pie = visualizer.create_match_distribution_pie(result)
            st.plotly_chart(fig_pie, use_container_width=True, key="dash_pie")
        
        with gauge_col3:
            st.markdown("#### Skill Comparison")
            fig_comparison = create_skill_comparison_chart(resume_skills, job_skills)
            st.plotly_chart(fig_comparison, use_container_width=True, key="dash_comparison")
    
    with chart_tabs[1]:
        skill_dist_col1, skill_dist_col2 = st.columns(2)
        
        with skill_dist_col1:
            st.markdown("#### Top Skills by Similarity")
            fig_bar = visualizer.create_skill_comparison_bar(result, 20)
            st.plotly_chart(fig_bar, use_container_width=True, key="dash_skills_bar")
        
        with skill_dist_col2:
            st.markdown("#### Category Breakdown")
            analyzer = M2SkillGapAnalyzer(sentence_model=None)
            categorized_resume = analyzer.categorize_skills(resume_skills)
            categorized_job = analyzer.categorize_skills(job_skills)
            
            # Create category comparison
            categories = ['Technical', 'Soft Skills', 'Other']
            resume_counts = [
                len(categorized_resume.get('technical', [])),
                len(categorized_resume.get('soft', [])),
                len(categorized_resume.get('other', []))
            ]
            job_counts = [
                len(categorized_job.get('technical', [])),
                len(categorized_job.get('soft', [])),
                len(categorized_job.get('other', []))
            ]
            
            fig_category = go.Figure(data=[
                go.Bar(name='Resume', x=categories, y=resume_counts, marker_color='#667eea'),
                go.Bar(name='Job Required', x=categories, y=job_counts, marker_color='#f093fb')
            ])
            fig_category.update_layout(
                barmode='group',
                title="Skills by Category",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.2)',
                font=dict(color='white'),
                height=400
            )
            st.plotly_chart(fig_category, use_container_width=True, key="dash_category")
    
    with chart_tabs[2]:
        if result.missing_skills:
            st.markdown("#### Priority Gap Analysis")
            fig_priority = visualizer.create_gap_priority_chart(result.missing_skills)
            st.plotly_chart(fig_priority, use_container_width=True, key="dash_priority")
            
            # Gap urgency breakdown
            ranker = SkillRanker()
            categorized_gaps = ranker.categorize_by_urgency(result.missing_skills)
            
            gap_col1, gap_col2, gap_col3 = st.columns(3)
            with gap_col1:
                st.markdown(f"""
                <div class='stat-box' style='border-left: 4px solid #ff6b6b;'>
                    <div style='font-size: 2rem;'>🔴</div>
                    <div class='stat-number'>{len(categorized_gaps['critical'])}</div>
                    <div class='stat-label'>Critical Gaps</div>
                </div>
                """, unsafe_allow_html=True)
            with gap_col2:
                st.markdown(f"""
                <div class='stat-box' style='border-left: 4px solid #feca57;'>
                    <div style='font-size: 2rem;'>🟡</div>
                    <div class='stat-number'>{len(categorized_gaps['important'])}</div>
                    <div class='stat-label'>Important Gaps</div>
                </div>
                """, unsafe_allow_html=True)
            with gap_col3:
                st.markdown(f"""
                <div class='stat-box' style='border-left: 4px solid #38ef7d;'>
                    <div style='font-size: 2rem;'>🟢</div>
                    <div class='stat-number'>{len(categorized_gaps['beneficial'])}</div>
                    <div class='stat-label'>Nice to Have</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🎉 No skill gaps identified! Excellent job match!", icon="🎉")
    
    with chart_tabs[3]:
        st.markdown("#### Semantic Similarity Matrix")
        st.markdown("*Hover over cells to see exact similarity scores between your skills and job requirements*")
        fig_heatmap = visualizer.create_similarity_heatmap(
            result.similarity_matrix, 
            result.resume_skills, 
            result.jd_skills
        )
        st.plotly_chart(fig_heatmap, use_container_width=True, key="dash_heatmap")
    
    with chart_tabs[4]:
        if result.missing_skills:
            st.markdown("#### Personalized Learning Timeline")
            path_gen = LearningPathGenerator()
            learning_plan = path_gen.generate_path(result.missing_skills, result.resume_skills)
            
            if learning_plan:
                timeline_data = []
                cumulative_weeks = 0
                for item in learning_plan[:10]:
                    weeks = path_gen.estimate_weeks(item['time_estimate'])
                    timeline_data.append({
                        'Skill': item['skill'],
                        'Start Week': cumulative_weeks,
                        'Duration': weeks,
                        'Priority': item['priority']
                    })
                    cumulative_weeks += weeks
                
                df_timeline = pd.DataFrame(timeline_data)
                
                # Create Gantt-style chart
                fig_timeline = go.Figure()
                
                colors = {'HIGH': '#ff6b6b', 'MEDIUM': '#feca57', 'LOW': '#38ef7d'}
                
                for idx, row in df_timeline.iterrows():
                    fig_timeline.add_trace(go.Bar(
                        name=row['Skill'],
                        y=[row['Skill']],
                        x=[row['Duration']],
                        orientation='h',
                        marker=dict(color=colors.get(row['Priority'], '#667eea')),
                        text=f"{row['Duration']} weeks",
                        textposition='inside',
                        hovertemplate=f"<b>{row['Skill']}</b><br>Duration: {row['Duration']} weeks<br>Priority: {row['Priority']}<extra></extra>"
                    ))
                
                fig_timeline.update_layout(
                    title=f"Estimated Learning Timeline: {cumulative_weeks} weeks total",
                    xaxis_title="Weeks",
                    yaxis_title="Skills",
                    showlegend=False,
                    height=500,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0.2)',
                    font=dict(color='white'),
                    yaxis=dict(autorange="reversed")
                )
                
                st.plotly_chart(fig_timeline, use_container_width=True, key="dash_timeline")
                
                st.info(f"📅 **Total estimated time:** {cumulative_weeks} weeks (~{cumulative_weeks/4:.1f} months)", icon="📅")
        else:
            st.success("🎉 All skills matched! No learning timeline needed.", icon="🎉")

    # Detailed Skill Tables
    st.markdown("## 📋 Comprehensive Skill Breakdown")
    
    detail_tabs = st.tabs([
        "✅ Matched Skills",
        "⚠️ Partial Matches", 
        "❌ Critical Gaps",
        "➕ Bonus Skills"
    ])
    
    with detail_tabs[0]:
        if result.matched_skills:
            st.markdown(f"### {len(result.matched_skills)} Strong Skill Matches")
            matched_data = []
            for match in result.matched_skills:
                matched_data.append({
                    'Required Skill': match.jd_skill,
                    'Your Skill': match.resume_skill,
                    'Similarity': f"{match.similarity*100:.1f}%",
                    'Confidence': match.confidence_level,
                    'Status': '✅ Strong Match'
                })
            df_matched = pd.DataFrame(matched_data)
            st.dataframe(df_matched, use_container_width=True, hide_index=True, height=400)
        else:
            st.info("No strong matches found")
    
    with detail_tabs[1]:
        if result.partial_matches:
            st.markdown(f"### {len(result.partial_matches)} Partial Matches - Strengthen These")
            partial_data = []
            for match in result.partial_matches:
                partial_data.append({
                    'Required Skill': match.jd_skill,
                    'Your Closest Skill': match.resume_skill,
                    'Similarity': f"{match.similarity*100:.1f}%",
                    'Gap': f"{(1-match.similarity)*100:.1f}%",
                    'Action': 'Strengthen knowledge'
                })
            df_partial = pd.DataFrame(partial_data)
            st.dataframe(df_partial, use_container_width=True, hide_index=True, height=400)
        else:
            st.info("No partial matches found")
    
    with detail_tabs[2]:
        if result.missing_skills:
            st.markdown(f"### {len(result.missing_skills)} Critical Skill Gaps")
            ranker = SkillRanker()
            ranked_missing = ranker.rank_by_importance(result.missing_skills)
            
            missing_data = []
            for match in ranked_missing[:20]:
                missing_data.append({
                    'Skill Required': match.jd_skill,
                    'Current Closest': match.resume_skill,
                    'Similarity': f"{match.similarity*100:.1f}%",
                    'Gap Severity': f"{(1-match.similarity)*100:.1f}%",
                    'Priority': f"{match.priority}",
                    'Action Required': 'Learn & Practice'
                })
            df_missing = pd.DataFrame(missing_data)
            st.dataframe(df_missing, use_container_width=True, hide_index=True, height=400)
            
            # Learning resources for top gaps
            if ranked_missing:
                st.markdown("#### 🎓 Recommended Learning Resources")
                path_gen = LearningPathGenerator()
                learning_plan = path_gen.generate_path(ranked_missing[:5], result.resume_skills)
                
                for i, item in enumerate(learning_plan, 1):
                    with st.expander(f"{i}. {item['skill']} - {item['priority']} Priority", expanded=(i==1)):
                        resource_col1, resource_col2 = st.columns([2, 1])
                        
                        with resource_col1:
                            if item['resources']:
                                st.markdown("**📚 Learning Resources:**")
                                for resource in item['resources']:
                                    st.markdown(f"• {resource}")
                            else:
                                st.info("Search online for latest courses and tutorials")
                        
                        with resource_col2:
                            st.metric("Difficulty", item['difficulty'])
                            st.metric("Time Required", item['time_estimate'])
                            st.metric("Current Level", f"{item['current_similarity']*100:.0f}%")
        else:
            st.success("🎉 No critical gaps!", icon="🎉")
    
    with detail_tabs[3]:
        if exact_analysis['extra']:
            st.markdown(f"### {len(exact_analysis['extra'])} Additional Skills - Your Competitive Edge")
            extra_data = []
            for skill in sorted(list(exact_analysis['extra']))[:20]:
                extra_data.append({
                    'Bonus Skill': skill,
                    'Status': '➕ Additional Advantage',
                    'Value': 'Differentiator',
                    'Note': 'Highlight in your application'
                })
            df_extra = pd.DataFrame(extra_data)
            st.dataframe(df_extra, use_container_width=True, hide_index=True, height=400)
        else:
            st.info("No additional skills beyond requirements")

    # Action Plan Section
    st.markdown("## 🎯 Personalized Action Plan")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        st.markdown(f"""
        <div class='dashboard-card' style='border-top: 4px solid #38ef7d;'>
            <h4 style='color: #38ef7d; margin-top: 0;'>✅ Immediate Actions</h4>
            <ul style='color: rgba(255,255,255,0.9); line-height: 2;'>
                <li>Highlight your {stats['matched_count']} strong matches in resume</li>
                <li>Prepare case studies for matched skills</li>
                <li>Update LinkedIn with verified skills</li>
                <li>Request endorsements from colleagues</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with action_col2:
        st.markdown(f"""
        <div class='dashboard-card' style='border-top: 4px solid #feca57;'>
            <h4 style='color: #feca57; margin-top: 0;'>⚠️ Short-term Goals (1-2 months)</h4>
            <ul style='color: rgba(255,255,255,0.9); line-height: 2;'>
                <li>Focus on {stats['partial_count']} partial matches</li>
                <li>Take online courses for top 3 gaps</li>
                <li>Build 2-3 portfolio projects</li>
                <li>Join relevant communities</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with action_col3:
        st.markdown(f"""
        <div class='dashboard-card' style='border-top: 4px solid #ff6b6b;'>
            <h4 style='color: #ff6b6b; margin-top: 0;'>🎓 Long-term Development (3-6 months)</h4>
            <ul style='color: rgba(255,255,255,0.9); line-height: 2;'>
                <li>Master {min(stats['missing_count'], 5)} critical skills</li>
                <li>Obtain relevant certifications</li>
                <li>Contribute to open-source projects</li>
                <li>Network with industry professionals</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Export Section with Preview
    st.markdown("## 💾 Download Comprehensive Reports")
    
    export_container = st.container()
    with export_container:
        report_gen = ReportGenerator()
        
        export_col1, export_col2, export_col3, export_col4 = st.columns(4)
        
        with export_col1:
            txt_report = report_gen.generate_text_report(result)
            st.download_button(
                "📄 Full Text Report",
                data=txt_report,
                file_name=f"skill_gap_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
                help="Comprehensive analysis in text format"
            )
        
        with export_col2:
            csv_report = report_gen.generate_csv_report(result)
            st.download_button(
                "📊 CSV Data Export",
                data=csv_report,
                file_name=f"skills_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Spreadsheet format for analysis"
            )
        
        with export_col3:
            json_report = report_gen.generate_json_report(result)
            st.download_button(
                "📋 JSON Structure",
                data=json_report,
                file_name=f"analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
                help="Structured data for integrations"
            )
        
        with export_col4:
            # Executive Summary
            summary = f"""EXECUTIVE SUMMARY
{'='*50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL PERFORMANCE
Overall Match Score: {stats['overall_score']:.1f}%
Readiness Level: {readiness_level}

SKILL METRICS
Total Resume Skills: {len(resume_skills)}
Total Job Requirements: {len(job_skills)}
Strong Matches: {stats['matched_count']}
Partial Matches: {stats['partial_count']}
Skill Gaps: {stats['missing_count']}
Bonus Skills: {exact_analysis['extra_count']}

KEY RECOMMENDATION
{recommendation_text.replace('<strong>', '').replace('</strong>', '')}

{'='*50}
"""
            st.download_button(
                "📑 Executive Summary",
                data=summary,
                file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
                help="Quick overview for sharing"
            )

    # Final Success Message
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2.5rem; border-radius: 20px; margin: 3rem 0 2rem 0; text-align: center;
                box-shadow: 0 15px 40px rgba(102, 126, 234, 0.5);'>
        <h2 style='color: white; margin: 0 0 1rem 0; font-size: 2rem;'>🎉 Analysis Complete!</h2>
        <p style='color: white; opacity: 0.95; margin: 0 0 1.5rem 0; font-size: 1.1rem; line-height: 1.8;'>
            Your comprehensive skill gap analysis has identified <strong>{stats['matched_count']} strong matches</strong> 
            and <strong>{stats['missing_count']} areas for development</strong>.<br>
            Use these insights to enhance your profile and accelerate your career growth!
        </p>
        <div style='display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; margin-top: 1.5rem;'>
            <div style='background: rgba(255,255,255,0.2); padding: 1rem 1.5rem; border-radius: 12px;'>
                <div style='font-size: 1.5rem; font-weight: 700; color: white;'>{stats['overall_score']:.0f}%</div>
                <div style='font-size: 0.85rem; color: rgba(255,255,255,0.9);'>Match Score</div>
            </div>
            <div style='background: rgba(255,255,255,0.2); padding: 1rem 1.5rem; border-radius: 12px;'>
                <div style='font-size: 1.5rem; font-weight: 700; color: white;'>{len(resume_skills) + len(job_skills)}</div>
                <div style='font-size: 0.85rem; color: rgba(255,255,255,0.9);'>Skills Analyzed</div>
            </div>
            <div style='background: rgba(255,255,255,0.2); padding: 1rem 1.5rem; border-radius: 12px;'>
                <div style='font-size: 1.5rem; font-weight: 700; color: white;'>AI-Powered</div>
                <div style='font-size: 0.85rem; color: rgba(255,255,255,0.9);'>BERT Analysis</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    show_navigation_buttons(prev_page="analysis", next_page=None)

# MAIN
def main():
    """Main entry point"""
    load_custom_css()
    init_session_state()
    sidebar_navigation()
    
    if st.session_state.current_page == "upload":
        page_upload_display()
    elif st.session_state.current_page == "extraction":
        page_extraction_display()
    elif st.session_state.current_page == "analysis":
        page_analysis_display()
    else:
        page_dashboard_display()

if __name__ == "__main__":
    main()