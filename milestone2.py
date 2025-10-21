"""
Complete AI Skill Gap Analyzer with Advanced NLP + File Upload
Fixed and fully integrated version
"""

import streamlit as st
import pandas as pd
import json
import os
import tempfile
from datetime import datetime
import plotly.graph_objects as go
import numpy as np
import re
from collections import defaultdict
from typing import List, Set, Dict, Optional

# Try importing NLP libraries with graceful fallback
try:
    import spacy
    SPACY_AVAILABLE = True
except:
    SPACY_AVAILABLE = False
    st.warning("⚠️ spaCy not available. Using keyword-based extraction.")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMER_AVAILABLE = True
except:
    SENTENCE_TRANSFORMER_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except:
    TRANSFORMERS_AVAILABLE = False

# File readers
def read_txt_file(file_path):
    """Read text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()

def read_pdf_file(file_path):
    """Read PDF file"""
    try:
        import PyPDF2
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def read_docx_file(file_path):
    """Read DOCX file"""
    try:
        import docx
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"

# ============================================
# MODEL LOADING
# ============================================

@st.cache_resource
def load_spacy_model():
    """Load spaCy model"""
    if not SPACY_AVAILABLE:
        return None
    try:
        nlp = spacy.load("en_core_web_sm")
        programming_langs = {'c', 'r', 'go', 'd', 'f'}
        for lang in programming_langs:
            nlp.Defaults.stop_words.discard(lang)
        return nlp
    except:
        try:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            return spacy.load("en_core_web_sm")
        except:
            return None

@st.cache_resource
def load_sentence_transformer():
    """Load sentence transformer"""
    if not SENTENCE_TRANSFORMER_AVAILABLE:
        return None
    try:
        return SentenceTransformer('paraphrase-MiniLM-L6-v2')
    except:
        return None

@st.cache_resource
def load_bert_model():
    """Load BERT model"""
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    try:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return tokenizer, model
    except:
        return None, None

# ============================================
# SKILL DATABASE
# ============================================

class SkillDatabase:
    """Comprehensive skill database"""
    
    def __init__(self):
        self.skills = self._initialize_skills()
        self.abbreviations = self._initialize_abbreviations()
        self.patterns = self._initialize_patterns()
    
    def _initialize_skills(self):
        return {
            'programming': ['Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'C', 
                           'Ruby', 'PHP', 'Swift', 'Kotlin', 'Go', 'Rust', 'Scala', 'R'],
            'web': ['React', 'Angular', 'Vue', 'Node.js', 'Express', 'Django', 'Flask', 
                   'FastAPI', 'Spring Boot', 'HTML', 'CSS', 'Bootstrap', 'jQuery'],
            'databases': ['MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'SQLite', 
                         'Oracle', 'SQL Server', 'Cassandra', 'Elasticsearch'],
            'ml_ai': ['Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision',
                     'Neural Networks', 'TensorFlow', 'PyTorch', 'Keras', 'Scikit-learn'],
            'tools': ['Git', 'GitHub', 'Docker', 'Kubernetes', 'Jenkins', 'AWS', 
                     'Azure', 'GCP', 'Visual Studio', 'Eclipse', 'Power BI', 'Tableau'],
            'soft_skills': ['Leadership', 'Communication', 'Problem Solving', 
                          'Team Management', 'Critical Thinking', 'Adaptability',
                          'Project Management', 'Collaboration', 'Agile', 'Scrum']
        }
    
    def _initialize_abbreviations(self):
        return {
            'ML': 'Machine Learning', 'DL': 'Deep Learning',
            'AI': 'Artificial Intelligence', 'NLP': 'Natural Language Processing',
            'CV': 'Computer Vision', 'K8s': 'Kubernetes',
            'AWS': 'Amazon Web Services', 'GCP': 'Google Cloud Platform',
            'JS': 'JavaScript', 'TS': 'TypeScript'
        }
    
    def _initialize_patterns(self):
        return [
            r'experience (?:in|with) ([\w\s\+\#\.\-]+)',
            r'proficient (?:in|with|at) ([\w\s\+\#\.\-]+)',
            r'expertise (?:in|with) ([\w\s\+\#\.\-]+)',
            r'knowledge of ([\w\s\+\#\.\-]+)',
            r'skilled (?:in|at|with) ([\w\s\+\#\.\-]+)',
            r'familiar with ([\w\s\+\#\.\-]+)'
        ]
    
    def get_all_skills(self):
        all_skills = []
        for category_skills in self.skills.values():
            all_skills.extend(category_skills)
        return all_skills
    
    def get_category(self, skill):
        skill_lower = skill.lower()
        for category, skills in self.skills.items():
            if any(s.lower() == skill_lower for s in skills):
                return category
        return 'other'

# ============================================
# SKILL EXTRACTORS
# ============================================

class KeywordSkillExtractor:
    """Basic keyword-based extraction (always available)"""
    
    def __init__(self, skill_db):
        self.skill_db = skill_db
    
    def extract(self, text):
        text_lower = text.lower()
        found_skills = set()
        
        for skill in self.skill_db.get_all_skills():
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(skill)
        
        # Check abbreviations
        for abbr, full in self.skill_db.abbreviations.items():
            if re.search(r'\b' + re.escape(abbr) + r'\b', text):
                found_skills.add(full)
        
        return found_skills

class PatternSkillExtractor:
    """Pattern-based extraction"""
    
    def __init__(self, skill_db):
        self.skill_db = skill_db
    
    def extract(self, text):
        skills = set()
        for pattern in self.skill_db.patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                skill_text = match.group(1).strip()
                # Split by common delimiters
                parts = re.split(r'[,;|/&]|\band\b', skill_text)
                for part in parts:
                    part = part.strip()
                    if len(part) > 2:
                        skills.add(part.title())
        return skills

class NERSkillExtractor:
    """NER-based extraction using spaCy"""
    
    def __init__(self, nlp_model, skill_db):
        self.nlp = nlp_model
        self.skill_db = skill_db
    
    def extract(self, text):
        if not self.nlp:
            return set()
        
        doc = self.nlp(text.lower())
        skills = set()
        
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'LANGUAGE']:
                skill_text = ent.text.strip()
                if self._is_valid_skill(skill_text):
                    skills.add(skill_text.title())
        
        return skills
    
    def _is_valid_skill(self, text):
        if len(text) < 2:
            return False
        all_skills_lower = [s.lower() for s in self.skill_db.get_all_skills()]
        return text.lower() in all_skills_lower

class AdvancedSkillExtractor:
    """Combines all extraction methods"""
    
    def __init__(self, nlp_model, skill_db):
        self.skill_db = skill_db
        self.keyword_extractor = KeywordSkillExtractor(skill_db)
        self.pattern_extractor = PatternSkillExtractor(skill_db)
        self.ner_extractor = NERSkillExtractor(nlp_model, skill_db) if nlp_model else None
    
    def extract_all(self, text):
        """Extract using all available methods"""
        results = {
            'keywords': self.keyword_extractor.extract(text),
            'patterns': self.pattern_extractor.extract(text)
        }
        
        if self.ner_extractor:
            results['ner'] = self.ner_extractor.extract(text)
        
        return results
    
    def extract_combined(self, text):
        """Get combined results from all methods"""
        all_results = self.extract_all(text)
        combined = set()
        
        for method_skills in all_results.values():
            combined.update(method_skills)
        
        return combined

# ============================================
# SKILL GAP ANALYZER
# ============================================

class SkillGapAnalyzer:
    """Analyze skill gaps"""
    
    def __init__(self, sentence_model, skill_db):
        self.sentence_model = sentence_model
        self.skill_db = skill_db
    
    def calculate_match(self, resume_skills, job_skills):
        """Calculate exact match"""
        matched = resume_skills.intersection(job_skills)
        missing = job_skills - resume_skills
        extra = resume_skills - job_skills
        match_pct = (len(matched) / len(job_skills) * 100) if job_skills else 0
        
        return {
            'matched': matched,
            'missing': missing,
            'extra': extra,
            'match_percentage': match_pct,
            'matched_count': len(matched),
            'missing_count': len(missing),
            'extra_count': len(extra)
        }
    
    def calculate_semantic_similarity(self, resume_skills, job_skills):
        """Calculate semantic similarity"""
        if not self.sentence_model or not resume_skills or not job_skills:
            return {}
        
        try:
            resume_list = list(resume_skills)
            job_list = list(job_skills)
            
            resume_emb = self.sentence_model.encode(resume_list, show_progress_bar=False)
            job_emb = self.sentence_model.encode(job_list, show_progress_bar=False)
            
            similarity_matrix = cosine_similarity(resume_emb, job_emb)
            
            semantic_matches = []
            for i, r_skill in enumerate(resume_list):
                for j, j_skill in enumerate(job_list):
                    sim = similarity_matrix[i][j]
                    if sim > 0.7 and r_skill.lower() != j_skill.lower():
                        semantic_matches.append({
                            'resume_skill': r_skill,
                            'job_skill': j_skill,
                            'similarity': float(sim)
                        })
            
            semantic_matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            return {
                'semantic_matches': semantic_matches,
                'average_similarity': float(np.mean(similarity_matrix))
            }
        except:
            return {}
    
    def categorize_skills(self, skills):
        """Categorize skills"""
        categorized = defaultdict(list)
        
        for skill in skills:
            category = self.skill_db.get_category(skill)
            categorized[category].append(skill)
        
        return dict(categorized)

# ============================================
# UI COMPONENTS
# ============================================

st.set_page_config(
    page_title="AI Skill Gap Analyzer",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Poppins', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .skill-tag {
        display: inline-block;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        margin: 5px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Advanced AI Skill Gap Analyzer</h1>
        <p>Multi-Model NLP-Powered Resume Analysis</p>
    </div>
    """, unsafe_allow_html=True)

def create_skill_tags(skills):
    if not skills:
        return "<p style='color: #999;'>No skills detected</p>"
    
    html = '<div style="margin-top: 1rem;">'
    for skill in sorted(list(skills))[:50]:
        html += f'<span class="skill-tag">{skill}</span>'
    if len(skills) > 50:
        html += f'<span class="skill-tag" style="background: #95a5a6;">+{len(skills)-50} more</span>'
    html += '</div>'
    return html

def create_gap_chart(analysis):
    fig = go.Figure(data=[go.Pie(
        labels=['Matched', 'Missing', 'Extra'],
        values=[analysis['matched_count'], analysis['missing_count'], analysis['extra_count']],
        hole=0.5,
        marker=dict(colors=['#38ef7d', '#ff6b6b', '#4facfe']),
        textinfo='label+percent'
    )])
    
    fig.update_layout(
        title=f"Match Score: {analysis['match_percentage']:.1f}%",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=14),
        height=400
    )
    
    return fig

# ============================================
# MAIN APP
# ============================================

def main():
    render_header()
    
    # Initialize models
    if 'models_loaded' not in st.session_state:
        with st.spinner("🚀 Loading AI models..."):
            st.session_state.nlp = load_spacy_model()
            st.session_state.sentence_model = load_sentence_transformer()
            st.session_state.models_loaded = True
    
    # Initialize components
    skill_db = SkillDatabase()
    extractor = AdvancedSkillExtractor(st.session_state.nlp, skill_db)
    analyzer = SkillGapAnalyzer(st.session_state.sentence_model, skill_db)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Extract", "📊 Gap Analysis", "🔬 Advanced Analysis"])
    
    with tab1:
        st.markdown("### 📄 Document Upload & Skill Extraction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="glass-card"><h4>📄 Resume</h4></div>', unsafe_allow_html=True)
            
            input_method = st.radio("Input method:", ["Paste Text", "Upload File"], key="resume_input")
            
            resume_text = None
            if input_method == "Paste Text":
                resume_text = st.text_area("Paste resume text", height=250, key="resume_text")
            else:
                resume_file = st.file_uploader("Upload resume", type=['pdf', 'docx', 'txt'], key="resume_file")
                if resume_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{resume_file.name.split('.')[-1]}") as tmp:
                        tmp.write(resume_file.getvalue())
                        tmp_path = tmp.name
                    
                    try:
                        ext = resume_file.name.split('.')[-1].lower()
                        if ext == 'txt':
                            resume_text = read_txt_file(tmp_path)
                        elif ext == 'pdf':
                            resume_text = read_pdf_file(tmp_path)
                        elif ext == 'docx':
                            resume_text = read_docx_file(tmp_path)
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
            
            if st.button("🔍 Extract Resume Skills", key="btn_resume"):
                if resume_text:
                    with st.spinner("Analyzing..."):
                        skills = extractor.extract_combined(resume_text)
                        st.session_state.resume_skills = skills
                        st.session_state.resume_text = resume_text
                        
                        st.success(f"✅ Found {len(skills)} skills!")
                        st.markdown(create_skill_tags(skills), unsafe_allow_html=True)
                else:
                    st.warning("Please provide resume text")
        
        with col2:
            st.markdown('<div class="glass-card"><h4>💼 Job Description</h4></div>', unsafe_allow_html=True)
            
            jd_input_method = st.radio("Input method:", ["Paste Text", "Upload File"], key="jd_input")
            
            jd_text = None
            if jd_input_method == "Paste Text":
                jd_text = st.text_area("Paste job description", height=250, key="jd_text")
            else:
                jd_file = st.file_uploader("Upload JD", type=['pdf', 'docx', 'txt'], key="jd_file")
                if jd_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{jd_file.name.split('.')[-1]}") as tmp:
                        tmp.write(jd_file.getvalue())
                        tmp_path = tmp.name
                    
                    try:
                        ext = jd_file.name.split('.')[-1].lower()
                        if ext == 'txt':
                            jd_text = read_txt_file(tmp_path)
                        elif ext == 'pdf':
                            jd_text = read_pdf_file(tmp_path)
                        elif ext == 'docx':
                            jd_text = read_docx_file(tmp_path)
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
            
            if st.button("🔍 Extract JD Skills", key="btn_jd"):
                if jd_text:
                    with st.spinner("Analyzing..."):
                        skills = extractor.extract_combined(jd_text)
                        st.session_state.jd_skills = skills
                        st.session_state.jd_text = jd_text
                        
                        st.success(f"✅ Found {len(skills)} required skills!")
                        st.markdown(create_skill_tags(skills), unsafe_allow_html=True)
                else:
                    st.warning("Please provide job description")
    
    with tab2:
        st.markdown("### 🎯 Skill Gap Analysis")
        
        if 'resume_skills' in st.session_state and 'jd_skills' in st.session_state:
            analysis = analyzer.calculate_match(
                st.session_state.resume_skills,
                st.session_state.jd_skills
            )
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Match %", f"{analysis['match_percentage']:.1f}%")
            col2.metric("Matched", analysis['matched_count'])
            col3.metric("Missing", analysis['missing_count'])
            col4.metric("Extra", analysis['extra_count'])
            
            # Chart
            st.plotly_chart(create_gap_chart(analysis), use_container_width=True)
            
            # Detailed breakdown
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### ✅ Matched Skills")
                st.markdown(create_skill_tags(analysis['matched']), unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### ❌ Missing Skills")
                st.markdown(create_skill_tags(analysis['missing']), unsafe_allow_html=True)
            
            with col3:
                st.markdown("#### 🌟 Additional Skills")
                st.markdown(create_skill_tags(analysis['extra']), unsafe_allow_html=True)
            
            # Semantic similarity
            if st.session_state.sentence_model:
                st.markdown("---")
                st.markdown("### 🧠 Semantic Similarity Analysis")
                
                semantic = analyzer.calculate_semantic_similarity(
                    st.session_state.resume_skills,
                    st.session_state.jd_skills
                )
                
                if semantic and 'semantic_matches' in semantic:
                    if semantic['semantic_matches']:
                        df = pd.DataFrame(semantic['semantic_matches'])
                        df['similarity'] = df['similarity'].apply(lambda x: f"{x:.1%}")
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No semantic matches (all are exact matches)")
        else:
            st.info("👆 Please extract skills from both resume and job description first")
    
    with tab3:
        st.markdown("### 🔬 Advanced Extraction Analysis")
        
        if 'resume_text' in st.session_state:
            text_choice = st.radio("Analyze:", ["Resume", "Job Description"], horizontal=True)
            text = st.session_state.resume_text if text_choice == "Resume" else st.session_state.get('jd_text', '')
            
            if text and st.button("🔬 Deep Analysis"):
                with st.spinner("Running analysis..."):
                    all_methods = extractor.extract_all(text)
                    
                    st.markdown("#### Extraction Methods Comparison")
                    
                    methods_df = pd.DataFrame([
                        {'Method': method.upper(), 'Skills Found': len(skills)}
                        for method, skills in all_methods.items()
                    ])
                    st.dataframe(methods_df, use_container_width=True, hide_index=True)
                    
                    # Detailed results
                    st.markdown("---")
                    for method, skills in all_methods.items():
                        if len(skills) > 0:
                            with st.expander(f"{method.upper()} - {len(skills)} skills"):
                                st.markdown(create_skill_tags(skills), unsafe_allow_html=True)
                    
                    # Category breakdown
                    combined = extractor.extract_combined(text)
                    categorized = analyzer.categorize_skills(combined)
                    
                    st.markdown("---")
                    st.markdown("#### 📊 Skills by Category")
                    
                    for category, skills in categorized.items():
                        if skills:
                            st.markdown(f"**{category.replace('_', ' ').title()}**")
                            st.markdown(create_skill_tags(skills), unsafe_allow_html=True)
        else:
            st.info("👆 Extract skills first to use advanced features")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Features")
        st.markdown("""
        **Extraction Methods:**
        - 🔤 Named Entity Recognition
        - 🎯 Keyword Matching
        - 🔍 Pattern Matching
        - 🧠 Semantic Analysis
        
        **Analysis:**
        - ✅ Exact Matching
        - 🧠 Semantic Similarity
        - 📊 Categorization
        - 📈 Gap Visualization
        """)
        
        if 'resume_skills' in st.session_state and 'jd_skills' in st.session_state:
            st.markdown("---")
            st.markdown("### 📊 Quick Stats")
            st.metric("Resume Skills", len(st.session_state.resume_skills))
            st.metric("JD Requirements", len(st.session_state.jd_skills))
            
            match_pct = len(st.session_state.resume_skills.intersection(st.session_state.jd_skills)) / len(st.session_state.jd_skills) * 100 if st.session_state.jd_skills else 0
            st.metric("Match Rate", f"{match_pct:.1f}%")
    
    # Export
    if 'resume_skills' in st.session_state and 'jd_skills' in st.session_state:
        st.markdown("---")
        st.markdown("### 📥 Export Results")
        
        analysis = analyzer.calculate_match(
            st.session_state.resume_skills,
            st.session_state.jd_skills
        )
        
        export_data = {
            'date': datetime.now().isoformat(),
            'resume_skills': sorted(list(st.session_state.resume_skills)),
            'jd_skills': sorted(list(st.session_state.jd_skills)),
            'matched': sorted(list(analysis['matched'])),
            'missing': sorted(list(analysis['missing'])),
            'extra': sorted(list(analysis['extra'])),
            'match_percentage': analysis['match_percentage']
        }
        
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            "📋 Download Analysis (JSON)",
            json_str,
            file_name=f"skill_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()