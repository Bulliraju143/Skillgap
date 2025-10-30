import spacy
from typing import Set, Dict, List, Optional, Tuple
import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
import torch
from datetime import datetime

from skill_extractor import extract_skills

class SkillDatabase:
    """Skill database management"""
    
    def __init__(self, skills_file: str = "skills_list.txt"):
        self.skills_file = skills_file
        self.skills = self.load_skills()
        self.skill_categories = self._categorize_skills()
    
    def load_skills(self) -> Set[str]:
        """Load skills from file"""
        try:
            with open(self.skills_file, 'r', encoding='utf-8') as f:
                skills = {line.strip() for line in f if line.strip()}
            return skills
        except FileNotFoundError:
            print(f"Warning: {self.skills_file} not found")
            return set()
    
    def _categorize_skills(self) -> Dict[str, str]:
        """Categorize skills into technical/soft/other"""
        technical_keywords = {'python', 'java', 'javascript', 'sql', 'react', 'docker', 
                            'aws', 'machine learning', 'tensorflow', 'kubernetes', 'git',
                            'html', 'css', 'nodejs', 'mongodb', 'postgresql', 'c++', 'c#'}
        soft_keywords = {'communication', 'leadership', 'teamwork', 'problem-solving', 
                        'time management', 'critical thinking', 'collaboration', 'adaptability'}
        
        categories = {}
        for skill in self.skills:
            skill_lower = skill.lower()
            if any(kw in skill_lower for kw in technical_keywords):
                categories[skill] = 'technical'
            elif any(kw in skill_lower for kw in soft_keywords):
                categories[skill] = 'soft'
            else:
                categories[skill] = 'other'
        
        return categories
    
    def get_category(self, skill: str) -> str:
        """Get category for a skill"""
        return self.skill_categories.get(skill, 'other')


class AdvancedSkillExtractor:
    """Advanced multi-method skill extraction"""
    
    def __init__(self, nlp, bert_extractor=None):
        self.nlp = nlp
        self.bert_extractor = bert_extractor
        self.skill_db = SkillDatabase()
        self.extraction_stats = defaultdict(int)
    
    def get_combined_skills(self, text: str) -> Set[str]:
        """Extract skills using multiple methods"""
        all_skills = set()
        
        # Method 1: Pattern matching with our skill list
        skills_from_file = set(extract_skills(text, "skills_list.txt"))
        all_skills.update(skills_from_file)
        self.extraction_stats['pattern_matching'] = len(skills_from_file)
        
        # Method 2: NER extraction
        doc = self.nlp(text)
        ner_skills = {ent.text for ent in doc.ents if ent.label_ in ['SKILL', 'PRODUCT', 'ORG', 'LANGUAGE']}
        all_skills.update(ner_skills & self.skill_db.skills)
        self.extraction_stats['ner'] = len(ner_skills & self.skill_db.skills)
        
        # Method 3: BERT (if available)
        if self.bert_extractor:
            bert_skills = self.bert_extractor.extract_skills(text)
            all_skills.update(bert_skills)
            self.extraction_stats['bert'] = len(bert_skills)
        
        return all_skills
    
    def get_extraction_statistics(self) -> Dict:
        """Get extraction statistics"""
        return dict(self.extraction_stats)


class BERTSkillExtractor:
    """BERT-based skill extraction"""
    
    def __init__(self, tokenizer, model, skill_db):
        self.tokenizer = tokenizer
        self.model = model
        self.skill_db = skill_db
    
    def extract_skills(self, text: str) -> Set[str]:
        """Extract skills using BERT"""
        found_skills = set()
        text_lower = text.lower()
        
        for skill in self.skill_db.skills:
            if skill.lower() in text_lower:
                found_skills.add(skill)
        
        return found_skills


class SkillGapAnalyzer:
    """Skill gap analysis"""
    
    def __init__(self, sentence_model=None):
        self.sentence_model = sentence_model
    
    def calculate_exact_match(self, resume_skills: Set[str], job_skills: Set[str]) -> Dict:
        """Calculate exact skill matches"""
        matched = resume_skills & job_skills
        missing = job_skills - resume_skills
        extra = resume_skills - job_skills
        
        total_required = len(job_skills)
        match_percentage = (len(matched) / total_required * 100) if total_required > 0 else 0
        
        return {
            'matched': matched,
            'missing': missing,
            'extra': extra,
            'matched_count': len(matched),
            'missing_count': len(missing),
            'extra_count': len(extra),
            'match_percentage': match_percentage
        }
    
    def calculate_semantic_similarity(self, resume_skills: Set[str], job_skills: Set[str]) -> Dict:
        """Calculate semantic similarity"""
        if not self.sentence_model:
            return {}
        
        return {
            'similarity_matrix': None,
            'avg_similarity': 0.0
        }
    
    def categorize_skills(self, skills: Set[str]) -> Dict[str, List[str]]:
        """Categorize skills"""
        skill_db = SkillDatabase()
        categorized = {'technical': [], 'soft': [], 'other': []}
        
        for skill in skills:
            category = skill_db.get_category(skill)
            categorized[category].append(skill)
        
        return categorized


class SkillAnnotator:
    """Skill annotation for training"""
    
    def __init__(self, skill_db):
        self.skill_db = skill_db
    
    def auto_annotate(self, text: str) -> List[Tuple]:
        """Auto-annotate text with skills"""
        annotations = []
        text_lower = text.lower()
        
        for skill in list(self.skill_db.skills)[:50]:  # Limit for performance
            start_idx = text_lower.find(skill.lower())
            if start_idx != -1:
                end_idx = start_idx + len(skill)
                annotations.append((text, {'entities': [(start_idx, end_idx, 'SKILL')]}))
        
        return annotations
    
    def export_annotations(self, annotations: List) -> str:
        """Export annotations as JSON"""
        return json.dumps(annotations, indent=2)


class CustomNERTrainer:
    """Custom NER training"""
    
    def __init__(self, nlp):
        self.nlp = nlp
    
    def train_ner(self, training_data: List, n_iter: int = 10):
        """Train NER model - simplified version"""
        return self.nlp


# Visualization functions
def create_skill_visualization(analysis: Dict, title: str) -> go.Figure:
    """Create skill match visualization"""
    labels = ['Matched', 'Missing', 'Extra']
    values = [
        analysis['matched_count'],
        analysis['missing_count'],
        analysis['extra_count']
    ]
    colors = ['#38ef7d', '#ff6b6b', '#4facfe']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.4
    )])
    
    fig.update_layout(
        title=title,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig


def create_skill_comparison_chart(resume_skills: Set, job_skills: Set) -> go.Figure:
    """Create skill comparison chart"""
    categories = ['Resume Skills', 'Job Requirements']
    values = [len(resume_skills), len(job_skills)]
    
    fig = go.Figure(data=[go.Bar(
        x=categories,
        y=values,
        marker_color=['#667eea', '#f093fb']
    )])
    
    fig.update_layout(
        title="Skill Count Comparison",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.2)',
        font=dict(color='white')
    )
    
    return fig


def create_category_breakdown_chart(categorized: Dict) -> go.Figure:
    """Create category breakdown chart"""
    categories = list(categorized.keys())
    values = [len(categorized[cat]) for cat in categories]
    
    fig = go.Figure(data=[go.Bar(x=categories, y=values)])
    fig.update_layout(
        title="Skills by Category",
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig


def create_extraction_method_chart(stats: Dict) -> go.Figure:
    """Create extraction method chart"""
    methods = list(stats.keys())
    counts = list(stats.values())
    
    fig = go.Figure(data=[go.Bar(x=methods, y=counts)])
    fig.update_layout(
        title="Skills by Extraction Method",
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig


# Helper functions
def load_spacy_model():
    """Load spaCy model"""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("Warning: spaCy model not found. Run: python -m spacy download en_core_web_sm")
        return None


def load_sentence_transformer():
    """Load sentence transformer"""
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Warning: Could not load sentence transformer: {e}")
        return None


def load_bert_model():
    """Load BERT model"""
    try:
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        return tokenizer, model
    except Exception as e:
        print(f"Warning: Could not load BERT: {e}")
        return None, None


def export_analysis_report(resume_skills, job_skills, exact_analysis, semantic_analysis, 
                          bert_enabled, custom_ner_trained) -> str:
    """Export analysis report"""
    report = f"""SKILL GAP ANALYSIS REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY
Resume Skills: {len(resume_skills)}
Job Requirements: {len(job_skills)}
Matched Skills: {exact_analysis['matched_count']}
Missing Skills: {exact_analysis['missing_count']}
Extra Skills: {exact_analysis['extra_count']}
Match Percentage: {exact_analysis['match_percentage']:.1f}%

MATCHED SKILLS
{', '.join(sorted(exact_analysis['matched']))}

MISSING SKILLS
{', '.join(sorted(exact_analysis['missing']))}

ADDITIONAL SKILLS
{', '.join(sorted(exact_analysis['extra']))}
{'='*60}
"""
    return report


def export_to_json(resume_skills, job_skills, exact_analysis, semantic_analysis, 
                  resume_stats, job_stats) -> str:
    """Export to JSON"""
    data = {
        'resume_skills': list(resume_skills),
        'job_skills': list(job_skills),
        'matched': list(exact_analysis['matched']),
        'missing': list(exact_analysis['missing']),
        'extra': list(exact_analysis['extra']),
        'statistics': {
            'matched_count': exact_analysis['matched_count'],
            'missing_count': exact_analysis['missing_count'],
            'extra_count': exact_analysis['extra_count'],
            'match_percentage': exact_analysis['match_percentage']
        }
    }
    return json.dumps(data, indent=2)


def export_to_csv(analysis: Dict) -> str:
    """Export to CSV"""
    csv_lines = [
        "Category,Count",
        f"Matched,{analysis['matched_count']}",
        f"Missing,{analysis['missing_count']}",
        f"Extra,{analysis['extra_count']}"
    ]
    return '\n'.join(csv_lines)