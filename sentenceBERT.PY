"""
Milestone 3: Skill Gap Analysis & Similarity Matching
Advanced BERT-based semantic similarity with comprehensive gap analysis

Student Name: Happy Patel
Student ID: 15

Features:
✅ BERT-based semantic similarity using Sentence-BERT
✅ Cosine similarity computation
✅ Multi-level skill gap identification
✅ Importance-based ranking system
✅ Interactive similarity matrices
✅ Comprehensive visualizations
✅ Multiple export formats
✅ Learning path recommendations
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime
import json
import logging
import re


# ==================== DATA CLASSES ====================

@dataclass
class SkillMatch:
    """Data class for skill match information"""
    jd_skill: str
    resume_skill: str
    similarity: float
    category: str
    confidence_level: str
    priority: str = "MEDIUM"
    
    def to_dict(self) -> Dict:
        return {
            'jd_skill': self.jd_skill,
            'resume_skill': self.resume_skill,
            'similarity': self.similarity,
            'category': self.category,
            'confidence_level': self.confidence_level,
            'priority': self.priority
        }


@dataclass
class GapAnalysisResult:
    """Complete gap analysis results"""
    matched_skills: List[SkillMatch]
    partial_matches: List[SkillMatch]
    missing_skills: List[SkillMatch]
    overall_score: float
    category_scores: Dict[str, float]
    similarity_matrix: np.ndarray
    resume_skills: List[str]
    jd_skills: List[str]
    
    def get_statistics(self) -> Dict:
        total = len(self.jd_skills)
        return {
            'total_required_skills': total,
            'matched_count': len(self.matched_skills),
            'partial_count': len(self.partial_matches),
            'missing_count': len(self.missing_skills),
            'match_percentage': (len(self.matched_skills) / total * 100) if total > 0 else 0,
            'overall_score': self.overall_score * 100
        }


# ==================== BERT ENCODER ====================

class SentenceBERTEncoder:
    """Handles BERT embedding generation using Sentence-BERT"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize Sentence-BERT model"""
        self.model_name = model_name
        self.logger = self._setup_logger()
        self.embedding_cache = {}
        
        try:
            self.logger.info(f"Loading model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def encode_skills(self, skills: List[str], use_cache: bool = True, 
                     show_progress: bool = False) -> np.ndarray:
        """Encode list of skills into embeddings"""
        if not skills:
            raise ValueError("Skills list cannot be empty")
        
        if use_cache:
            cached_embeddings = []
            uncached_skills = []
            uncached_indices = []
            
            for i, skill in enumerate(skills):
                if skill in self.embedding_cache:
                    cached_embeddings.append(self.embedding_cache[skill])
                else:
                    uncached_skills.append(skill)
                    uncached_indices.append(i)
            
            if uncached_skills:
                new_embeddings = self.model.encode(
                    uncached_skills, 
                    show_progress_bar=show_progress,
                    batch_size=32
                )
                
                for skill, embedding in zip(uncached_skills, new_embeddings):
                    self.embedding_cache[skill] = embedding
                
                all_embeddings = [None] * len(skills)
                cached_idx = 0
                uncached_idx = 0
                
                for i in range(len(skills)):
                    if i in uncached_indices:
                        all_embeddings[i] = new_embeddings[uncached_idx]
                        uncached_idx += 1
                    else:
                        all_embeddings[i] = cached_embeddings[cached_idx]
                        cached_idx += 1
                
                return np.array(all_embeddings)
            else:
                return np.array(cached_embeddings)
        else:
            embeddings = self.model.encode(
                skills, 
                show_progress_bar=show_progress,
                batch_size=32
            )
            return embeddings
    
    def get_embedding_for_skill(self, skill: str) -> np.ndarray:
        """Get embedding for a single skill"""
        if skill in self.embedding_cache:
            return self.embedding_cache[skill]
        
        embedding = self.model.encode([skill])[0]
        self.embedding_cache[skill] = embedding
        return embedding
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.embedding_cache.clear()
        self.logger.info("Embedding cache cleared")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('BERTEncoder')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


# ==================== SIMILARITY CALCULATOR ====================

class SimilarityCalculator:
    """Compute similarity scores between skills"""
    
    def __init__(self):
        self.logger = self._setup_logger()
    
    def compute_cosine_similarity(self, embedding1: np.ndarray, 
                                  embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        return float(similarity)
    
    def compute_similarity_matrix(self, resume_embeddings: np.ndarray,
                                  jd_embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise similarity matrix"""
        self.logger.info(f"Computing similarity matrix: {resume_embeddings.shape} x {jd_embeddings.shape}")
        similarity_matrix = cosine_similarity(resume_embeddings, jd_embeddings)
        self.logger.info(f"Similarity matrix computed: {similarity_matrix.shape}")
        return similarity_matrix
    
    def find_best_matches(self, similarity_matrix: np.ndarray, 
                         threshold: float = 0.5) -> List[Tuple[int, int, float]]:
        """Find best matches above threshold"""
        matches = []
        n_resume, n_jd = similarity_matrix.shape
        
        for jd_idx in range(n_jd):
            best_resume_idx = np.argmax(similarity_matrix[:, jd_idx])
            best_similarity = similarity_matrix[best_resume_idx, jd_idx]
            
            if best_similarity >= threshold:
                matches.append((best_resume_idx, jd_idx, best_similarity))
        
        return matches
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('SimilarityCalculator')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


# ==================== SKILL GAP ANALYZER ====================

class SkillGapAnalyzer:
    """Main skill gap analysis engine"""
    
    def __init__(self, encoder: SentenceBERTEncoder, calculator: SimilarityCalculator,
                 strong_threshold: float = 0.80, partial_threshold: float = 0.50):
        """Initialize gap analyzer"""
        self.encoder = encoder
        self.calculator = calculator
        self.strong_threshold = strong_threshold
        self.partial_threshold = partial_threshold
        self.logger = self._setup_logger()
    
    def analyze(self, resume_skills: List[str], jd_skills: List[str],
               skill_categories: Optional[Dict[str, str]] = None) -> GapAnalysisResult:
        """Perform complete gap analysis"""
        self.logger.info(f"Starting gap analysis: {len(resume_skills)} resume skills vs {len(jd_skills)} JD skills")
        
        if not resume_skills or not jd_skills:
            raise ValueError("Both resume_skills and jd_skills must be non-empty")
        
        # Generate embeddings
        self.logger.info("Step 1: Generating BERT embeddings...")
        resume_embeddings = self.encoder.encode_skills(resume_skills, show_progress=True)
        jd_embeddings = self.encoder.encode_skills(jd_skills, show_progress=True)
        
        # Compute similarity matrix
        self.logger.info("Step 2: Computing similarity matrix...")
        similarity_matrix = self.calculator.compute_similarity_matrix(
            resume_embeddings, 
            jd_embeddings
        )
        
        # Classify matches
        self.logger.info("Step 3: Classifying skill matches...")
        matched_skills = []
        partial_matches = []
        missing_skills = []
        
        for jd_idx, jd_skill in enumerate(jd_skills):
            best_resume_idx = np.argmax(similarity_matrix[:, jd_idx])
            best_similarity = float(similarity_matrix[best_resume_idx, jd_idx])
            resume_skill = resume_skills[best_resume_idx]
            
            category = skill_categories.get(jd_skill, 'other') if skill_categories else 'other'
            
            if best_similarity >= self.strong_threshold:
                match = SkillMatch(
                    jd_skill=jd_skill,
                    resume_skill=resume_skill,
                    similarity=best_similarity,
                    category='STRONG_MATCH',
                    confidence_level='HIGH',
                    priority='LOW'
                )
                matched_skills.append(match)
                
            elif best_similarity >= self.partial_threshold:
                match = SkillMatch(
                    jd_skill=jd_skill,
                    resume_skill=resume_skill,
                    similarity=best_similarity,
                    category='PARTIAL_MATCH',
                    confidence_level='MEDIUM',
                    priority='MEDIUM'
                )
                partial_matches.append(match)
                
            else:
                match = SkillMatch(
                    jd_skill=jd_skill,
                    resume_skill=resume_skill,
                    similarity=best_similarity,
                    category='MISSING',
                    confidence_level='LOW',
                    priority='HIGH'
                )
                missing_skills.append(match)
        
        # Calculate scores
        overall_score = self._calculate_overall_score(similarity_matrix)
        category_scores = self._calculate_category_scores(
            matched_skills, partial_matches, missing_skills
        )
        
        self.logger.info(f"Analysis complete: {len(matched_skills)} matched, "
                        f"{len(partial_matches)} partial, {len(missing_skills)} missing")
        
        return GapAnalysisResult(
            matched_skills=matched_skills,
            partial_matches=partial_matches,
            missing_skills=missing_skills,
            overall_score=overall_score,
            category_scores=category_scores,
            similarity_matrix=similarity_matrix,
            resume_skills=resume_skills,
            jd_skills=jd_skills
        )
    
    def _calculate_overall_score(self, similarity_matrix: np.ndarray) -> float:
        """Calculate overall match score"""
        max_similarities = similarity_matrix.max(axis=0)
        overall_score = float(np.mean(max_similarities))
        return overall_score
    
    def _calculate_category_scores(self, matched: List[SkillMatch],
                                   partial: List[SkillMatch],
                                   missing: List[SkillMatch]) -> Dict[str, float]:
        """Calculate scores by category"""
        category_scores = {}
        
        all_skills = matched + partial + missing
        categories = set(skill.category for skill in all_skills)
        
        for category in categories:
            cat_skills = [s for s in all_skills if s.category == category]
            if cat_skills:
                avg_similarity = np.mean([s.similarity for s in cat_skills])
                category_scores[category] = float(avg_similarity)
        
        return category_scores
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('SkillGapAnalyzer')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


# ==================== SKILL RANKER ====================

class SkillRanker:
    """Rank skills by importance and priority"""
    
    def __init__(self):
        self.logger = self._setup_logger()
    
    def rank_by_importance(self, skills: List[SkillMatch], 
                          importance_weights: Optional[Dict[str, float]] = None) -> List[SkillMatch]:
        """Rank skills by importance"""
        if not importance_weights:
            importance_weights = {
                'similarity': 0.4,
                'category': 0.3,
                'priority': 0.3
            }
        
        def calculate_importance_score(skill: SkillMatch) -> float:
            sim_score = skill.similarity
            
            if skill.category == 'MISSING':
                cat_score = 1.0
            elif skill.category == 'PARTIAL_MATCH':
                cat_score = 0.6
            else:
                cat_score = 0.2
            
            priority_map = {'HIGH': 1.0, 'MEDIUM': 0.6, 'LOW': 0.3}
            pri_score = priority_map.get(skill.priority, 0.5)
            
            importance = (
                importance_weights['similarity'] * sim_score +
                importance_weights['category'] * cat_score +
                importance_weights['priority'] * pri_score
            )
            
            return importance
        
        ranked_skills = sorted(skills, key=calculate_importance_score, reverse=True)
        
        self.logger.info(f"Ranked {len(skills)} skills by importance")
        return ranked_skills
    
    def categorize_by_urgency(self, missing_skills: List[SkillMatch]) -> Dict[str, List[SkillMatch]]:
        """Categorize missing skills by urgency"""
        categorized = {
            'critical': [],
            'important': [],
            'beneficial': []
        }
        
        for skill in missing_skills:
            if skill.priority == 'HIGH' or skill.similarity < 0.3:
                categorized['critical'].append(skill)
            elif skill.priority == 'MEDIUM' or skill.similarity < 0.4:
                categorized['important'].append(skill)
            else:
                categorized['beneficial'].append(skill)
        
        return categorized
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('SkillRanker')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
        return logger


# ==================== VISUALIZATIONS ====================

class GapVisualizer:
    """Create visualizations for gap analysis"""
    
    @staticmethod
    def create_similarity_heatmap(similarity_matrix: np.ndarray,
                                 resume_skills: List[str],
                                 jd_skills: List[str]) -> go.Figure:
        """Create interactive similarity heatmap"""
        max_display = 20
        display_resume = resume_skills[:max_display]
        display_jd = jd_skills[:max_display]
        display_matrix = similarity_matrix[:max_display, :max_display]
        
        fig = go.Figure(data=go.Heatmap(
            z=display_matrix,
            x=display_jd,
            y=display_resume,
            colorscale='RdYlGn',
            zmid=0.5,
            text=np.round(display_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(
                title="Similarity",
                titleside="right",
                tickmode="linear",
                tick0=0,
                dtick=0.2
            )
        ))
        
        fig.update_layout(
            title=f"Skill Similarity Heatmap (Top {min(max_display, len(resume_skills))} x {min(max_display, len(jd_skills))} skills)",
            xaxis_title="Job Description Skills",
            yaxis_title="Resume Skills",
            height=600,
            width=900,
            xaxis={'side': 'bottom'},
            yaxis={'autorange': 'reversed'}
        )
        
        return fig
    
    @staticmethod
    def create_match_distribution_pie(analysis_result: GapAnalysisResult) -> go.Figure:
        """Create pie chart for match distribution"""
        stats = analysis_result.get_statistics()
        
        labels = ['Strong Matches', 'Partial Matches', 'Missing Skills']
        values = [stats['matched_count'], stats['partial_count'], stats['missing_count']]
        colors = ['#28a745', '#ffc107', '#dc3545']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            hole=0.3,
            textposition='auto',
            textinfo='label+percent+value'
        )])
        
        fig.update_layout(
            title="Skill Match Distribution",
            height=500,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_skill_comparison_bar(analysis_result: GapAnalysisResult, top_n: int = 15) -> go.Figure:
        """Create bar chart comparing skill similarities"""
        all_matches = (analysis_result.matched_skills + 
                      analysis_result.partial_matches + 
                      analysis_result.missing_skills)
        
        all_matches_sorted = sorted(all_matches, key=lambda x: x.similarity, reverse=True)[:top_n]
        
        skills = [m.jd_skill for m in all_matches_sorted]
        similarities = [m.similarity * 100 for m in all_matches_sorted]
        colors_map = {'STRONG_MATCH': '#28a745', 'PARTIAL_MATCH': '#ffc107', 'MISSING': '#dc3545'}
        colors = [colors_map[m.category] for m in all_matches_sorted]
        
        fig = go.Figure(data=[go.Bar(
            y=skills,
            x=similarities,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{s:.1f}%" for s in similarities],
            textposition='auto'
        )])
        
        fig.update_layout(
            title=f"Top {top_n} Skills by Similarity Score",
            xaxis_title="Similarity Score (%)",
            yaxis_title="Skills",
            height=600,
            yaxis=dict(autorange="reversed"),
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_gap_priority_chart(missing_skills: List[SkillMatch]) -> go.Figure:
        """Create chart showing gap priorities"""
        if not missing_skills:
            fig = go.Figure()
            fig.update_layout(
                title="No Missing Skills",
                annotations=[dict(
                    text="All required skills are matched!",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=20)
                )]
            )
            return fig
        
        sorted_skills = sorted(missing_skills, key=lambda x: x.similarity)[:15]
        
        skills = [s.jd_skill for s in sorted_skills]
        similarities = [s.similarity * 100 for s in sorted_skills]
        priorities = [s.priority for s in sorted_skills]
        
        priority_colors = {
            'HIGH': '#dc3545',
            'MEDIUM': '#ffc107',
            'LOW': '#28a745'
        }
        colors = [priority_colors[p] for p in priorities]
        
        fig = go.Figure(data=[go.Bar(
            y=skills,
            x=similarities,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{s:.1f}% - {p}" for s, p in zip(similarities, priorities)],
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Missing Skills by Priority",
            xaxis_title="Current Similarity (%)",
            yaxis_title="Skills",
            height=500,
            yaxis=dict(autorange="reversed")
        )
        
        return fig
    
    @staticmethod
    def create_overall_score_gauge(overall_score: float) -> go.Figure:
        """Create gauge chart for overall match score"""
        score_percentage = overall_score * 100
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score_percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Match Score", 'font': {'size': 24}},
            delta={'reference': 70, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': '#ffcccc'},
                    {'range': [40, 70], 'color': '#ffffcc'},
                    {'range': [70, 100], 'color': '#ccffcc'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        return fig


# ==================== REPORT GENERATOR ====================

class ReportGenerator:
    """Generate comprehensive reports"""
    
    def __init__(self):
        self.timestamp = datetime.now()
    
    def generate_text_report(self, analysis_result: GapAnalysisResult) -> str:
        """Generate detailed text report"""
        stats = analysis_result.get_statistics()
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SKILL GAP ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nGenerated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("-" * 80)
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Overall Match Score: {stats['overall_score']:.1f}%")
        report_lines.append(f"Total Required Skills: {stats['total_required_skills']}")
        report_lines.append(f"Matched Skills: {stats['matched_count']} ({stats['match_percentage']:.1f}%)")
        report_lines.append(f"Partial Matches: {stats['partial_count']}")
        report_lines.append(f"Missing Skills: {stats['missing_count']}")
        report_lines.append("")
        
        # Strong Matches
        if analysis_result.matched_skills:
            report_lines.append("-" * 80)
            report_lines.append("✓ STRONG MATCHES (Similarity ≥ 80%)")
            report_lines.append("-" * 80)
            for match in analysis_result.matched_skills:
                report_lines.append(f"  • {match.jd_skill}")
                report_lines.append(f"    Resume: {match.resume_skill}")
                report_lines.append(f"    Similarity: {match.similarity*100:.1f}%")
                report_lines.append("")
        
        # Partial Matches
        if analysis_result.partial_matches:
            report_lines.append("-" * 80)
            report_lines.append("⚠ PARTIAL MATCHES (Similarity 50-80%)")
            report_lines.append("-" * 80)
            for match in analysis_result.partial_matches:
                report_lines.append(f"  • {match.jd_skill}")
                report_lines.append(f"    Closest: {match.resume_skill}")
                report_lines.append(f"    Similarity: {match.similarity*100:.1f}%")
                report_lines.append(f"    Recommendation: Strengthen knowledge in {match.jd_skill}")
                report_lines.append("")
        
        # Missing Skills
        if analysis_result.missing_skills:
            report_lines.append("-" * 80)
            report_lines.append("✗ CRITICAL GAPS (Similarity < 50%)")
            report_lines.append("-" * 80)
            for match in analysis_result.missing_skills:
                report_lines.append(f"  • {match.jd_skill} - {match.priority} PRIORITY")
                report_lines.append(f"    Current closest: {match.resume_skill} ({match.similarity*100:.1f}%)")
                report_lines.append(f"    Action: Acquire {match.jd_skill} through training/certification")
                report_lines.append("")
        
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def generate_csv_report(self, analysis_result: GapAnalysisResult) -> str:
        """Generate CSV report"""
        data = []
        
        for match in analysis_result.matched_skills:
            data.append({
                'JD Skill': match.jd_skill,
                'Resume Skill': match.resume_skill,
                'Similarity (%)': f"{match.similarity*100:.2f}",
                'Category': match.category,
                'Priority': match.priority,
                'Status': 'Matched'
            })
        
        for match in analysis_result.partial_matches:
            data.append({
                'JD Skill': match.jd_skill,
                'Resume Skill': match.resume_skill,
                'Similarity (%)': f"{match.similarity*100:.2f}",
                'Category': match.category,
                'Priority': match.priority,
                'Status': 'Partial'
            })
        
        for match in analysis_result.missing_skills:
            data.append({
                'JD Skill': match.jd_skill,
                'Resume Skill': match.resume_skill,
                'Similarity (%)': f"{match.similarity*100:.2f}",
                'Category': match.category,
                'Priority': match.priority,
                'Status': 'Missing'
            })
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    
    def generate_json_report(self, analysis_result: GapAnalysisResult) -> str:
        """Generate JSON report"""
        stats = analysis_result.get_statistics()
        
        report_data = {
            'timestamp': self.timestamp.isoformat(),
            'statistics': stats,
            'matched_skills': [match.to_dict() for match in analysis_result.matched_skills],
            'partial_matches': [match.to_dict() for match in analysis_result.partial_matches],
            'missing_skills': [match.to_dict() for match in analysis_result.missing_skills],
            'category_scores': analysis_result.category_scores,
            'resume_skills': analysis_result.resume_skills,
            'jd_skills': analysis_result.jd_skills
        }
        
        return json.dumps(report_data, indent=2)


# ==================== LEARNING PATH GENERATOR ====================

class LearningPathGenerator:
    """Generate personalized learning paths for skill gaps"""
    
    def __init__(self):
        self.resource_database = self._initialize_resources()
    
    def _initialize_resources(self) -> Dict:
        """Initialize learning resources database"""
        return {
            'Python': {
                'difficulty': 'Medium',
                'time_estimate': '4-8 weeks',
                'resources': [
                    'Python for Everybody (Coursera)',
                    'Automate the Boring Stuff with Python',
                    'Official Python Tutorial'
                ]
            },
            'Machine Learning': {
                'difficulty': 'Hard',
                'time_estimate': '12-16 weeks',
                'prerequisites': ['Python', 'Statistics'],
                'resources': [
                    'Andrew Ng Machine Learning Course',
                    'Hands-on Machine Learning with Scikit-Learn',
                    'Fast.ai Practical Deep Learning'
                ]
            },
            'TensorFlow': {
                'difficulty': 'Medium',
                'time_estimate': '6-8 weeks',
                'prerequisites': ['Python', 'Machine Learning'],
                'resources': [
                    'TensorFlow Developer Certificate',
                    'Deep Learning Specialization',
                    'TensorFlow Official Tutorials'
                ]
            },
            'AWS': {
                'difficulty': 'Medium',
                'time_estimate': '8-12 weeks',
                'resources': [
                    'AWS Cloud Practitioner Certification',
                    'AWS Solutions Architect Associate',
                    'AWS Free Tier Hands-on Labs'
                ]
            },
            'Docker': {
                'difficulty': 'Medium',
                'time_estimate': '2-4 weeks',
                'resources': [
                    'Docker Official Documentation',
                    'Docker Mastery Course',
                    'Docker for Developers'
                ]
            },
            'Kubernetes': {
                'difficulty': 'Hard',
                'time_estimate': '8-12 weeks',
                'prerequisites': ['Docker', 'Linux'],
                'resources': [
                    'Kubernetes Official Documentation',
                    'Certified Kubernetes Administrator (CKA)',
                    'Kubernetes for Developers'
                ]
            }
        }
    
    def generate_path(self, missing_skills: List[SkillMatch],
                     current_skills: List[str]) -> List[Dict]:
        """Generate learning path for missing skills"""
        learning_plan = []
        
        sorted_skills = sorted(missing_skills, 
                              key=lambda x: (0 if x.priority == 'HIGH' else 1 if x.priority == 'MEDIUM' else 2,
                                           x.similarity))
        
        for skill_match in sorted_skills:
            skill = skill_match.jd_skill
            
            plan_item = {
                'skill': skill,
                'current_similarity': skill_match.similarity,
                'priority': skill_match.priority,
                'difficulty': 'Unknown',
                'time_estimate': 'Varies',
                'resources': [],
                'prerequisites': [],
                'missing_prerequisites': []
            }
            
            if skill in self.resource_database:
                resource_info = self.resource_database[skill]
                plan_item['difficulty'] = resource_info.get('difficulty', 'Unknown')
                plan_item['time_estimate'] = resource_info.get('time_estimate', 'Varies')
                plan_item['resources'] = resource_info.get('resources', [])
                plan_item['prerequisites'] = resource_info.get('prerequisites', [])
                
                missing_prereqs = []
                for prereq in plan_item['prerequisites']:
                    if prereq.lower() not in [s.lower() for s in current_skills]:
                        missing_prereqs.append(prereq)
                
                plan_item['missing_prerequisites'] = missing_prereqs
            else:
                plan_item['resources'] = [f'Search for "{skill}" courses online']
            
            learning_plan.append(plan_item)
        
        return learning_plan
    
    def estimate_weeks(self, time_str: str) -> int:
        """Extract weeks from time estimate string"""
        match = re.search(r'(\d+)-?(\d+)?', time_str)
        if match:
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else start
            return (start + end) // 2
        return 8


# ==================== UTILITY FUNCTIONS ====================

def load_sentence_bert_model(model_name: str = 'all-MiniLM-L6-v2'):
    """Load Sentence-BERT model with error handling"""
    try:
        encoder = SentenceBERTEncoder(model_name)
        return encoder
    except Exception as e:
        print(f"Error loading Sentence-BERT model: {e}")
        return None


def perform_gap_analysis(resume_skills: List[str], 
                        jd_skills: List[str],
                        strong_threshold: float = 0.80,
                        partial_threshold: float = 0.50) -> Optional[GapAnalysisResult]:
    """Convenience function to perform complete gap analysis"""
    try:
        encoder = load_sentence_bert_model()
        if not encoder:
            return None
        
        calculator = SimilarityCalculator()
        analyzer = SkillGapAnalyzer(encoder, calculator, strong_threshold, partial_threshold)
        
        result = analyzer.analyze(resume_skills, jd_skills)
        return result
        
    except Exception as e:
        print(f"Error during gap analysis: {e}")
        return None


def export_all_reports(analysis_result: GapAnalysisResult, base_filename: str = "skill_gap_analysis"):
    """Export all report formats"""
    generator = ReportGenerator()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    reports = {
        'text': generator.generate_text_report(analysis_result),
        'csv': generator.generate_csv_report(analysis_result),
        'json': generator.generate_json_report(analysis_result)
    }
    
    return reports


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Example usage
    print("Milestone 3: Skill Gap Analysis & Similarity Matching")
    print("=" * 60)
    
    # Sample data
    resume_skills = [
        "Python", "Machine Learning", "SQL", "Data Analysis",
        "Pandas", "NumPy", "Scikit-learn", "Git", "Statistics"
    ]
    
    jd_skills = [
        "Python", "Deep Learning", "TensorFlow", "SQL",
        "AWS", "Docker", "Kubernetes", "Data Science",
        "Neural Networks", "Cloud Computing"
    ]
    
    print(f"\nResume Skills: {len(resume_skills)}")
    print(f"Job Description Skills: {len(jd_skills)}")
    print("\nPerforming gap analysis...\n")
    
    # Perform analysis
    result = perform_gap_analysis(resume_skills, jd_skills)
    
    if result:
        stats = result.get_statistics()
        
        print("=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)
        print(f"Overall Score: {stats['overall_score']:.1f}%")
        print(f"Total Required: {stats['total_required_skills']}")
        print(f"Matched: {stats['matched_count']}")
        print(f"Partial: {stats['partial_count']}")
        print(f"Missing: {stats['missing_count']}")
        print(f"Match Rate: {stats['match_percentage']:.1f}%")
        print("=" * 60)
        
        # Generate reports
        print("\nGenerating reports...")
        reports = export_all_reports(result)
        print("✓ Text report generated")
        print("✓ CSV report generated")
        print("✓ JSON report generated")
        
        # Generate learning path
        print("\nGenerating learning path...")
        path_gen = LearningPathGenerator()
        learning_path = path_gen.generate_path(result.missing_skills, resume_skills)
        print(f"✓ Learning path generated with {len(learning_path)} skills")
        
        print("\n✓ Milestone 3 complete!")
    else:
        print("Error: Could not perform gap analysis")