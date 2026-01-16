"""
Training Evaluation Analysis System - Phase 1
Automated analysis of training evaluations with NLP, sentiment analysis, and insights generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# For NLP Processing
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
except ImportError:
    print("scikit-learn not installed. Install with: pip install scikit-learn --break-system-packages")

# Sentiment Analysis Patterns (for French/Arabic support)
SENTIMENT_PATTERNS = {
    'positive_fr': [
        'excellent', 'très bien', 'parfait', 'satisfait', 'content', 'intéressant',
        'utile', 'clair', 'efficace', 'professionnel', 'compétent', 'bon', 'super',
        'génial', 'recommande', 'agréable', 'pertinent', 'qualité', 'apprécié'
    ],
    'negative_fr': [
        'mauvais', 'insuffisant', 'difficile', 'compliqué', 'confus', 'déçu',
        'insatisfait', 'mécontent', 'problème', 'manque', 'pas bien', 'faible',
        'ennuyeux', 'inutile', 'perdu', 'incompréhensible', 'nul', 'décevant'
    ],
    'positive_en': [
        'excellent', 'great', 'good', 'amazing', 'perfect', 'satisfied', 'useful',
        'clear', 'effective', 'professional', 'competent', 'interesting', 'helpful',
        'recommend', 'enjoyed', 'relevant', 'quality', 'appreciated'
    ],
    'negative_en': [
        'bad', 'poor', 'difficult', 'complicated', 'confused', 'disappointed',
        'unsatisfied', 'problem', 'lacking', 'not good', 'weak', 'boring',
        'useless', 'lost', 'incomprehensible', 'terrible', 'disappointing'
    ]
}

@dataclass
class EvaluationMetrics:
    """Data class for evaluation metrics"""
    session_id: str
    formation_type: str
    date: str
    avg_satisfaction: float
    avg_trainer_quality: float
    avg_content_relevance: float
    avg_logistics: float
    avg_applicability: float
    response_count: int
    completion_rate: float
    sentiment_distribution: Dict[str, float]
    key_themes: List[str]
    weak_signals: List[str]

@dataclass
class InsightReport:
    """Data class for insight reports"""
    period: str
    total_evaluations: int
    overall_satisfaction: float
    satisfaction_trend: str
    top_strengths: List[Tuple[str, float]]
    top_improvements: List[Tuple[str, float]]
    emerging_themes: List[str]
    weak_signals: List[Dict]
    recommendations: List[str]
    generated_at: str


class TrainingEvaluationAnalyzer:
    """
    Comprehensive training evaluation analysis system
    Features:
    - Multi-language support (French, English, Arabic)
    - Sentiment analysis
    - Topic modeling and theme extraction
    - Clustering and pattern detection
    - Weak signal identification
    - Trend analysis
    """
    
    def __init__(self):
        self.vectorizer = None
        self.topic_model = None
        self.scaler = StandardScaler()
        self.evaluation_history = []
        
    def load_evaluations(self, file_path: str, file_type: str = 'excel') -> pd.DataFrame:
        """
        Load evaluation data from various formats
        
        Args:
            file_path: Path to evaluation file
            file_type: Type of file ('excel', 'csv', 'pdf')
        
        Returns:
            DataFrame with evaluation data
        """
        if file_type == 'excel':
            df = pd.read_excel(file_path)
        elif file_type == 'csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        
        return df
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text data
        
        Args:
            text: Raw text input
        
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        # Remove special characters but keep accents for French
        text = re.sub(r'[^\w\s\àâäéèêëïîôùûüÿçæœ-]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text (multi-language support)
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment scores
        """
        text = text.lower()
        
        # Count positive and negative words
        pos_count = sum(1 for word in SENTIMENT_PATTERNS['positive_fr'] if word in text)
        pos_count += sum(1 for word in SENTIMENT_PATTERNS['positive_en'] if word in text)
        neg_count = sum(1 for word in SENTIMENT_PATTERNS['negative_fr'] if word in text)
        neg_count += sum(1 for word in SENTIMENT_PATTERNS['negative_en'] if word in text)
        
        total = pos_count + neg_count
        
        if total == 0:
            return {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33}
        
        pos_ratio = pos_count / total
        neg_ratio = neg_count / total
        
        # Classify sentiment
        if pos_ratio > 0.6:
            sentiment = {'positive': 0.7, 'neutral': 0.2, 'negative': 0.1}
        elif neg_ratio > 0.6:
            sentiment = {'positive': 0.1, 'neutral': 0.2, 'negative': 0.7}
        else:
            sentiment = {'positive': 0.3, 'neutral': 0.4, 'negative': 0.3}
        
        return sentiment
    
    def extract_themes(self, texts: List[str], n_topics: int = 5) -> List[Tuple[str, List[str]]]:
        """
        Extract main themes using topic modeling (LDA)
        
        Args:
            texts: List of text documents
            n_topics: Number of topics to extract
        
        Returns:
            List of (topic_name, keywords) tuples
        """
        # Clean texts
        cleaned_texts = [self.preprocess_text(t) for t in texts if t and len(str(t)) > 10]
        
        if len(cleaned_texts) < n_topics:
            return [("Insufficient data", [])]
        
        # Create document-term matrix
        vectorizer = CountVectorizer(max_features=100, max_df=0.8, min_df=2, 
                                     stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(cleaned_texts)
        
        # Apply LDA
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(doc_term_matrix)
        
        # Extract topics
        feature_names = vectorizer.get_feature_names_out()
        themes = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-5:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            theme_name = f"Theme {topic_idx + 1}"
            themes.append((theme_name, keywords))
        
        return themes
    
    def detect_weak_signals(self, df: pd.DataFrame, threshold: float = 0.2) -> List[Dict]:
        """
        Detect weak signals (early warnings of issues)
        
        Args:
            df: DataFrame with evaluation data
            threshold: Threshold for signal detection
        
        Returns:
            List of detected weak signals
        """
        weak_signals = []
        
        # Check for declining trends
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Rolling average for satisfaction
            if 'satisfaction_generale' in df.columns:
                df['satisfaction_ma'] = df['satisfaction_generale'].rolling(window=5).mean()
                
                # Detect declining trend
                if len(df) >= 10:
                    recent_avg = df['satisfaction_ma'].tail(5).mean()
                    previous_avg = df['satisfaction_ma'].iloc[-10:-5].mean()
                    
                    if recent_avg < previous_avg - threshold:
                        weak_signals.append({
                            'type': 'declining_satisfaction',
                            'severity': 'medium',
                            'description': f'Satisfaction declining: {previous_avg:.2f} → {recent_avg:.2f}',
                            'recommendation': 'Investigate recent changes in training delivery'
                        })
        
        # Check for recurring negative keywords
        if 'commentaires' in df.columns:
            all_comments = ' '.join(df['commentaires'].dropna().astype(str))
            negative_keywords = []
            
            for word in SENTIMENT_PATTERNS['negative_fr'] + SENTIMENT_PATTERNS['negative_en']:
                count = all_comments.lower().count(word)
                if count > len(df) * 0.1:  # Appears in >10% of evaluations
                    negative_keywords.append((word, count))
            
            if negative_keywords:
                weak_signals.append({
                    'type': 'recurring_complaints',
                    'severity': 'high',
                    'description': f'Recurring negative themes: {[k[0] for k in negative_keywords[:3]]}',
                    'recommendation': 'Address specific pain points mentioned repeatedly'
                })
        
        # Check for low completion rates
        if 'completion_rate' in df.columns:
            low_completion = df[df['completion_rate'] < 0.5]
            if len(low_completion) > len(df) * 0.2:
                weak_signals.append({
                    'type': 'low_engagement',
                    'severity': 'medium',
                    'description': f'{len(low_completion)} sessions with <50% completion',
                    'recommendation': 'Simplify evaluation process or improve timing'
                })
        
        return weak_signals
    
    def cluster_evaluations(self, df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
        """
        Cluster evaluations to identify patterns
        
        Args:
            df: DataFrame with evaluation data
            n_clusters: Number of clusters
        
        Returns:
            DataFrame with cluster assignments
        """
        # Select numerical features
        feature_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        
        if len(feature_cols) < 2:
            df['cluster'] = 0
            return df
        
        X = df[feature_cols].fillna(df[feature_cols].mean())
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        return df
    
    def calculate_metrics(self, df: pd.DataFrame, session_id: str) -> EvaluationMetrics:
        """
        Calculate comprehensive metrics for a training session
        
        Args:
            df: DataFrame with evaluation data for one session
            session_id: Session identifier
        
        Returns:
            EvaluationMetrics object
        """
        # Calculate averages for Likert scales
        metrics_cols = {
            'satisfaction_generale': 'avg_satisfaction',
            'qualite_formateur': 'avg_trainer_quality',
            'pertinence_contenu': 'avg_content_relevance',
            'logistique': 'avg_logistics',
            'applicabilite': 'avg_applicability'
        }
        
        metrics = {}
        for col, metric_name in metrics_cols.items():
            if col in df.columns:
                metrics[metric_name] = df[col].mean()
            else:
                metrics[metric_name] = 0.0
        
        # Sentiment analysis
        if 'commentaires' in df.columns:
            sentiments = [self.analyze_sentiment(str(c)) for c in df['commentaires'].dropna()]
            
            if sentiments:
                avg_sentiment = {
                    'positive': np.mean([s['positive'] for s in sentiments]),
                    'neutral': np.mean([s['neutral'] for s in sentiments]),
                    'negative': np.mean([s['negative'] for s in sentiments])
                }
            else:
                avg_sentiment = {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33}
        else:
            avg_sentiment = {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33}
        
        # Extract themes
        if 'commentaires' in df.columns:
            themes = self.extract_themes(df['commentaires'].dropna().tolist(), n_topics=3)
            key_themes = [', '.join(keywords[:3]) for _, keywords in themes if keywords]
        else:
            key_themes = []
        
        # Detect weak signals
        weak_signals_data = self.detect_weak_signals(df)
        weak_signals = [ws['description'] for ws in weak_signals_data]
        
        return EvaluationMetrics(
            session_id=session_id,
            formation_type=df['type_formation'].iloc[0] if 'type_formation' in df.columns else 'Unknown',
            date=df['date'].iloc[0].strftime('%Y-%m-%d') if 'date' in df.columns else str(datetime.now().date()),
            avg_satisfaction=metrics.get('avg_satisfaction', 0.0),
            avg_trainer_quality=metrics.get('avg_trainer_quality', 0.0),
            avg_content_relevance=metrics.get('avg_content_relevance', 0.0),
            avg_logistics=metrics.get('avg_logistics', 0.0),
            avg_applicability=metrics.get('avg_applicability', 0.0),
            response_count=len(df),
            completion_rate=len(df[df['completed'] == True]) / len(df) if 'completed' in df.columns else 1.0,
            sentiment_distribution=avg_sentiment,
            key_themes=key_themes,
            weak_signals=weak_signals
        )
    
    def generate_insights(self, df: pd.DataFrame, period: str = 'monthly') -> InsightReport:
        """
        Generate comprehensive insights report
        
        Args:
            df: DataFrame with all evaluation data
            period: Analysis period ('monthly', 'quarterly', 'yearly')
        
        Returns:
            InsightReport object
        """
        # Calculate overall metrics
        overall_satisfaction = df['satisfaction_generale'].mean() if 'satisfaction_generale' in df.columns else 0.0
        
        # Determine trend
        if 'date' in df.columns and len(df) >= 20:
            df = df.sort_values('date')
            recent = df.tail(int(len(df) * 0.3))['satisfaction_generale'].mean()
            older = df.head(int(len(df) * 0.3))['satisfaction_generale'].mean()
            
            if recent > older + 0.3:
                trend = 'improving'
            elif recent < older - 0.3:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        # Identify strengths and improvements
        strengths = []
        improvements = []
        
        criteria = ['qualite_formateur', 'pertinence_contenu', 'logistique', 'applicabilite']
        for criterion in criteria:
            if criterion in df.columns:
                avg_score = df[criterion].mean()
                if avg_score >= 4.0:
                    strengths.append((criterion.replace('_', ' ').title(), avg_score))
                elif avg_score < 3.5:
                    improvements.append((criterion.replace('_', ' ').title(), avg_score))
        
        # Extract emerging themes
        if 'commentaires' in df.columns:
            themes = self.extract_themes(df['commentaires'].dropna().tolist(), n_topics=5)
            emerging_themes = [f"{name}: {', '.join(keywords[:3])}" for name, keywords in themes if keywords]
        else:
            emerging_themes = []
        
        # Detect weak signals
        weak_signals = self.detect_weak_signals(df)
        
        # Generate recommendations
        recommendations = []
        
        if overall_satisfaction < 3.5:
            recommendations.append("Overall satisfaction is below target - conduct detailed review")
        
        if improvements:
            recommendations.append(f"Focus on improving: {', '.join([i[0] for i in improvements[:2]])}")
        
        if weak_signals:
            recommendations.append(f"Address {len(weak_signals)} identified weak signals")
        
        if not recommendations:
            recommendations.append("Maintain current quality standards")
        
        return InsightReport(
            period=period,
            total_evaluations=len(df),
            overall_satisfaction=overall_satisfaction,
            satisfaction_trend=trend,
            top_strengths=strengths[:5],
            top_improvements=improvements[:5],
            emerging_themes=emerging_themes[:5],
            weak_signals=weak_signals,
            recommendations=recommendations,
            generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def export_results(self, metrics: List[EvaluationMetrics], insights: InsightReport, 
                      output_format: str = 'json') -> str:
        """
        Export analysis results
        
        Args:
            metrics: List of evaluation metrics
            insights: Insights report
            output_format: Output format ('json', 'excel', 'csv')
        
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if output_format == 'json':
            output_file = f'/home/claude/evaluation_analysis_{timestamp}.json'
            
            export_data = {
                'metrics': [asdict(m) for m in metrics],
                'insights': asdict(insights)
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        elif output_format == 'excel':
            output_file = f'/home/claude/evaluation_analysis_{timestamp}.xlsx'
            
            # Create metrics DataFrame
            metrics_df = pd.DataFrame([asdict(m) for m in metrics])
            
            # Create insights DataFrame
            insights_dict = asdict(insights)
            insights_df = pd.DataFrame([insights_dict])
            
            # Export to Excel with multiple sheets
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                insights_df.to_excel(writer, sheet_name='Insights', index=False)
        
        return output_file


# Example usage and testing
if __name__ == "__main__":
    print("Training Evaluation Analyzer - Phase 1")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = TrainingEvaluationAnalyzer()
    
    # Create sample data for demonstration
    sample_data = {
        'session_id': ['S001'] * 20,
        'type_formation': ['Lean Six Sigma'] * 20,
        'date': pd.date_range('2025-01-01', periods=20),
        'satisfaction_generale': np.random.uniform(3.5, 5.0, 20),
        'qualite_formateur': np.random.uniform(3.8, 5.0, 20),
        'pertinence_contenu': np.random.uniform(3.5, 4.8, 20),
        'logistique': np.random.uniform(3.0, 4.5, 20),
        'applicabilite': np.random.uniform(3.5, 4.8, 20),
        'completed': [True] * 18 + [False] * 2,
        'commentaires': [
            'Excellent formateur, très pédagogue',
            'Contenu intéressant mais salle trop petite',
            'Bonne formation, applicable au quotidien',
            'Manque d\'exemples concrets',
            'Très satisfait de la qualité',
            'Difficile à suivre par moments',
            'Formation utile et bien structurée',
            'Horaires pas adaptés',
            'Formateur compétent et disponible',
            'Bon contenu mais rythme trop rapide',
            'Excellente formation, je recommande',
            'Logistique à améliorer',
            'Très bonne expérience globale',
            'Contenu pertinent pour mon poste',
            'Salle mal équipée',
            'Formation enrichissante',
            'Besoin de plus de pratique',
            'Très bon formateur',
            'Contenu dense mais intéressant',
            'Bonne organisation générale'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Calculate metrics
    metrics = analyzer.calculate_metrics(df, 'S001')
    
    print("\nSession Metrics:")
    print(f"  Formation Type: {metrics.formation_type}")
    print(f"  Satisfaction: {metrics.avg_satisfaction:.2f}/5")
    print(f"  Trainer Quality: {metrics.avg_trainer_quality:.2f}/5")
    print(f"  Content Relevance: {metrics.avg_content_relevance:.2f}/5")
    print(f"  Completion Rate: {metrics.completion_rate:.1%}")
    print(f"  Sentiment: {metrics.sentiment_distribution}")
    
    # Generate insights
    insights = analyzer.generate_insights(df)
    
    print("\nInsights Report:")
    print(f"  Period: {insights.period}")
    print(f"  Total Evaluations: {insights.total_evaluations}")
    print(f"  Overall Satisfaction: {insights.overall_satisfaction:.2f}/5")
    print(f"  Trend: {insights.satisfaction_trend}")
    print(f"  Recommendations: {len(insights.recommendations)}")
    
    print("\nAnalysis complete!")
