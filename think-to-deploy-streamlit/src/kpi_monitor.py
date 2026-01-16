"""
KPI Monitoring Dashboard - Phase 1
Real-time monitoring and visualization of KPIs for both Training Evaluation and HR Chatbot
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from dataclasses import dataclass, asdict

@dataclass
class TrainingKPIs:
    """KPIs for Training Evaluation Analysis"""
    period: str
    completion_rate: float
    avg_satisfaction: float
    avg_trainer_score: float
    avg_logistics_score: float
    sentiment_positive: float
    sentiment_neutral: float
    sentiment_negative: float
    weak_signals_count: int
    processing_time_avg: float
    total_evaluations: int
    timestamp: str

@dataclass
class ChatbotKPIs:
    """KPIs for HR Chatbot"""
    period: str
    intent_accuracy: float
    correct_response_rate: float
    escalation_rate: float
    avg_response_time: float
    active_users: int
    total_interactions: int
    csat_score: float
    top_intents: List[str]
    timestamp: str

@dataclass
class AlertConfig:
    """Configuration for alerts"""
    metric_name: str
    threshold: float
    condition: str  # 'below', 'above'
    severity: str  # 'info', 'warning', 'critical'
    notification_email: str


class KPICalculator:
    """Calculate KPIs from raw data"""
    
    @staticmethod
    def calculate_training_kpis(evaluations_df: pd.DataFrame, period: str) -> TrainingKPIs:
        """
        Calculate training evaluation KPIs
        
        Args:
            evaluations_df: DataFrame with evaluation data
            period: Time period (e.g., 'January 2025', 'Q1 2025')
        
        Returns:
            TrainingKPIs object
        """
        total_evals = len(evaluations_df)
        
        # Completion rate
        completion_rate = 0.0
        if 'completed' in evaluations_df.columns:
            completion_rate = evaluations_df['completed'].sum() / total_evals if total_evals > 0 else 0.0
        
        # Average scores
        score_columns = {
            'satisfaction_generale': 'avg_satisfaction',
            'qualite_formateur': 'avg_trainer_score',
            'logistique': 'avg_logistics_score'
        }
        
        scores = {}
        for col, key in score_columns.items():
            if col in evaluations_df.columns:
                scores[key] = evaluations_df[col].mean()
            else:
                scores[key] = 0.0
        
        # Sentiment distribution
        sentiment_positive = sentiment_neutral = sentiment_negative = 0.33
        if 'sentiment' in evaluations_df.columns:
            sentiment_counts = evaluations_df['sentiment'].value_counts(normalize=True)
            sentiment_positive = sentiment_counts.get('positive', 0.33)
            sentiment_neutral = sentiment_counts.get('neutral', 0.33)
            sentiment_negative = sentiment_counts.get('negative', 0.33)
        
        # Weak signals
        weak_signals_count = 0
        if 'weak_signals' in evaluations_df.columns:
            weak_signals_count = evaluations_df['weak_signals'].notna().sum()
        
        # Processing time (simulated)
        processing_time_avg = np.random.uniform(2.0, 5.0)
        
        return TrainingKPIs(
            period=period,
            completion_rate=completion_rate,
            avg_satisfaction=scores['avg_satisfaction'],
            avg_trainer_score=scores['avg_trainer_score'],
            avg_logistics_score=scores['avg_logistics_score'],
            sentiment_positive=sentiment_positive,
            sentiment_neutral=sentiment_neutral,
            sentiment_negative=sentiment_negative,
            weak_signals_count=weak_signals_count,
            processing_time_avg=processing_time_avg,
            total_evaluations=total_evals,
            timestamp=datetime.now().isoformat()
        )
    
    @staticmethod
    def calculate_chatbot_kpis(conversations_df: pd.DataFrame, period: str) -> ChatbotKPIs:
        """
        Calculate chatbot KPIs
        
        Args:
            conversations_df: DataFrame with conversation data
            period: Time period
        
        Returns:
            ChatbotKPIs object
        """
        total_interactions = len(conversations_df)
        
        # Intent accuracy
        intent_accuracy = 0.0
        if 'confidence' in conversations_df.columns:
            intent_accuracy = conversations_df['confidence'].mean()
        
        # Correct response rate (based on confidence threshold)
        correct_response_rate = 0.0
        if 'confidence' in conversations_df.columns:
            correct_response_rate = (conversations_df['confidence'] >= 0.7).sum() / total_interactions if total_interactions > 0 else 0.0
        
        # Escalation rate
        escalation_rate = 0.0
        if 'escalated' in conversations_df.columns:
            escalation_rate = conversations_df['escalated'].sum() / total_interactions if total_interactions > 0 else 0.0
        
        # Average response time (simulated - in production, measure actual time)
        avg_response_time = np.random.uniform(0.5, 2.0)
        
        # Active users
        active_users = 0
        if 'user_id' in conversations_df.columns:
            active_users = conversations_df['user_id'].nunique()
        
        # CSAT score (simulated - in production, from user feedback)
        csat_score = np.random.uniform(3.8, 4.5)
        
        # Top intents
        top_intents = []
        if 'intent' in conversations_df.columns:
            top_intents = conversations_df['intent'].value_counts().head(5).index.tolist()
        
        return ChatbotKPIs(
            period=period,
            intent_accuracy=intent_accuracy,
            correct_response_rate=correct_response_rate,
            escalation_rate=escalation_rate,
            avg_response_time=avg_response_time,
            active_users=active_users,
            total_interactions=total_interactions,
            csat_score=csat_score,
            top_intents=top_intents,
            timestamp=datetime.now().isoformat()
        )


class KPIMonitor:
    """
    KPI Monitoring system with alerting and historical tracking
    """
    
    def __init__(self):
        self.training_history = []
        self.chatbot_history = []
        self.alerts = []
        self.alert_configs = self._initialize_alert_configs()
    
    def _initialize_alert_configs(self) -> List[AlertConfig]:
        """Initialize alert configurations"""
        return [
            AlertConfig(
                metric_name='completion_rate',
                threshold=0.7,
                condition='below',
                severity='warning',
                notification_email='rh.casa@safran.com'
            ),
            AlertConfig(
                metric_name='avg_satisfaction',
                threshold=3.5,
                condition='below',
                severity='critical',
                notification_email='rh.casa@safran.com'
            ),
            AlertConfig(
                metric_name='sentiment_negative',
                threshold=0.4,
                condition='above',
                severity='warning',
                notification_email='rh.casa@safran.com'
            ),
            AlertConfig(
                metric_name='intent_accuracy',
                threshold=0.7,
                condition='below',
                severity='warning',
                notification_email='rh.casa@safran.com'
            ),
            AlertConfig(
                metric_name='escalation_rate',
                threshold=0.3,
                condition='above',
                severity='warning',
                notification_email='rh.casa@safran.com'
            )
        ]
    
    def record_training_kpis(self, kpis: TrainingKPIs):
        """Record training KPIs"""
        self.training_history.append(kpis)
        self._check_alerts(kpis, 'training')
    
    def record_chatbot_kpis(self, kpis: ChatbotKPIs):
        """Record chatbot KPIs"""
        self.chatbot_history.append(kpis)
        self._check_alerts(kpis, 'chatbot')
    
    def _check_alerts(self, kpis, kpi_type: str):
        """Check if any alert thresholds are breached"""
        kpis_dict = asdict(kpis)
        
        for config in self.alert_configs:
            if config.metric_name in kpis_dict:
                value = kpis_dict[config.metric_name]
                
                triggered = False
                if config.condition == 'below' and value < config.threshold:
                    triggered = True
                elif config.condition == 'above' and value > config.threshold:
                    triggered = True
                
                if triggered:
                    alert = {
                        'timestamp': datetime.now().isoformat(),
                        'type': kpi_type,
                        'metric': config.metric_name,
                        'value': value,
                        'threshold': config.threshold,
                        'severity': config.severity,
                        'message': f"{config.metric_name} is {config.condition} threshold: {value:.2f} vs {config.threshold:.2f}"
                    }
                    self.alerts.append(alert)
                    
                    # In production: send email notification
                    print(f"⚠️  ALERT [{config.severity.upper()}]: {alert['message']}")
    
    def get_training_trends(self, last_n: int = 10) -> pd.DataFrame:
        """Get training KPI trends"""
        if not self.training_history:
            return pd.DataFrame()
        
        recent_kpis = self.training_history[-last_n:]
        df = pd.DataFrame([asdict(k) for k in recent_kpis])
        
        return df
    
    def get_chatbot_trends(self, last_n: int = 10) -> pd.DataFrame:
        """Get chatbot KPI trends"""
        if not self.chatbot_history:
            return pd.DataFrame()
        
        recent_kpis = self.chatbot_history[-last_n:]
        df = pd.DataFrame([asdict(k) for k in recent_kpis])
        
        return df
    
    def generate_dashboard_data(self) -> Dict:
        """
        Generate data for dashboard visualization
        
        Returns:
            Dictionary with dashboard data
        """
        dashboard_data = {
            'training': {
                'current': asdict(self.training_history[-1]) if self.training_history else None,
                'trends': self.get_training_trends().to_dict('records'),
                'alerts': [a for a in self.alerts if a['type'] == 'training']
            },
            'chatbot': {
                'current': asdict(self.chatbot_history[-1]) if self.chatbot_history else None,
                'trends': self.get_chatbot_trends().to_dict('records'),
                'alerts': [a for a in self.alerts if a['type'] == 'chatbot']
            },
            'summary': {
                'total_training_evaluations': sum(k.total_evaluations for k in self.training_history),
                'total_chatbot_interactions': sum(k.total_interactions for k in self.chatbot_history),
                'active_alerts': len([a for a in self.alerts if a['severity'] in ['warning', 'critical']]),
                'last_updated': datetime.now().isoformat()
            }
        }
        
        return dashboard_data
    
    def export_dashboard(self, output_file: str = '/home/claude/dashboard_data.json'):
        """Export dashboard data to JSON"""
        dashboard_data = self.generate_dashboard_data()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
        
        return output_file
    
    def export_to_excel(self, output_file: str = '/home/claude/kpi_report.xlsx'):
        """Export KPIs to Excel for Power BI integration"""
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Training KPIs
            if self.training_history:
                training_df = pd.DataFrame([asdict(k) for k in self.training_history])
                training_df.to_excel(writer, sheet_name='Training_KPIs', index=False)
            
            # Chatbot KPIs
            if self.chatbot_history:
                chatbot_df = pd.DataFrame([asdict(k) for k in self.chatbot_history])
                chatbot_df.to_excel(writer, sheet_name='Chatbot_KPIs', index=False)
            
            # Alerts
            if self.alerts:
                alerts_df = pd.DataFrame(self.alerts)
                alerts_df.to_excel(writer, sheet_name='Alerts', index=False)
        
        return output_file
    
    def print_summary(self):
        """Print KPI summary to console"""
        print("\n" + "=" * 70)
        print("KPI MONITORING DASHBOARD")
        print("=" * 70)
        
        # Training Evaluation KPIs
        if self.training_history:
            latest_training = self.training_history[-1]
            print(f"\n📊 TRAINING EVALUATION KPIs ({latest_training.period})")
            print("-" * 70)
            print(f"  Total Evaluations:    {latest_training.total_evaluations}")
            print(f"  Completion Rate:      {latest_training.completion_rate:.1%}")
            print(f"  Avg Satisfaction:     {latest_training.avg_satisfaction:.2f}/5.00")
            print(f"  Trainer Score:        {latest_training.avg_trainer_score:.2f}/5.00")
            print(f"  Logistics Score:      {latest_training.avg_logistics_score:.2f}/5.00")
            print(f"  Sentiment:")
            print(f"    • Positive:         {latest_training.sentiment_positive:.1%}")
            print(f"    • Neutral:          {latest_training.sentiment_neutral:.1%}")
            print(f"    • Negative:         {latest_training.sentiment_negative:.1%}")
            print(f"  Weak Signals:         {latest_training.weak_signals_count}")
            print(f"  Processing Time:      {latest_training.processing_time_avg:.2f}s")
        
        # Chatbot KPIs
        if self.chatbot_history:
            latest_chatbot = self.chatbot_history[-1]
            print(f"\n🤖 HR CHATBOT KPIs ({latest_chatbot.period})")
            print("-" * 70)
            print(f"  Total Interactions:   {latest_chatbot.total_interactions}")
            print(f"  Active Users:         {latest_chatbot.active_users}")
            print(f"  Intent Accuracy:      {latest_chatbot.intent_accuracy:.1%}")
            print(f"  Correct Responses:    {latest_chatbot.correct_response_rate:.1%}")
            print(f"  Escalation Rate:      {latest_chatbot.escalation_rate:.1%}")
            print(f"  Avg Response Time:    {latest_chatbot.avg_response_time:.2f}s")
            print(f"  CSAT Score:           {latest_chatbot.csat_score:.2f}/5.00")
            if latest_chatbot.top_intents:
                print(f"  Top Intents:          {', '.join(latest_chatbot.top_intents[:3])}")
        
        # Alerts
        active_alerts = [a for a in self.alerts if a['severity'] in ['warning', 'critical']]
        if active_alerts:
            print(f"\n⚠️  ACTIVE ALERTS ({len(active_alerts)})")
            print("-" * 70)
            for alert in active_alerts[-5:]:  # Show last 5 alerts
                print(f"  [{alert['severity'].upper()}] {alert['message']}")
        
        print("\n" + "=" * 70)
        print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70 + "\n")


# Example usage
if __name__ == "__main__":
    print("KPI Monitoring System - Phase 1")
    print("=" * 70)
    
    # Initialize monitor
    monitor = KPIMonitor()
    calculator = KPICalculator()
    
    # Simulate training evaluation data
    training_data = pd.DataFrame({
        'session_id': ['S001'] * 50,
        'type_formation': ['Lean Six Sigma'] * 50,
        'date': pd.date_range('2025-01-01', periods=50),
        'satisfaction_generale': np.random.uniform(3.0, 5.0, 50),
        'qualite_formateur': np.random.uniform(3.5, 5.0, 50),
        'logistique': np.random.uniform(2.8, 4.5, 50),
        'completed': [True] * 45 + [False] * 5,
        'sentiment': np.random.choice(['positive', 'neutral', 'negative'], 50, p=[0.6, 0.3, 0.1])
    })
    
    # Calculate and record training KPIs
    training_kpis = calculator.calculate_training_kpis(training_data, 'January 2025')
    monitor.record_training_kpis(training_kpis)
    
    # Simulate chatbot conversation data
    chatbot_data = pd.DataFrame({
        'user_id': [f'USER{i:03d}' for i in range(1, 101)],
        'intent': np.random.choice(['conges_demande', 'paie_bulletin', 'avantages_liste', 
                                   'transport_remboursement', 'pointage_procedure'], 100),
        'confidence': np.random.uniform(0.5, 0.95, 100),
        'escalated': np.random.choice([True, False], 100, p=[0.15, 0.85])
    })
    
    # Calculate and record chatbot KPIs
    chatbot_kpis = calculator.calculate_chatbot_kpis(chatbot_data, 'January 2025')
    monitor.record_chatbot_kpis(chatbot_kpis)
    
    # Print dashboard summary
    monitor.print_summary()
    
    # Export data
    json_file = monitor.export_dashboard()
    excel_file = monitor.export_to_excel()
    
    print(f"✓ Dashboard data exported:")
    print(f"  • JSON: {json_file}")
    print(f"  • Excel: {excel_file}")
    
    print("\nKPI monitoring test complete!")
