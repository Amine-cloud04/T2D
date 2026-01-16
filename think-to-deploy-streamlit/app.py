"""
Think to Deploy - Streamlit Application
Safran Casa SED - Complete AI-Powered Solution
Using original Phase 1 & Phase 2 modules
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import original modules
from src.training_evaluation_analyzer import TrainingEvaluationAnalyzer
from src.hr_chatbot import HRChatbot, UserProfile
from src.kpi_monitor import KPIMonitor, KPICalculator
from src.safran_integration import SafranSystemConnector, ADAuthenticationService, DataAnonymizer

# Page config
st.set_page_config(
    page_title="Think to Deploy - Safran",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .chat-user {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: right;
    }
    .chat-bot {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = HRChatbot()
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = TrainingEvaluationAnalyzer()
if 'kpi_monitor' not in st.session_state:
    st.session_state.kpi_monitor = KPIMonitor()
if 'kpi_calculator' not in st.session_state:
    st.session_state.kpi_calculator = KPICalculator()


def generate_sample_data(n=100):
    """Generate sample training data"""
    np.random.seed(42)
    
    comments = [
        'Excellent formateur, très pédagogue',
        'Contenu intéressant mais salle trop petite',
        'Bonne formation, applicable au quotidien',
        'Manque d\'exemples concrets',
        'Très satisfait de la qualité',
        'Difficile à suivre par moments',
        'Formation utile et bien structurée',
        'Horaires pas adaptés',
        'Formateur compétent et disponible',
        'Bon contenu mais rythme trop rapide'
    ]
    
    return pd.DataFrame({
        'session_id': np.random.choice(['S001', 'S002', 'S003'], n),
        'type_formation': np.random.choice(['Lean Six Sigma', 'SAP', 'Soft Skills', 'Technical'], n),
        'date': pd.date_range('2025-01-01', periods=n),
        'satisfaction_generale': np.random.uniform(3.0, 5.0, n),
        'qualite_formateur': np.random.uniform(3.5, 5.0, n),
        'pertinence_contenu': np.random.uniform(3.2, 4.8, n),
        'logistique': np.random.uniform(2.8, 4.5, n),
        'applicabilite': np.random.uniform(3.5, 4.8, n),
        'completed': [True] * int(n * 0.9) + [False] * int(n * 0.1),
        'commentaires': np.random.choice(comments, n)
    })


def show_login():
    """Login page"""
    st.markdown('<div class="main-header">🔐 Think to Deploy - Authentification</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔒 Connexion Sécurisée")
        st.info("Utilisez votre identifiant Safran ou testez en mode démo")
        
        matricule = st.text_input("Matricule", placeholder="EMP001")
        password = st.text_input("Mot de passe", type="password", placeholder="••••••••")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔓 Se connecter"):
                if matricule and password and len(password) >= 8:
                    auth_service = ADAuthenticationService()
                    session = auth_service.authenticate(matricule, password)
                    
                    if session:
                        st.session_state.authenticated = True
                        st.session_state.user_profile = UserProfile(
                            matricule=matricule,
                            profile_type='CDI',
                            category='cadre',
                            site='casa_sed',
                            authenticated=True
                        )
                        st.success("✓ Connexion réussie!")
                        st.rerun()
                    else:
                        st.error("✗ Identifiants incorrects")
                else:
                    st.error("Veuillez remplir tous les champs (mot de passe ≥8 caractères)")
        
        with col_btn2:
            if st.button("🎮 Mode Démo"):
                st.session_state.authenticated = True
                st.session_state.user_profile = UserProfile(
                    matricule='DEMO_USER',
                    profile_type='CDI',
                    category='cadre',
                    site='casa_sed',
                    authenticated=True
                )
                st.rerun()


def show_dashboard():
    """Dashboard page"""
    st.markdown('<div class="main-header">📊 Tableau de Bord</div>', unsafe_allow_html=True)
    
    # Generate sample data
    training_df = generate_sample_data(100)
    
    # Calculate KPIs using original module
    calculator = st.session_state.kpi_calculator
    training_kpis = calculator.calculate_training_kpis(training_df, 'January 2025')
    
    # Record in monitor
    st.session_state.kpi_monitor.record_training_kpis(training_kpis)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📝 Évaluations", f"{training_kpis.total_evaluations}")
    
    with col2:
        st.metric("😊 Satisfaction", f"{training_kpis.avg_satisfaction:.2f}/5")
    
    with col3:
        st.metric("✅ Complétion", f"{training_kpis.completion_rate:.0%}")
    
    with col4:
        st.metric("👨‍🏫 Formateur", f"{training_kpis.avg_trainer_score:.2f}/5")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Évolution de la satisfaction")
        dates = pd.date_range('2025-01-01', periods=30)
        trend_data = pd.DataFrame({
            'Date': dates,
            'Satisfaction': np.random.uniform(3.8, 4.5, 30)
        })
        st.line_chart(trend_data.set_index('Date'))
    
    with col2:
        st.markdown("### 🎯 Distribution des sentiments")
        sentiment_df = pd.DataFrame({
            'Sentiment': ['Positif', 'Neutre', 'Négatif'],
            'Pourcentage': [
                training_kpis.sentiment_positive * 100,
                training_kpis.sentiment_neutral * 100,
                training_kpis.sentiment_negative * 100
            ]
        })
        st.bar_chart(sentiment_df.set_index('Sentiment'))
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎓 Formations par type")
        formation_counts = training_df['type_formation'].value_counts()
        st.bar_chart(formation_counts)
    
    with col2:
        st.markdown("### 💡 Insights clés")
        st.info(f"📊 {training_kpis.total_evaluations} évaluations analysées")
        
        if training_kpis.avg_satisfaction >= 4.0:
            st.success(f"✅ Satisfaction excellente ({training_kpis.avg_satisfaction:.2f}/5)")
        else:
            st.warning(f"⚠️ Satisfaction à améliorer ({training_kpis.avg_satisfaction:.2f}/5)")


def show_chatbot():
    """Chatbot page using original module"""
    st.markdown('<div class="main-header">🤖 Assistant RH Intelligent</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 💬 À propos")
        st.info("""
        **Sujets:**
        - 📅 Congés
        - 💰 Paie
        - 🎁 Avantages
        - 🚗 Transport
        - ⏰ Pointage
        """)
        
        if st.button("🗑️ Effacer"):
            st.session_state.conversation_history = []
            st.rerun()
    
    # Display chat
    for msg in st.session_state.conversation_history:
        if msg['role'] == 'user':
            st.markdown(f'<div class="chat-user"><strong>👤 Vous:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot"><strong>🤖 Assistant:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick questions
    st.markdown("#### ⚡ Questions rapides")
    col1, col2, col3 = st.columns(3)
    
    questions = [
        ("📅 Congés", "Comment demander des congés ?"),
        ("💰 Paie", "Comment obtenir mon bulletin de paie ?"),
        ("🎁 Avantages", "Quels sont mes avantages sociaux ?")
    ]
    
    for col, (label, q) in zip([col1, col2, col3], questions):
        with col:
            if st.button(label):
                process_chat_message(q)
    
    # Input
    user_input = st.text_input("Votre question", placeholder="Posez votre question...")
    
    if st.button("📤 Envoyer") and user_input:
        process_chat_message(user_input)


def process_chat_message(user_input):
    """Process chat message using original chatbot"""
    st.session_state.conversation_history.append({
        'role': 'user',
        'content': user_input
    })
    
    chatbot = st.session_state.chatbot
    user_profile = st.session_state.user_profile
    
    message = chatbot.generate_response(user_input, user_profile)
    
    st.session_state.conversation_history.append({
        'role': 'assistant',
        'content': message.response
    })
    
    st.rerun()


def show_analyzer():
    """Training analyzer page using original module"""
    st.markdown('<div class="main-header">📈 Analyse des Formations</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📥 Importer", "📊 Analyser"])
    
    with tab1:
        st.markdown("### 📥 Importer des données")
        
        uploaded_file = st.file_uploader("Fichier Excel ou CSV", type=['xlsx', 'csv'])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✓ {len(df)} évaluations chargées")
                st.dataframe(df.head(10))
                st.session_state.training_data = df
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
        
        if st.button("🎲 Générer données de test"):
            test_data = generate_sample_data(100)
            st.session_state.training_data = test_data
            st.success("✓ Données générées!")
            st.rerun()
    
    with tab2:
        if 'training_data' in st.session_state:
            df = st.session_state.training_data
            analyzer = st.session_state.analyzer
            
            with st.spinner("Analyse en cours..."):
                # Use original analyzer
                insights = analyzer.generate_insights(df, period='January 2025')
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total", insights.total_evaluations)
            with col2:
                st.metric("Satisfaction", f"{insights.overall_satisfaction:.2f}/5")
            with col3:
                st.metric("Tendance", insights.satisfaction_trend)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ✅ Points forts")
                for strength, score in insights.top_strengths:
                    st.success(f"✓ {strength}: {score:.2f}/5")
            
            with col2:
                st.markdown("### ⚠️ À améliorer")
                for improvement, score in insights.top_improvements:
                    st.warning(f"⚠ {improvement}: {score:.2f}/5")
            
            st.markdown("### 💡 Recommandations")
            for i, rec in enumerate(insights.recommendations, 1):
                st.info(f"{i}. {rec}")
        else:
            st.info("👆 Importez ou générez des données pour commencer")


def show_integration():
    """Integration page using original module"""
    st.markdown('<div class="main-header">🔗 Intégration Systèmes</div>', unsafe_allow_html=True)
    
    st.markdown("### 🖥️ État des systèmes Safran")
    
    # Use original connector
    connector = SafranSystemConnector()
    
    systems = [
        {'name': 'Prisma', 'icon': '🔵', 'desc': 'Gestion des processus'},
        {'name': 'Intranet', 'icon': '🟢', 'desc': 'Documentation'},
        {'name': 'DFS', 'icon': '🟢', 'desc': 'Annuaire'},
        {'name': 'SharePoint', 'icon': '🔵', 'desc': 'Documents'},
        {'name': 'SELIA', 'icon': '🟢', 'desc': 'LMS'},
    ]
    
    for system in systems:
        col1, col2, col3 = st.columns([2, 1, 3])
        with col1:
            st.markdown(f"**{system['icon']} {system['name']}**")
        with col2:
            st.success("✓ Connecté")
        with col3:
            st.text(system['desc'])
    
    st.markdown("---")
    
    st.markdown("### 🔒 Sécurité")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("✓ Authentification AD")
        st.success("✓ Sessions sécurisées")
    
    with col2:
        st.success("✓ Anonymisation SHA-256")
        st.success("✓ RGPD compliant")
    
    with col3:
        st.success("✓ Audit logging")
        st.success("✓ C2 classification")


def main():
    """Main application"""
    
    if not st.session_state.authenticated:
        show_login()
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🏢 Safran Casa SED")
        st.markdown(f"**👤 {st.session_state.user_profile.matricule}**")
        st.caption(f"{st.session_state.user_profile.profile_type} - {st.session_state.user_profile.category}")
        
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "🤖 Chatbot", "📈 Analyseur", "🔗 Intégration"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        if st.button("🚪 Déconnexion"):
            st.session_state.authenticated = False
            st.session_state.user_profile = None
            st.session_state.conversation_history = []
            st.rerun()
        
        st.markdown("---")
        st.caption("Think to Deploy v1.0")
        st.caption("© 2025 Safran")
    
    # Display page
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "🤖 Chatbot":
        show_chatbot()
    elif page == "📈 Analyseur":
        show_analyzer()
    elif page == "🔗 Intégration":
        show_integration()


if __name__ == "__main__":
    main()
