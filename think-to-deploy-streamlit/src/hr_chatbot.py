"""
HR Chatbot - Phase 1
Intelligent chatbot for HR information with RAG, profile-based responses, and multi-language support
Supports: Congés, Avantages sociaux, Transport, Pointage, Paie
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib

@dataclass
class UserProfile:
    """User profile for personalized responses"""
    matricule: str
    profile_type: str  # CDI, CDD, Interim, Stagiaire, Apprenti
    category: str  # Cadre, Non-cadre
    site: str  # casa_sed or other
    authenticated: bool = False

@dataclass
class ChatMessage:
    """Chat message structure"""
    message_id: str
    user_id: str
    content: str
    intent: str
    response: str
    confidence: float
    timestamp: str
    language: str
    sources: List[str]

@dataclass
class ConversationLog:
    """Conversation logging for GDPR compliance"""
    session_id: str
    user_profile: str  # Anonymized
    messages: List[ChatMessage]
    started_at: str
    ended_at: Optional[str]
    escalated: bool = False


class HRKnowledgeBase:
    """
    Knowledge base for HR information
    Stores and retrieves HR policies, procedures, and FAQ
    """
    
    def __init__(self):
        self.knowledge = self._initialize_knowledge()
        self.faq = self._initialize_faq()
        
    def _initialize_knowledge(self) -> Dict:
        """Initialize HR knowledge base"""
        return {
            'conges': {
                'CDI_cadre': {
                    'conges_payes': {
                        'jours': 25,
                        'acquisition': 'Acquisition sur 12 mois de juin à mai',
                        'demande': 'Via Prisma, validation manager puis RH',
                        'delai_prevenance': '15 jours ouvrés minimum',
                        'fractionnement': 'Possibilité de fractionner, maximum 2 périodes',
                        'report': 'Report possible jusquau 31 mai de lannée suivante'
                    },
                    'rtt': {
                        'jours': 10,
                        'conditions': 'Pour cadres au forfait jours',
                        'utilisation': 'Par journée ou demi-journée',
                        'validite': 'Année civile, pas de report'
                    },
                    'conges_exceptionnels': {
                        'mariage': {'jours': 4, 'justificatif': 'Acte de mariage'},
                        'naissance': {'jours': 3, 'justificatif': 'Acte de naissance'},
                        'deces_conjoint': {'jours': 3, 'justificatif': 'Acte de décès'},
                        'demenagement': {'jours': 1, 'justificatif': 'Justificatif de domicile'}
                    }
                },
                'CDI_non_cadre': {
                    'conges_payes': {
                        'jours': 25,
                        'acquisition': 'Acquisition sur 12 mois de juin à mai',
                        'demande': 'Via Prisma, validation manager puis RH',
                        'delai_prevenance': '15 jours ouvrés minimum'
                    },
                    'rtt': None,  # Pas de RTT pour non-cadres
                    'conges_exceptionnels': {
                        'mariage': {'jours': 4, 'justificatif': 'Acte de mariage'},
                        'naissance': {'jours': 3, 'justificatif': 'Acte de naissance'},
                        'deces_conjoint': {'jours': 3, 'justificatif': 'Acte de décès'}
                    }
                },
                'CDD': {
                    'conges_payes': {
                        'jours': 'Prorata temporis basé sur 25 jours/an',
                        'calcul': '2.08 jours par mois travaillé',
                        'demande': 'Via Prisma, validation manager puis RH',
                        'indemnite': 'Indemnité de congés payés à la fin du contrat'
                    }
                },
                'stagiaire': {
                    'conges': {
                        'droit': 'Pas de congés payés légaux',
                        'convention': 'Congés selon convention de stage',
                        'recommandation': 'Consulter convention et maître de stage'
                    }
                }
            },
            'avantages_sociaux': {
                'tous_profils': {
                    'mutuelle': {
                        'nom': 'Mutuelle Safran',
                        'cotisation': 'Part employeur 60%, part salarié 40%',
                        'couverture': 'Hospitalisation, soins courants, optique, dentaire',
                        'adhesion': 'Obligatoire sauf dispense justifiée',
                        'contact': 'service.mutuelle@safran.com'
                    },
                    'prevoyance': {
                        'garanties': 'Décès, invalidité, incapacité',
                        'cotisation': '100% employeur',
                        'beneficiaires': 'Désignation via RH'
                    },
                    'cantine': {
                        'tarif_cdi_cadre': '5.50 EUR par repas',
                        'tarif_cdi_non_cadre': '4.50 EUR par repas',
                        'tarif_stagiaire': '3.50 EUR par repas',
                        'badges': 'Rechargement via kiosques ou RH',
                        'horaires': '12h00-14h00'
                    }
                },
                'CDI': {
                    'participation': {
                        'eligible': True,
                        'versement': 'Annuel, selon résultats entreprise',
                        'deblocage': 'Immédiat ou PEE/PERCO'
                    },
                    'interessement': {
                        'eligible': True,
                        'calcul': 'Selon performance collective',
                        'versement': 'Juin de chaque année'
                    },
                    'ce_cse': {
                        'cheques_vacances': 'Selon quotient familial',
                        'activites': 'Billetterie, événements',
                        'subventions': 'Loisirs, sport, culture'
                    }
                }
            },
            'transport': {
                'casa_sed': {
                    'remboursement_transport': {
                        'taux': '50% du coût abonnement',
                        'types': 'Tramway, bus, navette',
                        'procedure': 'Justificatif mensuel à soumettre via Prisma',
                        'delai': 'Remboursement sur paie du mois suivant'
                    },
                    'parking': {
                        'disponible': True,
                        'acces': 'Badge gratuit, demande via RH',
                        'places': 'Dans la limite des places disponibles'
                    },
                    'covoiturage': {
                        'plateforme': 'Intranet Safran Covoiturage',
                        'prime': '50 EUR/mois si covoiturage régulier (>8 trajets/mois)'
                    }
                }
            },
            'pointage': {
                'CDI_non_cadre': {
                    'obligatoire': True,
                    'methode': 'Badge sur bornes pointeuses',
                    'horaires': 'Pointage entrée/sortie + pauses >20min',
                    'oubli': 'Régularisation via manager et RH sous 48h',
                    'controle': 'Suivi mensuel, validation manager'
                },
                'CDI_cadre': {
                    'obligatoire': False,
                    'regime': 'Forfait jours (218 jours/an)',
                    'suivi': 'Déclaration CRA mensuelle',
                    'amplitude': 'Autonomie avec respect équilibre vie pro/perso'
                },
                'CDD_interim': {
                    'obligatoire': True,
                    'methode': 'Badge sur bornes pointeuses',
                    'particularite': 'Suivi strict pour calcul heures'
                }
            },
            'paie': {
                'tous_profils': {
                    'date_versement': '28 de chaque mois',
                    'bulletins': 'Disponibles sur Prisma rubrique "Mes documents"',
                    'questions': 'Contact paie.casa@safran.com',
                    'acompte': {
                        'possible': True,
                        'conditions': 'Maximum 1 par mois, 50% salaire',
                        'demande': 'Via Prisma, validation N+1, 5 jours avant paie'
                    }
                },
                'elements_fixes': {
                    'salaire_base': 'Selon contrat',
                    'primes_fixes': 'Ancienneté, fonction si applicable'
                },
                'elements_variables': {
                    'heures_sup': 'Pour non-cadres, taux majoré selon convention',
                    'primes_variables': 'Selon performance et accord collectif',
                    'indemnites': 'Transport, repas si applicable'
                }
            }
        }
    
    def _initialize_faq(self) -> List[Dict]:
        """Initialize frequently asked questions"""
        return [
            {
                'question': 'Comment demander des congés ?',
                'keywords': ['congés', 'demande', 'vacances', 'cp'],
                'answer': 'Pour demander des congés, connectez-vous à Prisma, allez dans "Mes demandes" > "Congés". Sélectionnez les dates, le type de congé (CP, RTT si applicable), puis validez. Votre manager recevra une notification. Pensez à faire votre demande au moins 15 jours à l\'avance.',
                'intent': 'conges_demande'
            },
            {
                'question': 'Combien de jours de congés ai-je ?',
                'keywords': ['jours', 'congés', 'solde', 'compteur'],
                'answer': 'Votre solde de congés est visible sur Prisma dans "Mon espace" > "Mes compteurs". Vous y trouverez le détail de vos CP, RTT (si applicable), et congés exceptionnels. Le solde est mis à jour quotidiennement.',
                'intent': 'conges_solde'
            },
            {
                'question': 'Comment obtenir mon bulletin de paie ?',
                'keywords': ['bulletin', 'paie', 'salaire', 'fiche'],
                'answer': 'Votre bulletin de paie est disponible sur Prisma dans "Mes documents" > "Bulletins de paie". Vous pouvez télécharger vos bulletins des 12 derniers mois. Pour les bulletins plus anciens, contactez paie.casa@safran.com',
                'intent': 'paie_bulletin'
            },
            {
                'question': 'Quels sont mes avantages sociaux ?',
                'keywords': ['avantages', 'mutuelle', 'cantine', 'ce'],
                'answer': 'Vous bénéficiez de plusieurs avantages : mutuelle d\'entreprise (60% pris en charge), prévoyance, cantine à tarif préférentiel, remboursement transport (50%), et selon votre profil : participation, intéressement, avantages CE/CSE. Détails disponibles sur l\'intranet section "Avantages sociaux".',
                'intent': 'avantages_liste'
            },
            {
                'question': 'Comment faire pointer ?',
                'keywords': ['pointer', 'pointage', 'badge', 'horaire'],
                'answer': 'Si vous êtes soumis au pointage, utilisez votre badge sur les bornes à l\'entrée/sortie. Pointez à chaque entrée, sortie, et pour les pauses de plus de 20 minutes. En cas d\'oubli, régularisez sous 48h via votre manager et RH.',
                'intent': 'pointage_procedure'
            },
            {
                'question': 'Comment recharger mon badge cantine ?',
                'keywords': ['cantine', 'badge', 'recharge', 'repas'],
                'answer': 'Vous pouvez recharger votre badge cantine via les kiosques situés près de la cantine (paiement CB) ou auprès du service RH (chèque ou espèces). Le rechargement minimum est de 20 EUR.',
                'intent': 'cantine_recharge'
            },
            {
                'question': 'Comment me faire rembourser les transports ?',
                'keywords': ['transport', 'remboursement', 'abonnement', 'tramway'],
                'answer': 'Safran rembourse 50% de votre abonnement transport en commun. Soumettez votre justificatif mensuel via Prisma dans "Mes demandes" > "Frais et remboursements". Le remboursement apparaît sur la paie du mois suivant.',
                'intent': 'transport_remboursement'
            }
        ]
    
    def search_knowledge(self, query: str, user_profile: UserProfile) -> List[Dict]:
        """
        Search knowledge base with profile-based filtering
        
        Args:
            query: User query
            user_profile: User profile for personalization
        
        Returns:
            List of relevant knowledge items
        """
        results = []
        query_lower = query.lower()
        
        # Search in main knowledge base
        for category, content in self.knowledge.items():
            if any(keyword in query_lower for keyword in category.split('_')):
                # Filter by user profile
                profile_key = f"{user_profile.profile_type}_{user_profile.category}" if user_profile.category != 'N/A' else user_profile.profile_type
                
                if profile_key in content:
                    results.append({
                        'category': category,
                        'content': content[profile_key],
                        'source': f'Knowledge Base - {category}',
                        'relevance': 0.9
                    })
                elif 'tous_profils' in content:
                    results.append({
                        'category': category,
                        'content': content['tous_profils'],
                        'source': f'Knowledge Base - {category}',
                        'relevance': 0.8
                    })
        
        # Search in FAQ
        for faq_item in self.faq:
            if any(keyword in query_lower for keyword in faq_item['keywords']):
                results.append({
                    'category': 'FAQ',
                    'content': faq_item,
                    'source': 'FAQ',
                    'relevance': 0.85
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)
        
        return results[:5]  # Return top 5 results


class IntentClassifier:
    """
    Classify user intent from their message
    Supports French, English, and basic Arabic
    """
    
    INTENT_PATTERNS = {
        'conges_demande': [
            r'comment.*demander.*congés?',
            r'demande.*congés?',
            r'poser.*congés?',
            r'prendre.*congés?',
            r'request.*leave',
            r'ask.*leave'
        ],
        'conges_solde': [
            r'combien.*jours.*congés?',
            r'solde.*congés?',
            r'compteur.*congés?',
            r'reste.*congés?',
            r'how many.*days',
            r'leave balance'
        ],
        'paie_bulletin': [
            r'bulletin.*paie',
            r'fiche.*paie',
            r'salaire',
            r'payslip',
            r'salary.*slip'
        ],
        'paie_date': [
            r'quand.*payé',
            r'date.*paie',
            r'versement.*salaire',
            r'when.*paid',
            r'payment.*date'
        ],
        'avantages_liste': [
            r'quels?.*avantages?',
            r'liste.*avantages?',
            r'bénéfices?',
            r'what.*benefits?',
            r'list.*benefits?'
        ],
        'transport_remboursement': [
            r'transport',
            r'remboursement.*transport',
            r'abonnement',
            r'tramway',
            r'bus',
            r'transport.*reimbursement'
        ],
        'pointage_procedure': [
            r'pointer',
            r'pointage',
            r'badge.*pointer',
            r'horaires?',
            r'clock.*in',
            r'time.*tracking'
        ],
        'cantine': [
            r'cantine',
            r'repas',
            r'déjeuner',
            r'badge.*cantine',
            r'cafeteria',
            r'lunch'
        ],
        'mutuelle': [
            r'mutuelle',
            r'santé',
            r'couverture.*médicale',
            r'health.*insurance',
            r'medical.*coverage'
        ],
        'contact_rh': [
            r'contact.*rh',
            r'joindre.*rh',
            r'parler.*rh',
            r'contact.*hr',
            r'reach.*hr'
        ]
    }
    
    def classify(self, message: str) -> Tuple[str, float]:
        """
        Classify user intent
        
        Args:
            message: User message
        
        Returns:
            Tuple of (intent, confidence)
        """
        message_lower = message.lower()
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return intent, 0.85
        
        return 'general_query', 0.5


class HRChatbot:
    """
    Main HR Chatbot class
    Features:
    - Multi-language support (French, English, Arabic)
    - Profile-based responses
    - Intent classification
    - RAG (Retrieval-Augmented Generation)
    - Conversation logging (GDPR compliant)
    - Escalation to human agent
    """
    
    def __init__(self):
        self.knowledge_base = HRKnowledgeBase()
        self.intent_classifier = IntentClassifier()
        self.conversation_logs = []
        self.active_sessions = {}
        
    def authenticate_user(self, matricule: str, password: str) -> Optional[UserProfile]:
        """
        Authenticate user (simplified for POC)
        In production, integrate with AD/LDAP
        
        Args:
            matricule: Employee ID
            password: Password
        
        Returns:
            UserProfile if authenticated, None otherwise
        """
        # Simplified authentication for POC
        # In production: integrate with AD (LDAP)
        
        # Hash password for security
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Mock user database
        mock_users = {
            'EMP001': {
                'password_hash': hashlib.sha256('password123'.encode()).hexdigest(),
                'profile_type': 'CDI',
                'category': 'cadre',
                'site': 'casa_sed'
            },
            'EMP002': {
                'password_hash': hashlib.sha256('password456'.encode()).hexdigest(),
                'profile_type': 'CDI',
                'category': 'non_cadre',
                'site': 'casa_sed'
            }
        }
        
        if matricule in mock_users:
            user_data = mock_users[matricule]
            if user_data['password_hash'] == password_hash:
                return UserProfile(
                    matricule=self._anonymize_matricule(matricule),
                    profile_type=user_data['profile_type'],
                    category=user_data['category'],
                    site=user_data['site'],
                    authenticated=True
                )
        
        return None
    
    def _anonymize_matricule(self, matricule: str) -> str:
        """Anonymize employee ID for GDPR compliance"""
        return hashlib.sha256(matricule.encode()).hexdigest()[:12]
    
    def detect_language(self, message: str) -> str:
        """
        Detect message language
        
        Args:
            message: User message
        
        Returns:
            Language code ('fr', 'en', 'ar')
        """
        # Simple language detection based on character sets
        if any(char in message for char in 'àâäéèêëïîôùûüÿçæœ'):
            return 'fr'
        
        arabic_chars = set('ابتثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئة')
        if any(char in arabic_chars for char in message):
            return 'ar'
        
        return 'en'  # Default to English
    
    def generate_response(self, message: str, user_profile: UserProfile) -> ChatMessage:
        """
        Generate chatbot response
        
        Args:
            message: User message
            user_profile: User profile
        
        Returns:
            ChatMessage object
        """
        # Detect language
        language = self.detect_language(message)
        
        # Classify intent
        intent, confidence = self.intent_classifier.classify(message)
        
        # Search knowledge base
        search_results = self.knowledge_base.search_knowledge(message, user_profile)
        
        # Generate response based on intent and search results
        if search_results and confidence > 0.7:
            response = self._format_response(search_results[0], user_profile, language)
            sources = [r['source'] for r in search_results[:3]]
        else:
            # Fallback response
            response = self._generate_fallback_response(language)
            sources = []
            confidence = 0.3
        
        # Create message object
        chat_message = ChatMessage(
            message_id=self._generate_message_id(),
            user_id=user_profile.matricule,
            content=message,
            intent=intent,
            response=response,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            language=language,
            sources=sources
        )
        
        return chat_message
    
    def _format_response(self, result: Dict, user_profile: UserProfile, language: str) -> str:
        """Format response from search results"""
        content = result['content']
        
        if result['category'] == 'FAQ':
            return content['answer']
        
        # Format structured data
        response_parts = []
        
        if isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, dict):
                    response_parts.append(f"**{key.replace('_', ' ').title()}:**")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, str):
                            response_parts.append(f"  • {sub_key.replace('_', ' ').title()}: {sub_value}")
                        elif isinstance(sub_value, (int, float)):
                            response_parts.append(f"  • {sub_key.replace('_', ' ').title()}: {sub_value}")
                else:
                    if value is not None:
                        response_parts.append(f"• {key.replace('_', ' ').title()}: {value}")
        
        response = "\n".join(response_parts)
        
        # Add profile-specific note
        if user_profile.profile_type and user_profile.category:
            if language == 'fr':
                response += f"\n\n*Information pour profil {user_profile.profile_type} {user_profile.category}*"
            else:
                response += f"\n\n*Information for {user_profile.profile_type} {user_profile.category} profile*"
        
        return response if response else "Information trouvée mais format non supporté."
    
    def _generate_fallback_response(self, language: str) -> str:
        """Generate fallback response when no good match found"""
        responses = {
            'fr': "Je n'ai pas trouvé d'information précise sur votre question. Voici ce que je peux faire :\n\n"
                  "• Répondre aux questions sur les congés, la paie, les avantages sociaux\n"
                  "• Vous guider sur les procédures RH\n"
                  "• Vous donner les contacts du service RH\n\n"
                  "Pouvez-vous reformuler votre question ou contacter directement le service RH à rh.casa@safran.com ?",
            'en': "I couldn't find specific information about your question. Here's what I can help with:\n\n"
                  "• Answer questions about leave, payroll, benefits\n"
                  "• Guide you through HR procedures\n"
                  "• Provide HR contact information\n\n"
                  "Could you rephrase your question or contact HR directly at rh.casa@safran.com?",
            'ar': "لم أجد معلومات محددة حول سؤالك. يمكنني المساعدة في:\n\n"
                  "• الإجابة على أسئلة حول الإجازات والرواتب والمزايا\n"
                  "• إرشادك خلال الإجراءات\n"
                  "• تزويدك بمعلومات الاتصال\n\n"
                  "هل يمكنك إعادة صياغة سؤالك أو الاتصال مباشرة بالموارد البشرية؟"
        }
        
        return responses.get(language, responses['en'])
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        return f"MSG-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hash(datetime.now()) % 10000:04d}"
    
    def should_escalate(self, message: ChatMessage) -> bool:
        """
        Determine if conversation should be escalated to human agent
        
        Args:
            message: Chat message
        
        Returns:
            True if should escalate
        """
        # Escalate if confidence is low
        if message.confidence < 0.5:
            return True
        
        # Escalate for sensitive topics
        sensitive_keywords = ['salaire personnel', 'confidentiel', 'plainte', 'réclamation', 
                             'salary', 'confidential', 'complaint']
        
        if any(keyword in message.content.lower() for keyword in sensitive_keywords):
            return True
        
        return False
    
    def log_conversation(self, session_id: str, messages: List[ChatMessage], 
                        user_profile: UserProfile) -> ConversationLog:
        """
        Log conversation for monitoring and improvement
        
        Args:
            session_id: Session identifier
            messages: List of chat messages
            user_profile: User profile (anonymized)
        
        Returns:
            ConversationLog object
        """
        log = ConversationLog(
            session_id=session_id,
            user_profile=user_profile.matricule,  # Already anonymized
            messages=messages,
            started_at=messages[0].timestamp if messages else datetime.now().isoformat(),
            ended_at=datetime.now().isoformat(),
            escalated=any(self.should_escalate(m) for m in messages)
        )
        
        self.conversation_logs.append(log)
        
        return log
    
    def export_logs(self, output_file: str = '/home/claude/chatbot_logs.json'):
        """Export conversation logs for analysis"""
        logs_data = [asdict(log) for log in self.conversation_logs]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(logs_data, f, indent=2, ensure_ascii=False)
        
        return output_file


# Example usage
if __name__ == "__main__":
    print("HR Chatbot - Phase 1")
    print("=" * 60)
    
    # Initialize chatbot
    chatbot = HRChatbot()
    
    # Authenticate user
    user_profile = chatbot.authenticate_user('EMP001', 'password123')
    
    if user_profile:
        print(f"✓ User authenticated: {user_profile.profile_type} {user_profile.category}")
        print()
        
        # Test queries
        test_queries = [
            "Comment demander des congés ?",
            "Combien de jours de congés ai-je ?",
            "Comment obtenir mon bulletin de paie ?",
            "Quels sont mes avantages sociaux ?",
            "How do I reload my canteen badge?"
        ]
        
        session_messages = []
        
        for query in test_queries:
            print(f"\n👤 User: {query}")
            
            message = chatbot.generate_response(query, user_profile)
            session_messages.append(message)
            
            print(f"🤖 Bot ({message.language}): {message.response[:200]}...")
            print(f"   Intent: {message.intent} | Confidence: {message.confidence:.2f}")
            
            if message.sources:
                print(f"   Sources: {', '.join(message.sources)}")
            
            if chatbot.should_escalate(message):
                print("   ⚠️ Low confidence - escalation recommended")
        
        # Log conversation
        session_id = f"SESSION-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        log = chatbot.log_conversation(session_id, session_messages, user_profile)
        
        print(f"\n✓ Conversation logged: {session_id}")
        print(f"  Messages: {len(log.messages)}")
        print(f"  Escalated: {log.escalated}")
    
    else:
        print("✗ Authentication failed")
    
    print("\n" + "=" * 60)
    print("Chatbot test complete!")
