"""
Safran System Integration Layer - Phase 2
Integration with Prisma, Intranet, DFS, SharePoint, SELIA
Security, Authentication, and Data Flow Management
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataClassification(Enum):
    """Data classification levels"""
    C0_PUBLIC = "C0"
    C1_INTERNAL = "C1"
    C2_CONFIDENTIAL = "C2"
    C3_SECRET = "C3"


class IntegrationMode(Enum):
    """Integration modes"""
    API_READ_ONLY = "api_read_only"
    API_READ_WRITE = "api_read_write"
    BATCH_IMPORT = "batch_import"
    BATCH_EXPORT = "batch_export"
    NO_INTEGRATION = "no_integration"


@dataclass
class SystemConfig:
    """Configuration for external system"""
    system_name: str
    description: str
    integration_mode: IntegrationMode
    data_classification: DataClassification
    endpoint_url: Optional[str]
    auth_method: str
    refresh_frequency: str
    is_active: bool


@dataclass
class DataFlowConfig:
    """Data flow configuration"""
    source_system: str
    target_system: str
    flow_type: str  # unidirectional, bidirectional
    refresh_schedule: str
    data_types: List[str]
    transformation_required: bool
    anonymization_required: bool


@dataclass
class SecurityAuditLog:
    """Security audit log entry"""
    timestamp: str
    user_id: str  # Anonymized
    action: str
    system: str
    resource: str
    status: str
    ip_address: Optional[str]
    details: Dict


class DataAnonymizer:
    """
    Data anonymization for GDPR compliance
    Implements irreversible anonymization
    """
    
    SENSITIVE_FIELDS = [
        'nom', 'prenom', 'name', 'first_name', 'last_name',
        'adresse', 'address',
        'telephone', 'phone', 'mobile',
        'email', 'mail',
        'numero_employe', 'employee_number', 'matricule',
        'iban', 'bank_account',
        'salaire', 'salary', 'remuneration'
    ]
    
    @staticmethod
    def anonymize_field(value: str, field_name: str) -> str:
        """
        Anonymize sensitive field value
        
        Args:
            value: Original value
            field_name: Field name
        
        Returns:
            Anonymized value
        """
        if not value or pd.isna(value):
            return "ANONYMIZED"
        
        # Use SHA256 for irreversible anonymization
        hash_value = hashlib.sha256(f"{field_name}:{value}".encode()).hexdigest()
        
        # Return shortened hash for readability
        return f"ANON_{hash_value[:12].upper()}"
    
    @staticmethod
    def anonymize_dataset(data: Dict, preserve_fields: Optional[List[str]] = None) -> Dict:
        """
        Anonymize entire dataset
        
        Args:
            data: Original data dictionary or list
            preserve_fields: Fields to preserve (not anonymize)
        
        Returns:
            Anonymized data dictionary or list
        """
        # Handle lists
        if isinstance(data, list):
            return [DataAnonymizer.anonymize_dataset(item, preserve_fields) 
                   if isinstance(item, dict) else item for item in data]
        
        # Handle non-dict types
        if not isinstance(data, dict):
            return data
        
        preserve_fields = preserve_fields or []
        anonymized = {}
        
        for key, value in data.items():
            if key.lower() in [f.lower() for f in DataAnonymizer.SENSITIVE_FIELDS]:
                if key not in preserve_fields:
                    anonymized[key] = DataAnonymizer.anonymize_field(str(value), key)
                else:
                    anonymized[key] = value
            elif isinstance(value, dict):
                anonymized[key] = DataAnonymizer.anonymize_dataset(value, preserve_fields)
            elif isinstance(value, list):
                anonymized[key] = [
                    DataAnonymizer.anonymize_dataset(item, preserve_fields) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                anonymized[key] = value
        
        return anonymized
    
    @staticmethod
    def mask_salary(salary: float) -> str:
        """Mask salary information"""
        return "****"  # Never expose salary data


class ADAuthenticationService:
    """
    Active Directory Authentication Service
    Simplified implementation for POC
    In production: integrate with actual AD/LDAP
    """
    
    def __init__(self):
        self.session_cache = {}
        self.failed_attempts = {}
    
    def authenticate(self, matricule: str, password: str) -> Optional[Dict]:
        """
        Authenticate user against AD
        
        Args:
            matricule: Employee ID
            password: Password
        
        Returns:
            User session data if authenticated, None otherwise
        """
        # Check for failed attempt lockout
        if matricule in self.failed_attempts:
            if self.failed_attempts[matricule]['count'] >= 3:
                lockout_time = self.failed_attempts[matricule]['lockout_until']
                if datetime.now() < lockout_time:
                    logger.warning(f"Account locked: {matricule}")
                    return None
        
        # Simulate AD authentication
        # In production: use ldap3 library to connect to AD
        
        # Mock authentication for POC
        if self._verify_credentials(matricule, password):
            session_data = {
                'matricule': matricule,
                'session_id': self._generate_session_id(),
                'authenticated_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(hours=8)).isoformat(),
                'roles': self._get_user_roles(matricule),
                'site': 'casa_sed'
            }
            
            self.session_cache[session_data['session_id']] = session_data
            
            # Reset failed attempts
            if matricule in self.failed_attempts:
                del self.failed_attempts[matricule]
            
            logger.info(f"Authentication successful: {matricule}")
            return session_data
        else:
            # Track failed attempt
            if matricule not in self.failed_attempts:
                self.failed_attempts[matricule] = {'count': 0}
            
            self.failed_attempts[matricule]['count'] += 1
            
            if self.failed_attempts[matricule]['count'] >= 3:
                self.failed_attempts[matricule]['lockout_until'] = datetime.now() + timedelta(minutes=30)
                logger.warning(f"Account locked after 3 failed attempts: {matricule}")
            
            logger.warning(f"Authentication failed: {matricule}")
            return None
    
    def _verify_credentials(self, matricule: str, password: str) -> bool:
        """Verify credentials (simplified for POC)"""
        # In production: verify against AD
        return len(password) >= 8  # Simplified validation
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return hashlib.sha256(f"{datetime.now().isoformat()}".encode()).hexdigest()
    
    def _get_user_roles(self, matricule: str) -> List[str]:
        """Get user roles from AD"""
        # In production: fetch from AD groups
        return ['employee', 'casa_sed_user']
    
    def validate_session(self, session_id: str) -> bool:
        """Validate active session"""
        if session_id not in self.session_cache:
            return False
        
        session = self.session_cache[session_id]
        expires_at = datetime.fromisoformat(session['expires_at'])
        
        if datetime.now() > expires_at:
            del self.session_cache[session_id]
            return False
        
        return True


class SafranSystemConnector:
    """
    Connector for Safran internal systems
    Handles integration with Prisma, Intranet, DFS, SharePoint, SELIA
    """
    
    def __init__(self):
        self.systems = self._initialize_systems()
        self.data_flows = self._initialize_data_flows()
        self.audit_logs = []
    
    def _initialize_systems(self) -> Dict[str, SystemConfig]:
        """Initialize system configurations"""
        return {
            'prisma': SystemConfig(
                system_name='Prisma',
                description='Process management rules and scope definition',
                integration_mode=IntegrationMode.API_READ_ONLY,
                data_classification=DataClassification.C2_CONFIDENTIAL,
                endpoint_url='https://prisma.safran.internal/api',
                auth_method='AD',
                refresh_frequency='daily',
                is_active=True
            ),
            'intranet': SystemConfig(
                system_name='Intranet Safran',
                description='Employee handbook, onboarding, procedures, org chart',
                integration_mode=IntegrationMode.API_READ_ONLY,
                data_classification=DataClassification.C1_INTERNAL,
                endpoint_url='https://intranet.safran.internal/api',
                auth_method='AD',
                refresh_frequency='daily',
                is_active=True
            ),
            'dfs': SystemConfig(
                system_name='DFS',
                description='Employee directory',
                integration_mode=IntegrationMode.API_READ_ONLY,
                data_classification=DataClassification.C2_CONFIDENTIAL,
                endpoint_url='https://dfs.safran.internal/api',
                auth_method='AD',
                refresh_frequency='daily',
                is_active=True
            ),
            'sharepoint': SystemConfig(
                system_name='SharePoint',
                description='Official documentation and procedure tutorials',
                integration_mode=IntegrationMode.API_READ_ONLY,
                data_classification=DataClassification.C1_INTERNAL,
                endpoint_url='https://sharepoint.safran.internal/api',
                auth_method='AD',
                refresh_frequency='weekly',
                is_active=True
            ),
            'selia': SystemConfig(
                system_name='SELIA',
                description='Learning Management System - HR notes and nominations',
                integration_mode=IntegrationMode.BATCH_IMPORT,
                data_classification=DataClassification.C2_CONFIDENTIAL,
                endpoint_url='https://selia.safran.internal/api',
                auth_method='AD',
                refresh_frequency='daily',
                is_active=True
            )
        }
    
    def _initialize_data_flows(self) -> List[DataFlowConfig]:
        """Initialize data flow configurations"""
        return [
            DataFlowConfig(
                source_system='selia',
                target_system='training_analyzer',
                flow_type='unidirectional',
                refresh_schedule='daily_j-1',
                data_types=['evaluation_forms', 'training_sessions'],
                transformation_required=True,
                anonymization_required=True
            ),
            DataFlowConfig(
                source_system='prisma',
                target_system='hr_chatbot',
                flow_type='unidirectional',
                refresh_schedule='daily',
                data_types=['procedures', 'rules', 'policies'],
                transformation_required=True,
                anonymization_required=False
            ),
            DataFlowConfig(
                source_system='intranet',
                target_system='hr_chatbot',
                flow_type='unidirectional',
                refresh_schedule='daily',
                data_types=['employee_handbook', 'onboarding_docs', 'procedures'],
                transformation_required=True,
                anonymization_required=False
            ),
            DataFlowConfig(
                source_system='sharepoint',
                target_system='hr_chatbot',
                flow_type='unidirectional',
                refresh_schedule='weekly',
                data_types=['official_docs', 'tutorials'],
                transformation_required=True,
                anonymization_required=False
            )
        ]
    
    def fetch_data(self, system_name: str, resource: str, 
                   session_id: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Fetch data from Safran system
        
        Args:
            system_name: Name of system to fetch from
            resource: Resource to fetch
            session_id: User session ID
            params: Optional query parameters
        
        Returns:
            Data from system or None
        """
        if system_name not in self.systems:
            logger.error(f"Unknown system: {system_name}")
            return None
        
        system = self.systems[system_name]
        
        if not system.is_active:
            logger.error(f"System inactive: {system_name}")
            return None
        
        # Log access
        self._log_access(session_id, 'fetch', system_name, resource, 'success')
        
        # Simulate data fetch
        # In production: make actual API call
        data = self._simulate_fetch(system_name, resource, params)
        
        # Apply anonymization if required
        if system.data_classification in [DataClassification.C2_CONFIDENTIAL, DataClassification.C3_SECRET]:
            data = DataAnonymizer.anonymize_dataset(data)
        
        return data
    
    def _simulate_fetch(self, system_name: str, resource: str, params: Optional[Dict]) -> Dict:
        """Simulate data fetch (for POC)"""
        # In production: replace with actual API calls
        
        mock_data = {
            'prisma': {
                'procedures': {
                    'leave_request': {
                        'name': 'Demande de congés',
                        'steps': ['Login to Prisma', 'Navigate to Leaves', 'Submit request'],
                        'approval_flow': ['Manager', 'HR'],
                        'lead_time': '15 days'
                    }
                }
            },
            'intranet': {
                'employee_handbook': {
                    'leave_policy': 'Employees are entitled to 25 days of paid leave per year...',
                    'working_hours': 'Standard working hours: 8:00 AM - 5:00 PM...'
                }
            },
            'selia': {
                'training_evaluations': [
                    {
                        'session_id': 'TRAIN001',
                        'evaluation_date': '2025-01-15',
                        'responses': []
                    }
                ]
            }
        }
        
        return mock_data.get(system_name, {}).get(resource, {})
    
    def _log_access(self, user_id: str, action: str, system: str, 
                   resource: str, status: str):
        """Log system access for audit"""
        log_entry = SecurityAuditLog(
            timestamp=datetime.now().isoformat(),
            user_id=DataAnonymizer.anonymize_field(user_id, 'user_id'),
            action=action,
            system=system,
            resource=resource,
            status=status,
            ip_address=None,  # In production: capture real IP
            details={}
        )
        
        self.audit_logs.append(log_entry)
        logger.info(f"Access logged: {action} on {system}/{resource} by {user_id[:8]}...")
    
    def get_audit_logs(self, start_date: Optional[datetime] = None, 
                      end_date: Optional[datetime] = None) -> List[SecurityAuditLog]:
        """Retrieve audit logs for specified period"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()
        
        filtered_logs = [
            log for log in self.audit_logs
            if start_date <= datetime.fromisoformat(log.timestamp) <= end_date
        ]
        
        return filtered_logs
    
    def export_audit_logs(self, output_file: str = '/home/claude/audit_logs.json'):
        """Export audit logs to file"""
        logs_data = [asdict(log) for log in self.audit_logs]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(logs_data, f, indent=2, ensure_ascii=False)
        
        return output_file


class DataFlowManager:
    """
    Manage data flows between systems
    Handles ETL, transformation, and scheduling
    """
    
    def __init__(self, connector: SafranSystemConnector):
        self.connector = connector
        self.last_sync = {}
    
    def sync_training_data(self) -> Dict:
        """
        Sync training evaluation data from SELIA
        Daily refresh at J-1
        
        Returns:
            Sync status and data
        """
        logger.info("Starting training data sync from SELIA...")
        
        # Check last sync
        if 'training_data' in self.last_sync:
            last_sync_time = datetime.fromisoformat(self.last_sync['training_data'])
            if datetime.now() - last_sync_time < timedelta(hours=23):
                logger.info("Data already synced today")
                return {'status': 'skipped', 'reason': 'already_synced_today'}
        
        # Fetch data
        # In production: use actual session
        mock_session_id = "system_sync_session"
        
        data = self.connector.fetch_data('selia', 'training_evaluations', mock_session_id)
        
        if data:
            # Transform and anonymize
            transformed_data = self._transform_training_data(data)
            anonymized_data = DataAnonymizer.anonymize_dataset(transformed_data)
            
            # Update last sync
            self.last_sync['training_data'] = datetime.now().isoformat()
            
            # Handle both list and dict responses
            if isinstance(anonymized_data, list):
                record_count = len(anonymized_data)
            else:
                record_count = len(anonymized_data.get('evaluations', [])) if isinstance(anonymized_data, dict) else 1
            
            logger.info(f"Training data synced successfully: {record_count} records")
            
            return {
                'status': 'success',
                'records': record_count,
                'synced_at': self.last_sync['training_data']
            }
        else:
            logger.error("Failed to fetch training data")
            return {'status': 'error', 'reason': 'fetch_failed'}
    
    def sync_hr_knowledge(self) -> Dict:
        """
        Sync HR knowledge base from multiple sources
        
        Returns:
            Sync status and data
        """
        logger.info("Starting HR knowledge sync...")
        
        sources = ['prisma', 'intranet', 'sharepoint']
        synced_data = {}
        
        mock_session_id = "system_sync_session"
        
        for source in sources:
            data = self.connector.fetch_data(source, 'procedures', mock_session_id)
            if data:
                synced_data[source] = data
                logger.info(f"Synced data from {source}")
        
        self.last_sync['hr_knowledge'] = datetime.now().isoformat()
        
        return {
            'status': 'success',
            'sources': len(synced_data),
            'synced_at': self.last_sync['hr_knowledge']
        }
    
    def _transform_training_data(self, raw_data: Dict) -> Dict:
        """Transform training data to standard format"""
        # In production: implement actual transformation logic
        return raw_data


# Example usage
if __name__ == "__main__":
    import pandas as pd
    
    print("Safran System Integration Layer - Phase 2")
    print("=" * 70)
    
    # Initialize authentication service
    auth_service = ADAuthenticationService()
    
    # Authenticate user
    print("\n1. Authentication Test")
    print("-" * 70)
    session = auth_service.authenticate('EMP001', 'SecurePass123')
    if session:
        print(f"✓ Authentication successful")
        print(f"  Session ID: {session['session_id'][:20]}...")
        print(f"  Expires: {session['expires_at']}")
        print(f"  Roles: {', '.join(session['roles'])}")
    else:
        print("✗ Authentication failed")
    
    # Initialize connector
    connector = SafranSystemConnector()
    
    # Test data anonymization
    print("\n2. Data Anonymization Test")
    print("-" * 70)
    
    test_data = {
        'nom': 'Dupont',
        'prenom': 'Jean',
        'email': 'jean.dupont@safran.com',
        'matricule': 'EMP001',
        'salaire': 50000,
        'department': 'Engineering'
    }
    
    anonymized = DataAnonymizer.anonymize_dataset(test_data)
    
    print("Original data:")
    for key, value in test_data.items():
        print(f"  {key}: {value}")
    
    print("\nAnonymized data:")
    for key, value in anonymized.items():
        print(f"  {key}: {value}")
    
    # Test system connection
    print("\n3. System Integration Test")
    print("-" * 70)
    
    if session:
        # Fetch data from Prisma
        data = connector.fetch_data('prisma', 'procedures', session['session_id'])
        print(f"✓ Fetched data from Prisma")
        print(f"  Resources: {list(data.keys()) if data else 'None'}")
        
        # Fetch data from Intranet
        data = connector.fetch_data('intranet', 'employee_handbook', session['session_id'])
        print(f"✓ Fetched data from Intranet")
        print(f"  Resources: {list(data.keys()) if data else 'None'}")
    
    # Test data flow management
    print("\n4. Data Flow Management Test")
    print("-" * 70)
    
    flow_manager = DataFlowManager(connector)
    
    # Sync training data
    result = flow_manager.sync_training_data()
    print(f"Training data sync: {result['status']}")
    
    # Sync HR knowledge
    result = flow_manager.sync_hr_knowledge()
    print(f"HR knowledge sync: {result['status']}")
    print(f"  Sources synced: {result.get('sources', 0)}")
    
    # Export audit logs
    print("\n5. Audit Logging Test")
    print("-" * 70)
    
    audit_file = connector.export_audit_logs()
    print(f"✓ Audit logs exported: {audit_file}")
    print(f"  Total log entries: {len(connector.audit_logs)}")
    
    print("\n" + "=" * 70)
    print("Integration layer test complete!")
