#!/usr/bin/env python3
"""
K.E.N. Privacy Suite - Core Privacy Manager
Integrates 2FAuth and Vaultwarden for secure credential management
"""

import asyncio
import json
import logging
import secrets
import string
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import requests
import aiohttp
from pathlib import Path

# Import our existing 2FAuth integration
import sys
sys.path.append('/home/ubuntu/knowledge-evolution-nexus/services/2fauth-integration')
from ken_2fa_manager import KEN2FAManager

@dataclass
class GeneratedIdentity:
    """Complete generated identity profile"""
    identity_id: str
    name: str
    email: str
    phone: str
    address: str
    birthday: str
    passwords: Dict[str, str]  # service -> password
    two_fa_accounts: List[str]  # 2FA account IDs
    protonmail_account: Optional[str] = None
    privacy_card_id: Optional[str] = None
    biometric_profile_id: Optional[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

@dataclass
class VaultwardenEntry:
    """Vaultwarden credential entry"""
    name: str
    username: str
    password: str
    uri: str
    notes: str = ""
    folder_id: Optional[str] = None

class KENPrivacyManager:
    """
    Core Privacy Manager for K.E.N. v3.0
    Orchestrates all privacy/anonymity operations with secure credential storage
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Service integrations
        self.vaultwarden_url = self.config.get('vaultwarden_url', 'http://localhost:80')
        self.twofauth_manager = KEN2FAManager(
            base_url=self.config.get('2fauth_url', 'http://localhost:8000')
        )
        
        # Session management
        self.session = aiohttp.ClientSession()
        self.vaultwarden_token = None
        
        # Identity storage
        self.identity_vault_path = Path('/home/ubuntu/knowledge-evolution-nexus/services/privacy-suite/core/identity_vault.json')
        self.identities: Dict[str, GeneratedIdentity] = {}
        
        # Load existing identities
        self._load_identities()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging for privacy operations"""
        logger = logging.getLogger('KENPrivacyManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # File handler for audit trail
            file_handler = logging.FileHandler('/var/log/ken_privacy.log')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Console handler for development
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    async def authenticate_vaultwarden(self, email: str, password: str) -> bool:
        """Authenticate with Vaultwarden for credential storage"""
        try:
            auth_data = {
                'grant_type': 'password',
                'username': email,
                'password': password,
                'scope': 'api',
                'client_id': 'web'
            }
            
            async with self.session.post(
                f"{self.vaultwarden_url}/identity/connect/token",
                data=auth_data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.vaultwarden_token = data.get('access_token')
                    self.logger.info("Successfully authenticated with Vaultwarden")
                    return True
                else:
                    self.logger.error(f"Vaultwarden authentication failed: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Vaultwarden authentication error: {str(e)}")
            return False
    
    def generate_secure_password(self, length: int = 16, include_symbols: bool = True) -> str:
        """Generate cryptographically secure password"""
        characters = string.ascii_letters + string.digits
        if include_symbols:
            characters += "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        password = ''.join(secrets.choice(characters) for _ in range(length))
        
        # Ensure password meets complexity requirements
        if not any(c.isupper() for c in password):
            password = password[:-1] + secrets.choice(string.ascii_uppercase)
        if not any(c.islower() for c in password):
            password = password[:-1] + secrets.choice(string.ascii_lowercase)
        if not any(c.isdigit() for c in password):
            password = password[:-1] + secrets.choice(string.digits)
        if include_symbols and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            password = password[:-1] + secrets.choice("!@#$%^&*")
        
        return password
    
    async def store_credential_in_vaultwarden(self, entry: VaultwardenEntry) -> bool:
        """Store credential in Vaultwarden vault"""
        if not self.vaultwarden_token:
            self.logger.error("Not authenticated with Vaultwarden")
            return False
        
        try:
            cipher_data = {
                'type': 1,  # Login type
                'name': entry.name,
                'notes': entry.notes,
                'login': {
                    'username': entry.username,
                    'password': entry.password,
                    'uris': [{'uri': entry.uri}] if entry.uri else []
                },
                'folderId': entry.folder_id
            }
            
            headers = {
                'Authorization': f'Bearer {self.vaultwarden_token}',
                'Content-Type': 'application/json'
            }
            
            async with self.session.post(
                f"{self.vaultwarden_url}/api/ciphers",
                json=cipher_data,
                headers=headers
            ) as response:
                if response.status == 200:
                    self.logger.info(f"Stored credential for {entry.name} in Vaultwarden")
                    return True
                else:
                    self.logger.error(f"Failed to store credential: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error storing credential in Vaultwarden: {str(e)}")
            return False
    
    async def add_2fa_to_account(self, service_name: str, secret: str, account_email: str) -> Optional[str]:
        """Add 2FA account to 2FAuth system"""
        try:
            # Create TOTP URI
            totp_uri = f"otpauth://totp/{service_name}:{account_email}?secret={secret}&issuer={service_name}"
            
            # Add to 2FAuth
            success = self.twofauth_manager.add_account_from_qr(totp_uri, service_name)
            
            if success:
                self.logger.info(f"Added 2FA for {service_name} to 2FAuth")
                return totp_uri
            else:
                self.logger.error(f"Failed to add 2FA for {service_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error adding 2FA account: {str(e)}")
            return None
    
    async def create_account_with_credentials(self, 
                                           service_name: str, 
                                           identity: GeneratedIdentity,
                                           service_url: str = "",
                                           enable_2fa: bool = True) -> Dict[str, Any]:
        """
        Create account and automatically store credentials in Vaultwarden and 2FAuth
        """
        try:
            # Generate secure password
            password = self.generate_secure_password()
            
            # Store in Vaultwarden
            vault_entry = VaultwardenEntry(
                name=f"{service_name} - {identity.name}",
                username=identity.email,
                password=password,
                uri=service_url,
                notes=f"Auto-generated for identity: {identity.identity_id}\nCreated: {datetime.now().isoformat()}"
            )
            
            vault_success = await self.store_credential_in_vaultwarden(vault_entry)
            
            # Add 2FA if enabled
            twofa_success = False
            if enable_2fa:
                # Generate 2FA secret
                twofa_secret = secrets.token_hex(16)
                twofa_uri = await self.add_2fa_to_account(service_name, twofa_secret, identity.email)
                twofa_success = twofa_uri is not None
            
            # Update identity with new credentials
            identity.passwords[service_name] = password
            if twofa_success:
                identity.two_fa_accounts.append(service_name)
            
            # Save updated identity
            self._save_identities()
            
            result = {
                'service': service_name,
                'username': identity.email,
                'password': password,
                'vaultwarden_stored': vault_success,
                '2fa_enabled': twofa_success,
                'identity_id': identity.identity_id
            }
            
            self.logger.info(f"Created account for {service_name} with secure credential storage")
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating account with credentials: {str(e)}")
            return {'error': str(e)}
    
    def _load_identities(self):
        """Load existing identities from secure storage"""
        try:
            if self.identity_vault_path.exists():
                with open(self.identity_vault_path, 'r') as f:
                    data = json.load(f)
                    for identity_id, identity_data in data.items():
                        self.identities[identity_id] = GeneratedIdentity(**identity_data)
                self.logger.info(f"Loaded {len(self.identities)} existing identities")
        except Exception as e:
            self.logger.error(f"Error loading identities: {str(e)}")
    
    def _save_identities(self):
        """Save identities to secure storage"""
        try:
            # Ensure directory exists
            self.identity_vault_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to serializable format
            data = {
                identity_id: asdict(identity) 
                for identity_id, identity in self.identities.items()
            }
            
            with open(self.identity_vault_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Saved {len(self.identities)} identities to vault")
        except Exception as e:
            self.logger.error(f"Error saving identities: {str(e)}")
    
    async def get_credential_for_service(self, service_name: str, identity_id: str) -> Optional[Dict[str, str]]:
        """Retrieve stored credentials for a service and identity"""
        try:
            identity = self.identities.get(identity_id)
            if not identity:
                return None
            
            password = identity.passwords.get(service_name)
            if not password:
                return None
            
            # Get 2FA code if available
            twofa_code = None
            if service_name in identity.two_fa_accounts:
                twofa_code = self.twofauth_manager.autonomous_2fa_handler(service_name)
            
            return {
                'username': identity.email,
                'password': password,
                '2fa_code': twofa_code,
                'service': service_name
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving credentials: {str(e)}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for privacy suite"""
        status = {
            'privacy_manager': 'healthy',
            'vaultwarden_connection': False,
            '2fauth_connection': False,
            'identity_vault': len(self.identities),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Check Vaultwarden
            async with self.session.get(f"{self.vaultwarden_url}/alive") as response:
                status['vaultwarden_connection'] = response.status == 200
        except:
            pass
        
        try:
            # Check 2FAuth
            status['2fauth_connection'] = self.twofauth_manager.health_check()
        except:
            pass
        
        return status
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()

# Integration functions for K.E.N. core
async def ken_create_secure_account(service_name: str, identity_id: str, service_url: str = "") -> Dict[str, Any]:
    """Main entry point for K.E.N. secure account creation"""
    manager = KENPrivacyManager()
    
    # Get or create identity
    identity = manager.identities.get(identity_id)
    if not identity:
        # This would be called from identity generation module
        return {'error': 'Identity not found'}
    
    result = await manager.create_account_with_credentials(
        service_name=service_name,
        identity=identity,
        service_url=service_url,
        enable_2fa=True
    )
    
    await manager.cleanup()
    return result

async def ken_get_secure_credentials(service_name: str, identity_id: str) -> Optional[Dict[str, str]]:
    """Get credentials with 2FA for K.E.N. operations"""
    manager = KENPrivacyManager()
    
    credentials = await manager.get_credential_for_service(service_name, identity_id)
    
    await manager.cleanup()
    return credentials

if __name__ == "__main__":
    # Example usage
    async def main():
        manager = KENPrivacyManager()
        
        # Health check
        status = await manager.health_check()
        print("Privacy Suite Status:", json.dumps(status, indent=2))
        
        await manager.cleanup()
    
    asyncio.run(main())

