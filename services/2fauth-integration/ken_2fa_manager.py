#!/usr/bin/env python3
"""
K.E.N. 2FAuth Integration Manager
Autonomous 2FA management for K.E.N. v3.0 system
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class TwoFAAccount:
    """Represents a 2FA account in the system"""
    id: int
    service: str
    account: str
    secret: str
    algorithm: str = "sha1"
    digits: int = 6
    period: int = 30
    counter: Optional[int] = None
    
class KEN2FAManager:
    """
    K.E.N. Autonomous 2FA Manager
    Provides autonomous 2FA capabilities for K.E.N. system operations
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", api_token: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_token = api_token
        self.session = requests.Session()
        self.logger = self._setup_logging()
        
        if api_token:
            self.session.headers.update({
                'Authorization': f'Bearer {api_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the 2FA manager"""
        logger = logging.getLogger('KEN2FAManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def authenticate(self, email: str, password: str) -> bool:
        """
        Authenticate with 2FAuth system
        Returns True if authentication successful
        """
        try:
            auth_data = {
                'email': email,
                'password': password
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/user/login",
                json=auth_data
            )
            
            if response.status_code == 200:
                data = response.json()
                self.api_token = data.get('access_token')
                self.session.headers.update({
                    'Authorization': f'Bearer {self.api_token}'
                })
                self.logger.info("Successfully authenticated with 2FAuth")
                return True
            else:
                self.logger.error(f"Authentication failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Authentication error: {str(e)}")
            return False
    
    def get_all_accounts(self) -> List[TwoFAAccount]:
        """
        Retrieve all 2FA accounts from the system
        Returns list of TwoFAAccount objects
        """
        try:
            response = self.session.get(f"{self.base_url}/api/v1/twofaccounts")
            
            if response.status_code == 200:
                accounts_data = response.json()
                accounts = []
                
                for account_data in accounts_data:
                    account = TwoFAAccount(
                        id=account_data.get('id'),
                        service=account_data.get('service', ''),
                        account=account_data.get('account', ''),
                        secret=account_data.get('secret', ''),
                        algorithm=account_data.get('algorithm', 'sha1'),
                        digits=account_data.get('digits', 6),
                        period=account_data.get('period', 30),
                        counter=account_data.get('counter')
                    )
                    accounts.append(account)
                
                self.logger.info(f"Retrieved {len(accounts)} 2FA accounts")
                return accounts
            else:
                self.logger.error(f"Failed to retrieve accounts: {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error retrieving accounts: {str(e)}")
            return []
    
    def generate_otp(self, account_id: int) -> Optional[str]:
        """
        Generate OTP for specific account
        Returns OTP code as string or None if failed
        """
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/twofaccounts/{account_id}/otp"
            )
            
            if response.status_code == 200:
                data = response.json()
                otp_code = data.get('generated_at')
                self.logger.info(f"Generated OTP for account {account_id}")
                return otp_code
            else:
                self.logger.error(f"Failed to generate OTP: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating OTP: {str(e)}")
            return None
    
    def autonomous_2fa_handler(self, service_name: str) -> Optional[str]:
        """
        Autonomous 2FA handler for K.E.N. system
        Automatically finds and generates 2FA code for specified service
        """
        try:
            accounts = self.get_all_accounts()
            
            # Find matching account
            matching_account = None
            for account in accounts:
                if service_name.lower() in account.service.lower() or \
                   service_name.lower() in account.account.lower():
                    matching_account = account
                    break
            
            if not matching_account:
                self.logger.warning(f"No 2FA account found for service: {service_name}")
                return None
            
            # Generate OTP
            otp_code = self.generate_otp(matching_account.id)
            
            if otp_code:
                self.logger.info(f"Autonomous 2FA successful for {service_name}")
                return otp_code
            else:
                self.logger.error(f"Failed to generate 2FA code for {service_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Autonomous 2FA handler error: {str(e)}")
            return None
    
    def add_account_from_qr(self, qr_code_data: str, service_name: str = None) -> bool:
        """
        Add new 2FA account from QR code data
        Returns True if successful
        """
        try:
            account_data = {
                'uri': qr_code_data,
                'service': service_name or 'K.E.N. Service'
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/twofaccounts",
                json=account_data
            )
            
            if response.status_code == 201:
                self.logger.info(f"Successfully added 2FA account: {service_name}")
                return True
            else:
                self.logger.error(f"Failed to add account: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error adding account: {str(e)}")
            return False
    
    def health_check(self) -> bool:
        """
        Check if 2FAuth service is healthy and accessible
        Returns True if healthy
        """
        try:
            response = self.session.get(f"{self.base_url}/up")
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False
    
    def get_service_status(self) -> Dict:
        """
        Get comprehensive service status for K.E.N. monitoring
        """
        status = {
            'service': '2FAuth',
            'healthy': self.health_check(),
            'timestamp': datetime.now().isoformat(),
            'accounts_count': len(self.get_all_accounts()),
            'authenticated': bool(self.api_token)
        }
        
        return status

# K.E.N. Integration Functions
def ken_autonomous_2fa(service_name: str, config: Dict = None) -> Optional[str]:
    """
    Main entry point for K.E.N. autonomous 2FA operations
    """
    config = config or {}
    base_url = config.get('2fauth_url', 'http://localhost:8000')
    
    manager = KEN2FAManager(base_url=base_url)
    
    # Try to authenticate if credentials provided
    if config.get('email') and config.get('password'):
        if not manager.authenticate(config['email'], config['password']):
            return None
    
    return manager.autonomous_2fa_handler(service_name)

def ken_2fa_health_check(config: Dict = None) -> Dict:
    """
    Health check function for K.E.N. monitoring system
    """
    config = config or {}
    base_url = config.get('2fauth_url', 'http://localhost:8000')
    
    manager = KEN2FAManager(base_url=base_url)
    return manager.get_service_status()

if __name__ == "__main__":
    # Example usage
    manager = KEN2FAManager()
    
    # Health check
    print("2FAuth Health Status:", manager.get_service_status())
    
    # Example autonomous 2FA (requires authentication)
    # otp = manager.autonomous_2fa_handler("GitHub")
    # print(f"Generated OTP: {otp}")

