#!/usr/bin/env python3
"""
K.E.N. 2FAuth Integration Configuration
Configuration settings for autonomous 2FA operations
"""

import os
from typing import Dict, Any

class KEN2FAConfig:
    """Configuration class for K.E.N. 2FAuth integration"""
    
    # Service URLs
    TWOFAUTH_BASE_URL = os.getenv('TWOFAUTH_URL', 'http://localhost:8000')
    VAULTWARDEN_BASE_URL = os.getenv('VAULTWARDEN_URL', 'http://localhost:80')
    
    # Authentication
    TWOFAUTH_EMAIL = os.getenv('TWOFAUTH_EMAIL', '')
    TWOFAUTH_PASSWORD = os.getenv('TWOFAUTH_PASSWORD', '')
    TWOFAUTH_API_TOKEN = os.getenv('TWOFAUTH_API_TOKEN', '')
    
    # Operational Settings
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    HEALTH_CHECK_INTERVAL = 60
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', '/var/log/ken_2fa.log')
    
    # Security
    ENABLE_AUDIT_LOGGING = True
    SECURE_MODE = True
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get complete configuration as dictionary"""
        return {
            '2fauth_url': cls.TWOFAUTH_BASE_URL,
            'vaultwarden_url': cls.VAULTWARDEN_BASE_URL,
            'email': cls.TWOFAUTH_EMAIL,
            'password': cls.TWOFAUTH_PASSWORD,
            'api_token': cls.TWOFAUTH_API_TOKEN,
            'timeout': cls.DEFAULT_TIMEOUT,
            'max_retries': cls.MAX_RETRIES,
            'log_level': cls.LOG_LEVEL,
            'secure_mode': cls.SECURE_MODE
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        if not cls.TWOFAUTH_BASE_URL:
            return False
        
        # Check if we have authentication method
        if not (cls.TWOFAUTH_API_TOKEN or (cls.TWOFAUTH_EMAIL and cls.TWOFAUTH_PASSWORD)):
            return False
        
        return True

# Default configuration instance
config = KEN2FAConfig()

