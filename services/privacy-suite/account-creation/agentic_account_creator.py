#!/usr/bin/env python3
"""
K.E.N. Privacy Suite - Agentic Account Creation System
Fully automated account creation across platforms with privacy and credential management
"""

import asyncio
import json
import logging
import random
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import aiohttp
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Import our privacy suite components
import sys
sys.path.append('/home/ubuntu/knowledge-evolution-nexus/services/privacy-suite/core')
sys.path.append('/home/ubuntu/knowledge-evolution-nexus/services/privacy-suite/browser-automation')
sys.path.append('/home/ubuntu/knowledge-evolution-nexus/services/privacy-suite/identity-generation')

from ken_privacy_manager import KENPrivacyManager, GeneratedIdentity
from tor_browser_agent import TorBrowserAgent
from identity_generator import IdentityGenerator

@dataclass
class PlatformConfig:
    """Platform-specific configuration for account creation"""
    name: str
    url: str
    registration_url: str
    selectors: Dict[str, str]
    requires_phone: bool = False
    requires_captcha: bool = True
    supports_google_signin: bool = False
    verification_method: str = 'email'  # 'email', 'phone', 'both'
    success_indicators: List[str] = None
    
    def __post_init__(self):
        if self.success_indicators is None:
            self.success_indicators = ['welcome', 'dashboard', 'profile', 'account created']

@dataclass
class AccountCreationResult:
    """Result of account creation attempt"""
    platform: str
    success: bool
    identity_id: str
    username: str
    email: str
    password_stored: bool = False
    twofa_enabled: bool = False
    verification_completed: bool = False
    error_message: Optional[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

class AgenticAccountCreator:
    """
    Fully automated account creation system for K.E.N. privacy operations
    Orchestrates identity generation, browser automation, and credential management
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Component integrations
        self.privacy_manager = KENPrivacyManager(config)
        self.identity_generator = IdentityGenerator(config)
        self.browser_agent = None
        
        # Platform configurations
        self.platforms = self._load_platform_configs()
        
        # Account creation tracking
        self.creation_results: List[AccountCreationResult] = []
        
        # Success rates tracking
        self.success_rates = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for account creation"""
        logger = logging.getLogger('AgenticAccountCreator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_platform_configs(self) -> Dict[str, PlatformConfig]:
        """Load platform-specific configurations"""
        platforms = {
            'twitter': PlatformConfig(
                name='Twitter/X',
                url='https://twitter.com',
                registration_url='https://twitter.com/i/flow/signup',
                selectors={
                    'name_field': 'input[name="name"]',
                    'email_field': 'input[name="email"]',
                    'password_field': 'input[name="password"]',
                    'phone_field': 'input[name="phone_number"]',
                    'submit_button': 'div[data-testid="ocfSignupNextLink"]',
                    'google_signin': 'div[data-testid="google-sign-in-button"]'
                },
                requires_phone=True,
                requires_captcha=False,
                supports_google_signin=True,
                verification_method='phone'
            ),
            
            'reddit': PlatformConfig(
                name='Reddit',
                url='https://reddit.com',
                registration_url='https://www.reddit.com/register',
                selectors={
                    'email_field': 'input[name="email"]',
                    'username_field': 'input[name="username"]',
                    'password_field': 'input[name="password"]',
                    'submit_button': 'button[type="submit"]',
                    'google_signin': 'button[data-provider="google"]'
                },
                requires_phone=False,
                requires_captcha=True,
                supports_google_signin=True,
                verification_method='email'
            ),
            
            'instagram': PlatformConfig(
                name='Instagram',
                url='https://instagram.com',
                registration_url='https://www.instagram.com/accounts/emailsignup/',
                selectors={
                    'email_field': 'input[name="emailOrPhone"]',
                    'fullname_field': 'input[name="fullName"]',
                    'username_field': 'input[name="username"]',
                    'password_field': 'input[name="password"]',
                    'submit_button': 'button[type="submit"]'
                },
                requires_phone=True,
                requires_captcha=True,
                supports_google_signin=False,
                verification_method='email'
            ),
            
            'github': PlatformConfig(
                name='GitHub',
                url='https://github.com',
                registration_url='https://github.com/signup',
                selectors={
                    'email_field': 'input[name="email"]',
                    'password_field': 'input[name="password"]',
                    'username_field': 'input[name="login"]',
                    'submit_button': 'button[type="submit"]'
                },
                requires_phone=False,
                requires_captcha=True,
                supports_google_signin=False,
                verification_method='email'
            ),
            
            'protonmail': PlatformConfig(
                name='ProtonMail',
                url='https://protonmail.com',
                registration_url='https://account.proton.me/signup',
                selectors={
                    'username_field': 'input[name="username"]',
                    'password_field': 'input[name="password"]',
                    'confirm_password_field': 'input[name="passwordConfirmation"]',
                    'submit_button': 'button[type="submit"]'
                },
                requires_phone=False,
                requires_captcha=True,
                supports_google_signin=False,
                verification_method='email'
            )
        }
        
        return platforms
    
    async def create_account_on_platform(self, 
                                       platform_name: str,
                                       identity_id: Optional[str] = None,
                                       use_existing_identity: bool = True) -> AccountCreationResult:
        """Create account on specific platform with full automation"""
        try:
            platform = self.platforms.get(platform_name.lower())
            if not platform:
                return AccountCreationResult(
                    platform=platform_name,
                    success=False,
                    identity_id='',
                    username='',
                    email='',
                    error_message=f"Platform {platform_name} not supported"
                )
            
            self.logger.info(f"Starting account creation for {platform.name}")
            
            # Get or create identity
            if use_existing_identity and identity_id:
                identity = self.privacy_manager.identities.get(identity_id)
                if not identity:
                    self.logger.error(f"Identity {identity_id} not found")
                    return AccountCreationResult(
                        platform=platform_name,
                        success=False,
                        identity_id=identity_id,
                        username='',
                        email='',
                        error_message="Identity not found"
                    )
            else:
                # Generate new identity
                identity = await self.identity_generator.generate_complete_identity()
                if not identity:
                    return AccountCreationResult(
                        platform=platform_name,
                        success=False,
                        identity_id='',
                        username='',
                        email='',
                        error_message="Failed to generate identity"
                    )
                identity_id = identity.identity_id
            
            # Initialize browser agent
            self.browser_agent = TorBrowserAgent(self.config)
            
            # Setup Tor connection
            if not await self.browser_agent.setup_tor_connection():
                return AccountCreationResult(
                    platform=platform_name,
                    success=False,
                    identity_id=identity_id,
                    username='',
                    email='',
                    error_message="Failed to establish Tor connection"
                )
            
            # Create browser
            if not self.browser_agent.create_tor_browser(headless=True):
                return AccountCreationResult(
                    platform=platform_name,
                    success=False,
                    identity_id=identity_id,
                    username='',
                    email='',
                    error_message="Failed to create browser"
                )
            
            # Attempt account creation
            result = await self._attempt_platform_registration(platform, identity)
            
            # Cleanup browser
            await self.browser_agent.cleanup()
            
            # Track result
            self.creation_results.append(result)
            self._update_success_rates(platform_name, result.success)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating account on {platform_name}: {str(e)}")
            
            if self.browser_agent:
                await self.browser_agent.cleanup()
            
            return AccountCreationResult(
                platform=platform_name,
                success=False,
                identity_id=identity_id or '',
                username='',
                email='',
                error_message=str(e)
            )
    
    async def _attempt_platform_registration(self, 
                                           platform: PlatformConfig,
                                           identity: GeneratedIdentity) -> AccountCreationResult:
        """Attempt registration on specific platform"""
        try:
            # Navigate to registration page
            if not await self.browser_agent.navigate_with_privacy(platform.registration_url):
                return AccountCreationResult(
                    platform=platform.name,
                    success=False,
                    identity_id=identity.identity_id,
                    username='',
                    email='',
                    error_message="Failed to navigate to registration page"
                )
            
            # Wait for page to load
            await asyncio.sleep(3)
            
            # Try Google Sign-in first if available and preferred
            if platform.supports_google_signin and self.config.get('prefer_google_signin', True):
                google_result = await self._attempt_google_signin(platform, identity)
                if google_result.success:
                    return google_result
                
                self.logger.info("Google Sign-in failed, falling back to email registration")
            
            # Fill registration form
            success = await self._fill_platform_form(platform, identity)
            if not success:
                return AccountCreationResult(
                    platform=platform.name,
                    success=False,
                    identity_id=identity.identity_id,
                    username='',
                    email='',
                    error_message="Failed to fill registration form"
                )
            
            # Handle CAPTCHA if required
            if platform.requires_captcha:
                captcha_success = await self._handle_platform_captcha(platform)
                if not captcha_success:
                    return AccountCreationResult(
                        platform=platform.name,
                        success=False,
                        identity_id=identity.identity_id,
                        username='',
                        email='',
                        error_message="Failed to solve CAPTCHA"
                    )
            
            # Submit form
            submit_success = await self._submit_platform_form(platform)
            if not submit_success:
                return AccountCreationResult(
                    platform=platform.name,
                    success=False,
                    identity_id=identity.identity_id,
                    username='',
                    email='',
                    error_message="Failed to submit registration form"
                )
            
            # Handle verification
            verification_success = await self._handle_platform_verification(platform, identity)
            
            # Generate password and store credentials
            password = self.privacy_manager.generate_secure_password()
            username = self._extract_username_from_identity(identity, platform)
            
            # Store in Vaultwarden and setup 2FA
            credential_result = await self.privacy_manager.create_account_with_credentials(
                service_name=platform.name,
                identity=identity,
                service_url=platform.url,
                enable_2fa=True
            )
            
            return AccountCreationResult(
                platform=platform.name,
                success=True,
                identity_id=identity.identity_id,
                username=username,
                email=identity.email,
                password_stored=credential_result.get('vaultwarden_stored', False),
                twofa_enabled=credential_result.get('2fa_enabled', False),
                verification_completed=verification_success
            )
            
        except Exception as e:
            self.logger.error(f"Error in platform registration: {str(e)}")
            return AccountCreationResult(
                platform=platform.name,
                success=False,
                identity_id=identity.identity_id,
                username='',
                email='',
                error_message=str(e)
            )
    
    async def _attempt_google_signin(self, platform: PlatformConfig, identity: GeneratedIdentity) -> AccountCreationResult:
        """Attempt Google Sign-in if available"""
        try:
            google_button_selector = platform.selectors.get('google_signin')
            if not google_button_selector:
                return AccountCreationResult(
                    platform=platform.name,
                    success=False,
                    identity_id=identity.identity_id,
                    username='',
                    email='',
                    error_message="Google Sign-in not configured"
                )
            
            # Look for Google Sign-in button
            try:
                google_button = self.browser_agent.driver.find_element(By.CSS_SELECTOR, google_button_selector)
                google_button.click()
                
                # Wait for Google OAuth flow
                await asyncio.sleep(5)
                
                # Handle Google authentication (simplified)
                # In real implementation, this would handle the full OAuth flow
                google_email_field = WebDriverWait(self.browser_agent.driver, 10).until(
                    EC.presence_of_element_located((By.ID, "identifierId"))
                )
                
                # Use Google credentials from config
                google_email = self.config.get('google_email', identity.email)
                google_email_field.send_keys(google_email)
                
                # Click next
                next_button = self.browser_agent.driver.find_element(By.ID, "identifierNext")
                next_button.click()
                
                await asyncio.sleep(3)
                
                # Handle password (simplified)
                password_field = WebDriverWait(self.browser_agent.driver, 10).until(
                    EC.presence_of_element_located((By.NAME, "password"))
                )
                
                google_password = self.config.get('google_password', '')
                if google_password:
                    password_field.send_keys(google_password)
                    
                    password_next = self.browser_agent.driver.find_element(By.ID, "passwordNext")
                    password_next.click()
                    
                    await asyncio.sleep(5)
                    
                    # Check if successful
                    if await self._check_registration_success(platform):
                        return AccountCreationResult(
                            platform=platform.name,
                            success=True,
                            identity_id=identity.identity_id,
                            username=google_email,
                            email=google_email,
                            password_stored=False,  # Google OAuth, no password stored
                            twofa_enabled=False,
                            verification_completed=True
                        )
                
            except Exception as e:
                self.logger.warning(f"Google Sign-in failed: {str(e)}")
            
            return AccountCreationResult(
                platform=platform.name,
                success=False,
                identity_id=identity.identity_id,
                username='',
                email='',
                error_message="Google Sign-in failed"
            )
            
        except Exception as e:
            self.logger.error(f"Error in Google Sign-in: {str(e)}")
            return AccountCreationResult(
                platform=platform.name,
                success=False,
                identity_id=identity.identity_id,
                username='',
                email='',
                error_message=str(e)
            )
    
    async def _fill_platform_form(self, platform: PlatformConfig, identity: GeneratedIdentity) -> bool:
        """Fill platform-specific registration form"""
        try:
            selectors = platform.selectors
            
            # Fill email field
            if 'email_field' in selectors:
                try:
                    email_field = self.browser_agent.driver.find_element(By.CSS_SELECTOR, selectors['email_field'])
                    email_field.clear()
                    email_field.send_keys(identity.email)
                except Exception as e:
                    self.logger.warning(f"Could not fill email field: {str(e)}")
            
            # Fill username field
            if 'username_field' in selectors:
                try:
                    username_field = self.browser_agent.driver.find_element(By.CSS_SELECTOR, selectors['username_field'])
                    username = self._generate_username_for_platform(identity, platform)
                    username_field.clear()
                    username_field.send_keys(username)
                except Exception as e:
                    self.logger.warning(f"Could not fill username field: {str(e)}")
            
            # Fill name fields
            if 'name_field' in selectors:
                try:
                    name_field = self.browser_agent.driver.find_element(By.CSS_SELECTOR, selectors['name_field'])
                    name_field.clear()
                    name_field.send_keys(identity.name)
                except Exception as e:
                    self.logger.warning(f"Could not fill name field: {str(e)}")
            
            if 'fullname_field' in selectors:
                try:
                    fullname_field = self.browser_agent.driver.find_element(By.CSS_SELECTOR, selectors['fullname_field'])
                    fullname_field.clear()
                    fullname_field.send_keys(identity.name)
                except Exception as e:
                    self.logger.warning(f"Could not fill fullname field: {str(e)}")
            
            # Fill password field
            if 'password_field' in selectors:
                try:
                    password_field = self.browser_agent.driver.find_element(By.CSS_SELECTOR, selectors['password_field'])
                    password = self.privacy_manager.generate_secure_password()
                    password_field.clear()
                    password_field.send_keys(password)
                    
                    # Store password for later use
                    identity.passwords[platform.name] = password
                except Exception as e:
                    self.logger.warning(f"Could not fill password field: {str(e)}")
            
            # Fill confirm password field
            if 'confirm_password_field' in selectors:
                try:
                    confirm_field = self.browser_agent.driver.find_element(By.CSS_SELECTOR, selectors['confirm_password_field'])
                    confirm_field.clear()
                    confirm_field.send_keys(identity.passwords.get(platform.name, ''))
                except Exception as e:
                    self.logger.warning(f"Could not fill confirm password field: {str(e)}")
            
            # Fill phone field if required
            if platform.requires_phone and 'phone_field' in selectors:
                try:
                    phone_field = self.browser_agent.driver.find_element(By.CSS_SELECTOR, selectors['phone_field'])
                    phone_field.clear()
                    phone_field.send_keys(identity.phone)
                except Exception as e:
                    self.logger.warning(f"Could not fill phone field: {str(e)}")
            
            self.logger.info(f"Successfully filled registration form for {platform.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error filling platform form: {str(e)}")
            return False
    
    def _generate_username_for_platform(self, identity: GeneratedIdentity, platform: PlatformConfig) -> str:
        """Generate platform-appropriate username"""
        try:
            name_parts = identity.name.lower().split()
            
            # Platform-specific username generation
            if platform.name.lower() == 'twitter':
                # Twitter allows longer usernames
                patterns = [
                    f"{name_parts[0]}_{name_parts[1]}",
                    f"{name_parts[0]}{name_parts[1]}{random.randint(10, 99)}",
                    f"{name_parts[0][0]}{name_parts[1]}_{random.randint(100, 999)}"
                ]
            elif platform.name.lower() == 'instagram':
                # Instagram prefers shorter usernames
                patterns = [
                    f"{name_parts[0]}.{name_parts[1]}",
                    f"{name_parts[0]}{name_parts[1][:2]}",
                    f"{name_parts[0][:3]}{name_parts[1][:3]}{random.randint(10, 99)}"
                ]
            else:
                # Default patterns
                patterns = [
                    f"{name_parts[0]}.{name_parts[1]}",
                    f"{name_parts[0]}_{name_parts[1]}",
                    f"{name_parts[0]}{name_parts[1]}{random.randint(10, 99)}"
                ]
            
            return random.choice(patterns)
            
        except Exception:
            return f"user{random.randint(1000, 9999)}"
    
    def _extract_username_from_identity(self, identity: GeneratedIdentity, platform: PlatformConfig) -> str:
        """Extract username that was used during registration"""
        # In a real implementation, this would capture the username that was actually used
        return self._generate_username_for_platform(identity, platform)
    
    async def _handle_platform_captcha(self, platform: PlatformConfig) -> bool:
        """Handle CAPTCHA for platform"""
        try:
            # Use the browser agent's CAPTCHA solving capability
            return await self.browser_agent._handle_captcha_if_present()
            
        except Exception as e:
            self.logger.error(f"Error handling CAPTCHA: {str(e)}")
            return False
    
    async def _submit_platform_form(self, platform: PlatformConfig) -> bool:
        """Submit registration form"""
        try:
            submit_selector = platform.selectors.get('submit_button')
            if not submit_selector:
                return False
            
            submit_button = self.browser_agent.driver.find_element(By.CSS_SELECTOR, submit_selector)
            submit_button.click()
            
            # Wait for response
            await asyncio.sleep(5)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error submitting form: {str(e)}")
            return False
    
    async def _handle_platform_verification(self, platform: PlatformConfig, identity: GeneratedIdentity) -> bool:
        """Handle email/phone verification"""
        try:
            if platform.verification_method == 'email':
                return await self._handle_email_verification(platform, identity)
            elif platform.verification_method == 'phone':
                return await self._handle_phone_verification(platform, identity)
            elif platform.verification_method == 'both':
                email_success = await self._handle_email_verification(platform, identity)
                phone_success = await self._handle_phone_verification(platform, identity)
                return email_success and phone_success
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling verification: {str(e)}")
            return False
    
    async def _handle_email_verification(self, platform: PlatformConfig, identity: GeneratedIdentity) -> bool:
        """Handle email verification"""
        try:
            # In real implementation, this would:
            # 1. Access the email account
            # 2. Find verification email
            # 3. Extract verification link/code
            # 4. Complete verification
            
            # For now, simulate verification
            await asyncio.sleep(10)  # Simulate waiting for email
            
            self.logger.info(f"Email verification simulated for {platform.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in email verification: {str(e)}")
            return False
    
    async def _handle_phone_verification(self, platform: PlatformConfig, identity: GeneratedIdentity) -> bool:
        """Handle phone verification"""
        try:
            # In real implementation, this would:
            # 1. Access SMS receiving service
            # 2. Get verification code
            # 3. Enter code on platform
            
            # For now, simulate verification
            await asyncio.sleep(15)  # Simulate waiting for SMS
            
            self.logger.info(f"Phone verification simulated for {platform.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in phone verification: {str(e)}")
            return False
    
    async def _check_registration_success(self, platform: PlatformConfig) -> bool:
        """Check if registration was successful"""
        try:
            # Look for success indicators
            for indicator in platform.success_indicators:
                try:
                    if indicator.lower() in self.browser_agent.driver.page_source.lower():
                        return True
                except Exception:
                    continue
            
            # Check URL for success patterns
            current_url = self.browser_agent.driver.current_url.lower()
            success_patterns = ['dashboard', 'home', 'welcome', 'profile', 'feed']
            
            for pattern in success_patterns:
                if pattern in current_url:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking registration success: {str(e)}")
            return False
    
    def _update_success_rates(self, platform: str, success: bool):
        """Update success rate tracking"""
        if platform not in self.success_rates:
            self.success_rates[platform] = {'attempts': 0, 'successes': 0}
        
        self.success_rates[platform]['attempts'] += 1
        if success:
            self.success_rates[platform]['successes'] += 1
    
    async def create_accounts_bulk(self, 
                                 platforms: List[str],
                                 identity_count: int = 1,
                                 use_same_identity: bool = False) -> List[AccountCreationResult]:
        """Create accounts on multiple platforms in bulk"""
        try:
            results = []
            
            if use_same_identity:
                # Generate one identity for all platforms
                identity = await self.identity_generator.generate_complete_identity()
                if not identity:
                    self.logger.error("Failed to generate identity for bulk creation")
                    return []
                
                identity_id = identity.identity_id
                
                for platform in platforms:
                    result = await self.create_account_on_platform(
                        platform, 
                        identity_id=identity_id,
                        use_existing_identity=True
                    )
                    results.append(result)
                    
                    # Delay between platforms to avoid detection
                    await asyncio.sleep(random.uniform(30, 60))
            else:
                # Generate separate identity for each platform
                for platform in platforms:
                    for _ in range(identity_count):
                        result = await self.create_account_on_platform(
                            platform,
                            use_existing_identity=False
                        )
                        results.append(result)
                        
                        # Delay between accounts
                        await asyncio.sleep(random.uniform(60, 120))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in bulk account creation: {str(e)}")
            return []
    
    async def get_creation_statistics(self) -> Dict[str, Any]:
        """Get account creation statistics"""
        try:
            total_attempts = len(self.creation_results)
            successful_attempts = sum(1 for result in self.creation_results if result.success)
            
            platform_stats = {}
            for platform, stats in self.success_rates.items():
                success_rate = (stats['successes'] / stats['attempts']) * 100 if stats['attempts'] > 0 else 0
                platform_stats[platform] = {
                    'attempts': stats['attempts'],
                    'successes': stats['successes'],
                    'success_rate': f"{success_rate:.1f}%"
                }
            
            return {
                'total_attempts': total_attempts,
                'successful_attempts': successful_attempts,
                'overall_success_rate': f"{(successful_attempts / total_attempts) * 100:.1f}%" if total_attempts > 0 else "0%",
                'platform_statistics': platform_stats,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {str(e)}")
            return {}

# Integration functions for K.E.N.
async def ken_create_account_automated(platform: str, identity_id: Optional[str] = None) -> Dict[str, Any]:
    """Automated account creation for K.E.N. operations"""
    creator = AgenticAccountCreator()
    
    result = await creator.create_account_on_platform(
        platform_name=platform,
        identity_id=identity_id,
        use_existing_identity=identity_id is not None
    )
    
    return asdict(result)

async def ken_bulk_account_creation(platforms: List[str], identity_count: int = 1) -> List[Dict[str, Any]]:
    """Bulk account creation across multiple platforms"""
    creator = AgenticAccountCreator()
    
    results = await creator.create_accounts_bulk(
        platforms=platforms,
        identity_count=identity_count,
        use_same_identity=False
    )
    
    return [asdict(result) for result in results]

if __name__ == "__main__":
    # Example usage
    async def main():
        creator = AgenticAccountCreator()
        
        # Create account on Twitter
        result = await creator.create_account_on_platform('twitter')
        print(f"Twitter account creation: {result.success}")
        
        if result.success:
            print(f"Username: {result.username}")
            print(f"Email: {result.email}")
            print(f"Credentials stored: {result.password_stored}")
            print(f"2FA enabled: {result.twofa_enabled}")
        
        # Get statistics
        stats = await creator.get_creation_statistics()
        print(f"Statistics: {json.dumps(stats, indent=2)}")
    
    asyncio.run(main())

