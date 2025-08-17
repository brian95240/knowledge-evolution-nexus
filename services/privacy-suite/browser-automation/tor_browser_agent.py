#!/usr/bin/env python3
"""
K.E.N. Privacy Suite - Tor Browser Automation Agent
Headless Tor browser automation with anti-fingerprinting and CAPTCHA solving
"""

import asyncio
import json
import logging
import random
import time
import tempfile
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import requests
import aiohttp
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.common.exceptions import TimeoutException, WebDriverException
import undetected_chromedriver as uc
from fake_useragent import UserAgent

# Import our credential management
import sys
sys.path.append('/home/ubuntu/knowledge-evolution-nexus/services/privacy-suite/core')
from ken_privacy_manager import KENPrivacyManager

@dataclass
class BrowserProfile:
    """Browser fingerprint profile for anonymity"""
    user_agent: str
    screen_resolution: Tuple[int, int]
    timezone: str
    language: str
    platform: str
    webgl_vendor: str
    webgl_renderer: str
    canvas_fingerprint: str
    audio_fingerprint: str
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

@dataclass
class TorCircuit:
    """Tor circuit information"""
    circuit_id: str
    entry_node: str
    middle_node: str
    exit_node: str
    country: str
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

class TorBrowserAgent:
    """
    Advanced Tor browser automation with anti-fingerprinting and privacy features
    Integrates with K.E.N. credential management for secure operations
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Browser configuration
        self.tor_proxy_host = self.config.get('tor_proxy_host', '127.0.0.1')
        self.tor_proxy_port = self.config.get('tor_proxy_port', 9050)
        self.tor_control_port = self.config.get('tor_control_port', 9051)
        
        # Anti-fingerprinting
        self.user_agent_rotator = UserAgent()
        self.current_profile: Optional[BrowserProfile] = None
        self.current_circuit: Optional[TorCircuit] = None
        
        # Browser instances
        self.driver: Optional[webdriver.Firefox] = None
        self.session_id = None
        
        # CAPTCHA solving
        self.captcha_service = self.config.get('captcha_service', '2captcha')
        self.captcha_api_key = self.config.get('captcha_api_key', '')
        
        # Privacy manager integration
        self.privacy_manager = KENPrivacyManager(config)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for browser automation"""
        logger = logging.getLogger('TorBrowserAgent')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def generate_browser_profile(self) -> BrowserProfile:
        """Generate realistic browser fingerprint profile"""
        try:
            # Generate realistic user agent
            user_agent = self.user_agent_rotator.random
            
            # Common screen resolutions
            resolutions = [
                (1920, 1080), (1366, 768), (1440, 900), (1536, 864),
                (1280, 720), (1600, 900), (1024, 768), (1280, 1024)
            ]
            screen_resolution = random.choice(resolutions)
            
            # Common timezones
            timezones = [
                'America/New_York', 'America/Los_Angeles', 'Europe/London',
                'Europe/Berlin', 'Asia/Tokyo', 'Australia/Sydney'
            ]
            timezone = random.choice(timezones)
            
            # Languages
            languages = ['en-US', 'en-GB', 'de-DE', 'fr-FR', 'es-ES', 'it-IT']
            language = random.choice(languages)
            
            # Platforms
            platforms = ['Win32', 'Linux x86_64', 'MacIntel']
            platform = random.choice(platforms)
            
            # WebGL fingerprints
            webgl_vendors = ['Google Inc.', 'Mozilla', 'Apple Inc.']
            webgl_renderers = [
                'ANGLE (Intel HD Graphics 4000 Direct3D11 vs_5_0 ps_5_0)',
                'ANGLE (NVIDIA GeForce GTX 1060 Direct3D11 vs_5_0 ps_5_0)',
                'Intel Open Source Technology Center Mesa DRI Intel(R) HD Graphics'
            ]
            
            profile = BrowserProfile(
                user_agent=user_agent,
                screen_resolution=screen_resolution,
                timezone=timezone,
                language=language,
                platform=platform,
                webgl_vendor=random.choice(webgl_vendors),
                webgl_renderer=random.choice(webgl_renderers),
                canvas_fingerprint=self._generate_canvas_fingerprint(),
                audio_fingerprint=self._generate_audio_fingerprint()
            )
            
            self.current_profile = profile
            self.logger.info("Generated new browser profile for anonymity")
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Error generating browser profile: {str(e)}")
            return None
    
    def _generate_canvas_fingerprint(self) -> str:
        """Generate realistic canvas fingerprint"""
        # Simulate canvas fingerprint generation
        import hashlib
        random_data = f"{random.random()}{time.time()}"
        return hashlib.md5(random_data.encode()).hexdigest()[:16]
    
    def _generate_audio_fingerprint(self) -> str:
        """Generate realistic audio fingerprint"""
        # Simulate audio fingerprint generation
        import hashlib
        random_data = f"{random.random()}{time.time()}"
        return hashlib.md5(random_data.encode()).hexdigest()[:16]
    
    async def setup_tor_connection(self) -> bool:
        """Setup and verify Tor connection"""
        try:
            # Check if Tor is running
            tor_check_url = f"http://{self.tor_proxy_host}:{self.tor_proxy_port}"
            
            # Test SOCKS proxy connection
            proxies = {
                'http': f'socks5h://{self.tor_proxy_host}:{self.tor_proxy_port}',
                'https': f'socks5h://{self.tor_proxy_host}:{self.tor_proxy_port}'
            }
            
            async with aiohttp.ClientSession() as session:
                try:
                    # Test connection through Tor
                    async with session.get(
                        'https://check.torproject.org/api/ip',
                        proxy=f'socks5://{self.tor_proxy_host}:{self.tor_proxy_port}',
                        timeout=30
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get('IsTor'):
                                self.logger.info("Tor connection verified")
                                return True
                            else:
                                self.logger.warning("Connection not through Tor")
                                return False
                except Exception as e:
                    self.logger.error(f"Tor connection test failed: {str(e)}")
                    return False
            
        except Exception as e:
            self.logger.error(f"Error setting up Tor connection: {str(e)}")
            return False
    
    async def rotate_tor_circuit(self) -> bool:
        """Rotate Tor circuit for new IP"""
        try:
            # Connect to Tor control port
            import socket
            
            control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            control_socket.connect((self.tor_proxy_host, self.tor_control_port))
            
            # Authenticate (assuming no password for simplicity)
            control_socket.send(b'AUTHENTICATE\r\n')
            response = control_socket.recv(1024)
            
            if b'250 OK' in response:
                # Send new circuit signal
                control_socket.send(b'SIGNAL NEWNYM\r\n')
                response = control_socket.recv(1024)
                
                if b'250 OK' in response:
                    self.logger.info("Tor circuit rotated successfully")
                    control_socket.close()
                    
                    # Wait for new circuit to establish
                    await asyncio.sleep(10)
                    return True
            
            control_socket.close()
            return False
            
        except Exception as e:
            self.logger.error(f"Error rotating Tor circuit: {str(e)}")
            return False
    
    def create_tor_browser(self, headless: bool = True) -> Optional[webdriver.Firefox]:
        """Create Tor browser instance with anti-fingerprinting"""
        try:
            # Generate new profile if needed
            if not self.current_profile:
                self.generate_browser_profile()
            
            # Firefox options for Tor
            options = FirefoxOptions()
            
            if headless:
                options.add_argument('--headless')
            
            # Tor proxy configuration
            options.set_preference('network.proxy.type', 1)
            options.set_preference('network.proxy.socks', self.tor_proxy_host)
            options.set_preference('network.proxy.socks_port', self.tor_proxy_port)
            options.set_preference('network.proxy.socks_version', 5)
            options.set_preference('network.proxy.socks_remote_dns', True)
            
            # Anti-fingerprinting preferences
            options.set_preference('general.useragent.override', self.current_profile.user_agent)
            options.set_preference('intl.accept_languages', self.current_profile.language)
            options.set_preference('privacy.resistFingerprinting', True)
            options.set_preference('privacy.trackingprotection.enabled', True)
            options.set_preference('privacy.trackingprotection.socialtracking.enabled', True)
            
            # Disable WebRTC
            options.set_preference('media.peerconnection.enabled', False)
            options.set_preference('media.navigator.enabled', False)
            
            # Disable geolocation
            options.set_preference('geo.enabled', False)
            
            # Disable telemetry
            options.set_preference('toolkit.telemetry.enabled', False)
            options.set_preference('datareporting.healthreport.uploadEnabled', False)
            
            # Canvas fingerprinting protection
            options.set_preference('privacy.resistFingerprinting.autoDeclineNoUserInputCanvasPrompts', True)
            
            # Create driver
            self.driver = webdriver.Firefox(options=options)
            
            # Set window size to match profile
            self.driver.set_window_size(*self.current_profile.screen_resolution)
            
            # Execute anti-fingerprinting JavaScript
            self._inject_anti_fingerprinting_scripts()
            
            self.logger.info("Created Tor browser with anti-fingerprinting measures")
            return self.driver
            
        except Exception as e:
            self.logger.error(f"Error creating Tor browser: {str(e)}")
            return None
    
    def _inject_anti_fingerprinting_scripts(self):
        """Inject JavaScript to prevent fingerprinting"""
        try:
            if not self.driver:
                return
            
            # Canvas fingerprinting protection
            canvas_script = """
            const originalGetContext = HTMLCanvasElement.prototype.getContext;
            HTMLCanvasElement.prototype.getContext = function(type, ...args) {
                const context = originalGetContext.call(this, type, ...args);
                if (type === '2d' || type === 'webgl' || type === 'webgl2') {
                    // Add noise to canvas data
                    const originalGetImageData = context.getImageData;
                    context.getImageData = function(...args) {
                        const imageData = originalGetImageData.apply(this, args);
                        for (let i = 0; i < imageData.data.length; i += 4) {
                            imageData.data[i] += Math.floor(Math.random() * 3) - 1;
                        }
                        return imageData;
                    };
                }
                return context;
            };
            """
            
            # WebGL fingerprinting protection
            webgl_script = f"""
            const originalGetParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {{
                if (parameter === this.VENDOR) {{
                    return '{self.current_profile.webgl_vendor}';
                }}
                if (parameter === this.RENDERER) {{
                    return '{self.current_profile.webgl_renderer}';
                }}
                return originalGetParameter.call(this, parameter);
            }};
            """
            
            # Audio fingerprinting protection
            audio_script = """
            const originalCreateAnalyser = AudioContext.prototype.createAnalyser;
            AudioContext.prototype.createAnalyser = function() {
                const analyser = originalCreateAnalyser.call(this);
                const originalGetFloatFrequencyData = analyser.getFloatFrequencyData;
                analyser.getFloatFrequencyData = function(array) {
                    originalGetFloatFrequencyData.call(this, array);
                    for (let i = 0; i < array.length; i++) {
                        array[i] += Math.random() * 0.1 - 0.05;
                    }
                };
                return analyser;
            };
            """
            
            # Execute scripts
            self.driver.execute_script(canvas_script)
            self.driver.execute_script(webgl_script)
            self.driver.execute_script(audio_script)
            
            self.logger.info("Injected anti-fingerprinting scripts")
            
        except Exception as e:
            self.logger.error(f"Error injecting anti-fingerprinting scripts: {str(e)}")
    
    async def solve_captcha(self, captcha_type: str, site_key: str, page_url: str) -> Optional[str]:
        """Solve CAPTCHA using external service"""
        try:
            if not self.captcha_api_key:
                self.logger.warning("No CAPTCHA API key configured")
                return None
            
            if self.captcha_service == '2captcha':
                return await self._solve_2captcha(captcha_type, site_key, page_url)
            elif self.captcha_service == 'anticaptcha':
                return await self._solve_anticaptcha(captcha_type, site_key, page_url)
            else:
                self.logger.error(f"Unsupported CAPTCHA service: {self.captcha_service}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error solving CAPTCHA: {str(e)}")
            return None
    
    async def _solve_2captcha(self, captcha_type: str, site_key: str, page_url: str) -> Optional[str]:
        """Solve CAPTCHA using 2captcha service"""
        try:
            async with aiohttp.ClientSession() as session:
                # Submit CAPTCHA
                submit_data = {
                    'key': self.captcha_api_key,
                    'method': 'userrecaptcha',
                    'googlekey': site_key,
                    'pageurl': page_url,
                    'json': 1
                }
                
                async with session.post('http://2captcha.com/in.php', data=submit_data) as response:
                    result = await response.json()
                    
                    if result['status'] != 1:
                        self.logger.error(f"2captcha submission failed: {result}")
                        return None
                    
                    captcha_id = result['request']
                
                # Wait for solution
                for _ in range(30):  # Wait up to 5 minutes
                    await asyncio.sleep(10)
                    
                    async with session.get(
                        f'http://2captcha.com/res.php?key={self.captcha_api_key}&action=get&id={captcha_id}&json=1'
                    ) as response:
                        result = await response.json()
                        
                        if result['status'] == 1:
                            self.logger.info("CAPTCHA solved successfully")
                            return result['request']
                        elif result['request'] != 'CAPCHA_NOT_READY':
                            self.logger.error(f"CAPTCHA solving failed: {result}")
                            return None
                
                self.logger.error("CAPTCHA solving timeout")
                return None
                
        except Exception as e:
            self.logger.error(f"Error with 2captcha service: {str(e)}")
            return None
    
    async def _solve_anticaptcha(self, captcha_type: str, site_key: str, page_url: str) -> Optional[str]:
        """Solve CAPTCHA using AntiCaptcha service"""
        # Implementation for AntiCaptcha service
        # Similar to 2captcha but with different API
        pass
    
    async def navigate_with_privacy(self, url: str, wait_time: int = 5) -> bool:
        """Navigate to URL with privacy measures"""
        try:
            if not self.driver:
                self.logger.error("Browser not initialized")
                return False
            
            # Random delay to avoid timing analysis
            await asyncio.sleep(random.uniform(1, 3))
            
            # Navigate to URL
            self.driver.get(url)
            
            # Wait for page load
            WebDriverWait(self.driver, wait_time).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            
            # Inject privacy scripts after navigation
            self._inject_anti_fingerprinting_scripts()
            
            self.logger.info(f"Navigated to: {url}")
            return True
            
        except TimeoutException:
            self.logger.error(f"Timeout navigating to: {url}")
            return False
        except Exception as e:
            self.logger.error(f"Error navigating to {url}: {str(e)}")
            return False
    
    async def create_account_with_privacy(self, 
                                        site_url: str,
                                        identity_id: str,
                                        account_data: Dict[str, str]) -> Dict[str, Any]:
        """Create account with full privacy measures and credential storage"""
        try:
            # Navigate to registration page
            if not await self.navigate_with_privacy(site_url):
                return {'success': False, 'error': 'Failed to navigate to site'}
            
            # Fill registration form
            success = await self._fill_registration_form(account_data)
            if not success:
                return {'success': False, 'error': 'Failed to fill registration form'}
            
            # Handle CAPTCHA if present
            captcha_solved = await self._handle_captcha_if_present()
            if not captcha_solved:
                return {'success': False, 'error': 'Failed to solve CAPTCHA'}
            
            # Submit form
            submit_success = await self._submit_registration_form()
            if not submit_success:
                return {'success': False, 'error': 'Failed to submit registration'}
            
            # Store credentials in Vaultwarden and setup 2FA
            service_name = self._extract_service_name(site_url)
            credential_result = await self.privacy_manager.create_account_with_credentials(
                service_name=service_name,
                identity=self.privacy_manager.identities.get(identity_id),
                service_url=site_url,
                enable_2fa=True
            )
            
            self.logger.info(f"Successfully created account for {service_name}")
            
            return {
                'success': True,
                'service': service_name,
                'credentials_stored': credential_result.get('vaultwarden_stored', False),
                '2fa_enabled': credential_result.get('2fa_enabled', False),
                'identity_id': identity_id
            }
            
        except Exception as e:
            self.logger.error(f"Error creating account with privacy: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _fill_registration_form(self, account_data: Dict[str, str]) -> bool:
        """Fill registration form with account data"""
        try:
            # Common field mappings
            field_mappings = {
                'email': ['email', 'Email', 'user_email', 'login'],
                'password': ['password', 'Password', 'passwd', 'pass'],
                'confirm_password': ['confirm_password', 'password_confirmation', 'confirm_pass'],
                'first_name': ['first_name', 'firstName', 'fname'],
                'last_name': ['last_name', 'lastName', 'lname'],
                'username': ['username', 'user_name', 'login', 'handle']
            }
            
            for data_key, field_names in field_mappings.items():
                if data_key in account_data:
                    for field_name in field_names:
                        try:
                            element = self.driver.find_element(By.NAME, field_name)
                            element.clear()
                            element.send_keys(account_data[data_key])
                            break
                        except:
                            try:
                                element = self.driver.find_element(By.ID, field_name)
                                element.clear()
                                element.send_keys(account_data[data_key])
                                break
                            except:
                                continue
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error filling registration form: {str(e)}")
            return False
    
    async def _handle_captcha_if_present(self) -> bool:
        """Handle CAPTCHA if present on the page"""
        try:
            # Look for common CAPTCHA elements
            captcha_selectors = [
                '.g-recaptcha',
                '#recaptcha',
                '.h-captcha',
                '.captcha'
            ]
            
            for selector in captcha_selectors:
                try:
                    captcha_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    
                    # Extract site key
                    site_key = captcha_element.get_attribute('data-sitekey')
                    if site_key:
                        # Solve CAPTCHA
                        solution = await self.solve_captcha('recaptcha', site_key, self.driver.current_url)
                        
                        if solution:
                            # Inject solution
                            self.driver.execute_script(
                                f"document.getElementById('g-recaptcha-response').innerHTML = '{solution}';"
                            )
                            return True
                        
                except:
                    continue
            
            # No CAPTCHA found
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling CAPTCHA: {str(e)}")
            return False
    
    async def _submit_registration_form(self) -> bool:
        """Submit registration form"""
        try:
            # Look for submit button
            submit_selectors = [
                'input[type="submit"]',
                'button[type="submit"]',
                '.submit-btn',
                '.register-btn',
                '.signup-btn'
            ]
            
            for selector in submit_selectors:
                try:
                    submit_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                    submit_button.click()
                    
                    # Wait for response
                    await asyncio.sleep(3)
                    return True
                    
                except:
                    continue
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error submitting registration form: {str(e)}")
            return False
    
    def _extract_service_name(self, url: str) -> str:
        """Extract service name from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www. and common subdomains
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Extract main domain name
            parts = domain.split('.')
            if len(parts) >= 2:
                return parts[-2].capitalize()
            
            return domain.capitalize()
            
        except Exception:
            return "Unknown Service"
    
    async def cleanup(self):
        """Cleanup browser resources"""
        try:
            if self.driver:
                self.driver.quit()
                self.driver = None
            
            if self.privacy_manager:
                await self.privacy_manager.cleanup()
            
            self.logger.info("Browser cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

# Integration functions for K.E.N.
async def ken_create_anonymous_browser(headless: bool = True) -> Optional[TorBrowserAgent]:
    """Create anonymous Tor browser for K.E.N. operations"""
    agent = TorBrowserAgent()
    
    if not await agent.setup_tor_connection():
        return None
    
    if agent.create_tor_browser(headless=headless):
        return agent
    
    return None

async def ken_automated_account_creation(site_url: str, identity_id: str) -> Dict[str, Any]:
    """Automated account creation with privacy and credential storage"""
    agent = TorBrowserAgent()
    
    try:
        if not await agent.setup_tor_connection():
            return {'success': False, 'error': 'Tor connection failed'}
        
        if not agent.create_tor_browser(headless=True):
            return {'success': False, 'error': 'Browser creation failed'}
        
        # Get identity data
        identity = agent.privacy_manager.identities.get(identity_id)
        if not identity:
            return {'success': False, 'error': 'Identity not found'}
        
        account_data = {
            'email': identity.email,
            'password': agent.privacy_manager.generate_secure_password(),
            'first_name': identity.name.split()[0],
            'last_name': identity.name.split()[-1] if len(identity.name.split()) > 1 else '',
        }
        
        result = await agent.create_account_with_privacy(site_url, identity_id, account_data)
        
        await agent.cleanup()
        return result
        
    except Exception as e:
        await agent.cleanup()
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    # Example usage
    async def main():
        agent = TorBrowserAgent()
        
        # Setup Tor connection
        if await agent.setup_tor_connection():
            print("Tor connection established")
            
            # Create browser
            if agent.create_tor_browser(headless=False):
                print("Tor browser created")
                
                # Test navigation
                await agent.navigate_with_privacy("https://check.torproject.org")
                
                # Wait for user to see result
                await asyncio.sleep(10)
                
                await agent.cleanup()
        else:
            print("Tor connection failed - running in simulation mode")
    
    asyncio.run(main())

