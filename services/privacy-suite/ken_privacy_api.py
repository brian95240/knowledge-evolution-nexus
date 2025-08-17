#!/usr/bin/env python3
"""
K.E.N. Privacy Suite - Main API Integration
Unified API for all privacy and anonymity operations
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

# Import all privacy suite components
from core.ken_privacy_manager import KENPrivacyManager, GeneratedIdentity
from secure_os.qubes_manager import QubesVMManager, ken_create_privacy_vm
from secure_os.tails_manager import TailsEnvironmentManager, ken_create_tails_environment
from browser_automation.tor_browser_agent import TorBrowserAgent, ken_create_anonymous_browser
from identity_generation.identity_generator import IdentityGenerator, ken_generate_identity
from account_creation.agentic_account_creator import AgenticAccountCreator, ken_create_account_automated

# Pydantic models for API
class IdentityRequest(BaseModel):
    locale: str = 'en_US'
    country: str = 'US'
    enable_protonmail: bool = True
    enable_privacy_card: bool = True
    enable_ai_passport: bool = True

class AccountCreationRequest(BaseModel):
    platform: str
    identity_id: Optional[str] = None
    use_existing_identity: bool = True

class BulkAccountCreationRequest(BaseModel):
    platforms: List[str]
    identity_count: int = 1
    use_same_identity: bool = False

class PrivacyVMRequest(BaseModel):
    purpose: str = "privacy"
    use_tor: bool = True
    vm_type: str = "qubes"  # "qubes" or "tails"

class BrowserRequest(BaseModel):
    headless: bool = True
    use_tor: bool = True
    anti_fingerprinting: bool = True

# FastAPI app
app = FastAPI(
    title="K.E.N. Privacy Suite API",
    description="Complete privacy and anonymity operations for K.E.N. v3.0",
    version="1.0.0"
)

# Global managers
privacy_manager = None
identity_generator = None
account_creator = None

@app.on_event("startup")
async def startup_event():
    """Initialize privacy suite components"""
    global privacy_manager, identity_generator, account_creator
    
    config = {
        'vaultwarden_url': 'http://localhost:80',
        '2fauth_url': 'http://localhost:8000',
        'captcha_service': '2captcha',
        'prefer_google_signin': True
    }
    
    privacy_manager = KENPrivacyManager(config)
    identity_generator = IdentityGenerator(config)
    account_creator = AgenticAccountCreator(config)
    
    logging.info("K.E.N. Privacy Suite API initialized")

# Identity Management Endpoints
@app.post("/api/v1/identity/generate")
async def generate_identity(request: IdentityRequest):
    """Generate complete identity with all services"""
    try:
        identity = await identity_generator.generate_complete_identity(
            locale=request.locale,
            country=request.country,
            enable_protonmail=request.enable_protonmail,
            enable_privacy_card=request.enable_privacy_card,
            enable_ai_passport=request.enable_ai_passport
        )
        
        if identity:
            return {
                "success": True,
                "identity_id": identity.identity_id,
                "name": identity.name,
                "email": identity.email,
                "phone": identity.phone[:8] + "****",  # Masked
                "services": {
                    "protonmail": identity.protonmail_account is not None,
                    "privacy_card": identity.privacy_card_id is not None,
                    "ai_passport": identity.biometric_profile_id is not None
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to generate identity")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/identity/{identity_id}")
async def get_identity(identity_id: str):
    """Get identity information"""
    try:
        status = await identity_generator.get_identity_status(identity_id)
        
        if status.get('exists'):
            return status
        else:
            raise HTTPException(status_code=404, detail="Identity not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/identity/list")
async def list_identities():
    """List all identities"""
    try:
        identities = []
        for identity_id, identity in privacy_manager.identities.items():
            identities.append({
                "identity_id": identity_id,
                "name": identity.name,
                "email": identity.email,
                "created_at": identity.created_at,
                "accounts_count": len(identity.passwords)
            })
        
        return {"identities": identities}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Account Creation Endpoints
@app.post("/api/v1/account/create")
async def create_account(request: AccountCreationRequest):
    """Create account on platform with privacy"""
    try:
        result = await ken_create_account_automated(
            platform=request.platform,
            identity_id=request.identity_id
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/account/bulk-create")
async def bulk_create_accounts(request: BulkAccountCreationRequest):
    """Create accounts on multiple platforms"""
    try:
        results = await account_creator.create_accounts_bulk(
            platforms=request.platforms,
            identity_count=request.identity_count,
            use_same_identity=request.use_same_identity
        )
        
        return {"results": [result.__dict__ for result in results]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/account/statistics")
async def get_account_statistics():
    """Get account creation statistics"""
    try:
        stats = await account_creator.get_creation_statistics()
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Privacy VM Endpoints
@app.post("/api/v1/vm/create")
async def create_privacy_vm(request: PrivacyVMRequest):
    """Create privacy VM (Qubes or Tails)"""
    try:
        if request.vm_type.lower() == "qubes":
            vm_name = await ken_create_privacy_vm(
                purpose=request.purpose,
                use_tor=request.use_tor
            )
        elif request.vm_type.lower() == "tails":
            vm_name = await ken_create_tails_environment(
                purpose=request.purpose,
                enable_persistence=True
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid VM type")
        
        if vm_name:
            return {
                "success": True,
                "vm_id": vm_name,
                "vm_type": request.vm_type,
                "purpose": request.purpose,
                "tor_enabled": request.use_tor
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create VM")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Browser Automation Endpoints
@app.post("/api/v1/browser/create")
async def create_anonymous_browser(request: BrowserRequest):
    """Create anonymous Tor browser"""
    try:
        browser_agent = await ken_create_anonymous_browser(headless=request.headless)
        
        if browser_agent:
            return {
                "success": True,
                "browser_id": id(browser_agent),  # Use object ID as browser ID
                "tor_enabled": request.use_tor,
                "headless": request.headless,
                "anti_fingerprinting": request.anti_fingerprinting
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create browser")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Credential Management Endpoints
@app.get("/api/v1/credentials/{service_name}")
async def get_credentials(service_name: str, identity_id: Optional[str] = None):
    """Get credentials for service"""
    try:
        credentials = await privacy_manager.get_credential_for_service(service_name, identity_id)
        
        if credentials:
            # Return masked credentials for security
            return {
                "service": service_name,
                "username": credentials.get('username', ''),
                "password_stored": bool(credentials.get('password')),
                "2fa_enabled": credentials.get('2fa_enabled', False)
            }
        else:
            raise HTTPException(status_code=404, detail="Credentials not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2FA Management Endpoints
@app.post("/api/v1/2fa/generate/{service_name}")
async def generate_2fa_code(service_name: str):
    """Generate 2FA code for service"""
    try:
        code = await privacy_manager.autonomous_2fa_handler(service_name)
        
        if code:
            return {
                "service": service_name,
                "code": code,
                "generated_at": asyncio.get_event_loop().time()
            }
        else:
            raise HTTPException(status_code=404, detail="2FA not configured for service")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health Check Endpoints
@app.get("/api/v1/health")
async def health_check():
    """Health check for privacy suite"""
    try:
        health_status = {
            "privacy_manager": privacy_manager is not None,
            "identity_generator": identity_generator is not None,
            "account_creator": account_creator is not None,
            "vaultwarden_connection": await privacy_manager.test_vaultwarden_connection() if privacy_manager else False,
            "2fauth_connection": await privacy_manager.test_2fauth_connection() if privacy_manager else False,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        return health_status
        
    except Exception as e:
        return {"error": str(e), "healthy": False}

@app.get("/api/v1/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        # Get component statuses
        qubes_manager = QubesVMManager()
        tails_manager = TailsEnvironmentManager()
        
        status = {
            "privacy_suite_version": "1.0.0",
            "components": {
                "privacy_manager": privacy_manager is not None,
                "identity_generator": identity_generator is not None,
                "account_creator": account_creator is not None,
                "qubes_available": await qubes_manager.check_qubes_availability(),
                "tails_available": await tails_manager.check_virtualization_support()
            },
            "services": {
                "vaultwarden": await privacy_manager.test_vaultwarden_connection() if privacy_manager else False,
                "2fauth": await privacy_manager.test_2fauth_connection() if privacy_manager else False
            },
            "statistics": await account_creator.get_creation_statistics() if account_creator else {},
            "identities_count": len(privacy_manager.identities) if privacy_manager else 0
        }
        
        return status
        
    except Exception as e:
        return {"error": str(e), "status": "error"}

# Main K.E.N. Integration Functions
async def ken_privacy_suite_init(config: Dict[str, Any] = None) -> bool:
    """Initialize K.E.N. Privacy Suite"""
    try:
        global privacy_manager, identity_generator, account_creator
        
        privacy_manager = KENPrivacyManager(config or {})
        identity_generator = IdentityGenerator(config or {})
        account_creator = AgenticAccountCreator(config or {})
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to initialize K.E.N. Privacy Suite: {str(e)}")
        return False

async def ken_create_anonymous_identity_and_accounts(platforms: List[str]) -> Dict[str, Any]:
    """Create anonymous identity and accounts on specified platforms"""
    try:
        # Generate identity
        identity_id = await ken_generate_identity()
        if not identity_id:
            return {"success": False, "error": "Failed to generate identity"}
        
        # Create accounts on platforms
        results = []
        for platform in platforms:
            result = await ken_create_account_automated(platform, identity_id)
            results.append(result)
        
        return {
            "success": True,
            "identity_id": identity_id,
            "account_results": results
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "ken_privacy_api:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )

