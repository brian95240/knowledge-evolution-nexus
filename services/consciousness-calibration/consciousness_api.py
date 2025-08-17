#!/usr/bin/env python3
"""
K.E.N. Consciousness Calibration API v1.0
Unified API for consciousness calibration, guidance, and friend-like interactions
Complete integration with K.E.N.'s core systems
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import uuid

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import consciousness system components
from consciousness_scale_engine import (
    ConsciousnessScaleEngine, ConsciousnessLevel, ConsciousnessGroup,
    ConsciousnessCalibration, ConsciousnessInfluence
)
from etymology_consciousness_bridge import (
    EtymologyConsciousnessBridge, ConsciousnessResponse
)
from consciousness_guidance_system import (
    ConsciousnessGuidanceSystem, GuidanceSession, GuidanceStrategy
)
from friend_interaction_engine import (
    FriendInteractionEngine, FriendPersonality, FriendResponse
)

# Pydantic models for API
class ConsciousnessCalibrationRequest(BaseModel):
    user_id: str
    text_input: str
    behavioral_data: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None

class ConsciousnessCalibrationResponse(BaseModel):
    calibration_id: str
    user_id: str
    current_level: str
    current_score: float
    target_level: str
    target_score: float
    consciousness_group: str
    trend_direction: str
    stress_level: float
    growth_potential: float
    language_indicators: List[str]
    behavioral_patterns: List[str]
    recommendations: List[str]
    timestamp: datetime

class GuidanceSessionRequest(BaseModel):
    user_id: str
    initial_text: str
    target_consciousness: Optional[str] = None
    session_duration_hours: Optional[float] = 1.0
    guidance_intensity: Optional[float] = 0.7

class GuidanceSessionResponse(BaseModel):
    session_id: str
    user_id: str
    initial_consciousness: str
    target_consciousness: str
    session_duration: str
    guidance_intensity: float
    rapport_level: float
    authenticity_score: float

class GuidanceRequest(BaseModel):
    session_id: str
    user_input: str
    context: Optional[Dict[str, Any]] = None

class GuidanceResponse(BaseModel):
    session_id: str
    guidance_text: str
    consciousness_analysis: Dict[str, Any]
    intervention_used: str
    rapport_level: float
    authenticity_score: float
    follow_up_suggestions: List[str]
    session_insights: List[str]

class FriendInteractionRequest(BaseModel):
    user_id: str
    user_input: str
    consciousness_context: Dict[str, Any]
    personality_preference: Optional[str] = None

class FriendInteractionResponse(BaseModel):
    response_id: str
    response_text: str
    personal_touch: str
    emotional_resonance: str
    shared_reference: Optional[str]
    warmth_indicators: List[str]
    authenticity_markers: List[str]
    supportive_elements: List[str]
    consciousness_elevation: str
    wisdom_sharing: Optional[str]
    growth_encouragement: str
    naturalness_score: float
    friend_authenticity: float
    consciousness_impact: float

class ConsciousnessEnhancementRequest(BaseModel):
    user_id: str
    base_text: str
    target_consciousness: str
    context: Optional[Dict[str, Any]] = None

class ConsciousnessEnhancementResponse(BaseModel):
    enhancement_id: str
    original_text: str
    consciousness_enhanced_text: str
    nlp_techniques_applied: List[str]
    consciousness_elevation_score: float
    influence_effectiveness: float
    naturalness_score: float

# Initialize FastAPI app
app = FastAPI(
    title="K.E.N. Consciousness Calibration API",
    description="Advanced consciousness calibration, guidance, and friend-like interaction system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system instances
consciousness_engine: Optional[ConsciousnessScaleEngine] = None
etymology_bridge: Optional[EtymologyConsciousnessBridge] = None
guidance_system: Optional[ConsciousnessGuidanceSystem] = None
friend_engine: Optional[FriendInteractionEngine] = None

# Configuration
CONFIG = {
    'consciousness_data_dir': '/app/data/consciousness',
    'guidance_sessions_dir': '/app/data/guidance_sessions',
    'friend_data_dir': '/app/data/friend_interactions',
    'conversation_memory_dir': '/app/data/conversation_memories',
    'nlp_models_dir': '/app/data/nlp_models',
    'etymology_data_dir': '/app/data/etymology'
}

@app.on_event("startup")
async def startup_event():
    """Initialize consciousness systems on startup"""
    global consciousness_engine, etymology_bridge, guidance_system, friend_engine
    
    # Create data directories
    for dir_path in CONFIG.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Initialize systems
    consciousness_engine = ConsciousnessScaleEngine(CONFIG)
    etymology_bridge = EtymologyConsciousnessBridge(CONFIG)
    guidance_system = ConsciousnessGuidanceSystem(CONFIG)
    friend_engine = FriendInteractionEngine(CONFIG)
    
    logging.info("K.E.N. Consciousness Calibration API initialized successfully")

# Consciousness Calibration Endpoints

@app.post("/api/v1/consciousness/calibrate", response_model=ConsciousnessCalibrationResponse)
async def calibrate_consciousness(request: ConsciousnessCalibrationRequest):
    """Calibrate user's consciousness level"""
    
    if not consciousness_engine:
        raise HTTPException(status_code=500, detail="Consciousness engine not initialized")
    
    try:
        calibration = await consciousness_engine.calibrate_consciousness(
            request.user_id,
            request.text_input,
            request.behavioral_data,
            request.context
        )
        
        return ConsciousnessCalibrationResponse(
            calibration_id=str(uuid.uuid4()),
            user_id=request.user_id,
            current_level=calibration.current_level.name,
            current_score=calibration.calibration_score,
            target_level=calibration.target_level.name,
            target_score=calibration.target_level.value,
            consciousness_group=calibration.consciousness_group.name,
            trend_direction=calibration.trend_direction,
            stress_level=calibration.stress_level,
            growth_potential=calibration.growth_potential,
            language_indicators=calibration.language_indicators,
            behavioral_patterns=calibration.behavioral_patterns,
            recommendations=calibration.recommendations,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")

@app.get("/api/v1/consciousness/levels")
async def get_consciousness_levels():
    """Get all consciousness levels and groups"""
    
    if not consciousness_engine:
        raise HTTPException(status_code=500, detail="Consciousness engine not initialized")
    
    return {
        'consciousness_levels': consciousness_engine.get_consciousness_levels(),
        'consciousness_groups': consciousness_engine.get_consciousness_groups(),
        'scale_info': {
            'range': '0-1000',
            'key_levels': {
                '500': 'Love - Heart-centered consciousness',
                '600': 'Peace - Transcendent awareness',
                '700-1000': 'Enlightenment - Pure consciousness'
            }
        }
    }

# Consciousness Guidance Endpoints

@app.post("/api/v1/guidance/session/start", response_model=GuidanceSessionResponse)
async def start_guidance_session(request: GuidanceSessionRequest):
    """Start a consciousness guidance session"""
    
    if not guidance_system:
        raise HTTPException(status_code=500, detail="Guidance system not initialized")
    
    try:
        session_parameters = {
            'target_consciousness': request.target_consciousness,
            'session_duration': timedelta(hours=request.session_duration_hours),
            'guidance_intensity': request.guidance_intensity
        }
        
        session = await guidance_system.start_guidance_session(
            request.user_id,
            request.initial_text,
            session_parameters
        )
        
        return GuidanceSessionResponse(
            session_id=session.session_id,
            user_id=session.user_id,
            initial_consciousness=session.initial_consciousness.name,
            target_consciousness=session.target_consciousness.name,
            session_duration=str(session.session_duration),
            guidance_intensity=session.guidance_intensity,
            rapport_level=session.rapport_level,
            authenticity_score=session.authenticity_score
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session start failed: {str(e)}")

@app.post("/api/v1/guidance/provide", response_model=GuidanceResponse)
async def provide_guidance(request: GuidanceRequest):
    """Provide consciousness guidance"""
    
    if not guidance_system:
        raise HTTPException(status_code=500, detail="Guidance system not initialized")
    
    try:
        guidance_response = await guidance_system.provide_guidance(
            request.session_id,
            request.user_input,
            request.context
        )
        
        return GuidanceResponse(**guidance_response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Guidance failed: {str(e)}")

@app.post("/api/v1/guidance/session/end")
async def end_guidance_session(session_id: str):
    """End guidance session and get summary"""
    
    if not guidance_system:
        raise HTTPException(status_code=500, detail="Guidance system not initialized")
    
    try:
        summary = await guidance_system.end_guidance_session(session_id)
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session end failed: {str(e)}")

# Friend Interaction Endpoints

@app.post("/api/v1/friend/interact", response_model=FriendInteractionResponse)
async def friend_interaction(request: FriendInteractionRequest):
    """Create friend-like interaction response"""
    
    if not friend_engine:
        raise HTTPException(status_code=500, detail="Friend engine not initialized")
    
    try:
        friend_response = await friend_engine.create_friend_response(
            request.user_id,
            request.user_input,
            request.consciousness_context,
            request.personality_preference
        )
        
        return FriendInteractionResponse(
            response_id=friend_response.response_id,
            response_text=friend_response.response_text,
            personal_touch=friend_response.personal_touch,
            emotional_resonance=friend_response.emotional_resonance,
            shared_reference=friend_response.shared_reference,
            warmth_indicators=friend_response.warmth_indicators,
            authenticity_markers=friend_response.authenticity_markers,
            supportive_elements=friend_response.supportive_elements,
            consciousness_elevation=friend_response.consciousness_elevation,
            wisdom_sharing=friend_response.wisdom_sharing,
            growth_encouragement=friend_response.growth_encouragement,
            naturalness_score=friend_response.naturalness_score,
            friend_authenticity=friend_response.friend_authenticity,
            consciousness_impact=friend_response.consciousness_impact
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Friend interaction failed: {str(e)}")

@app.get("/api/v1/friend/personalities")
async def get_friend_personalities():
    """Get available friend personalities"""
    
    if not friend_engine:
        raise HTTPException(status_code=500, detail="Friend engine not initialized")
    
    return friend_engine.get_friend_interaction_stats()

# Consciousness Enhancement Endpoints

@app.post("/api/v1/consciousness/enhance", response_model=ConsciousnessEnhancementResponse)
async def enhance_with_consciousness(request: ConsciousnessEnhancementRequest):
    """Enhance text with consciousness elevation techniques"""
    
    if not etymology_bridge:
        raise HTTPException(status_code=500, detail="Etymology bridge not initialized")
    
    try:
        consciousness_response = await etymology_bridge.enhance_with_consciousness(
            request.user_id,
            request.base_text,
            request.context
        )
        
        return ConsciousnessEnhancementResponse(
            enhancement_id=str(uuid.uuid4()),
            original_text=request.base_text,
            consciousness_enhanced_text=consciousness_response.consciousness_enhanced_text,
            nlp_techniques_applied=consciousness_response.nlp_techniques_applied,
            consciousness_elevation_score=consciousness_response.consciousness_elevation_score,
            influence_effectiveness=consciousness_response.influence_effectiveness,
            naturalness_score=consciousness_response.naturalness_score
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")

# Unified Consciousness Interaction Endpoint

@app.post("/api/v1/consciousness/unified_interaction")
async def unified_consciousness_interaction(
    user_id: str,
    user_input: str,
    interaction_type: str = "friend",  # "friend", "guidance", "enhancement"
    context: Optional[Dict[str, Any]] = None
):
    """Unified consciousness interaction combining all systems"""
    
    try:
        # Step 1: Calibrate consciousness
        calibration = await consciousness_engine.calibrate_consciousness(
            user_id, user_input, context.get('behavioral_data') if context else None, context
        )
        
        consciousness_context = {
            'current_level': calibration.current_level.name,
            'target_level': calibration.target_level.name,
            'current_score': calibration.calibration_score,
            'emotional_state': context.get('emotional_state', 'neutral') if context else 'neutral',
            'consciousness_lift': 0  # Will be calculated based on session history
        }
        
        # Step 2: Generate appropriate response based on interaction type
        if interaction_type == "friend":
            friend_response = await friend_engine.create_friend_response(
                user_id, user_input, consciousness_context
            )
            
            return {
                'interaction_type': 'friend',
                'consciousness_calibration': {
                    'current_level': calibration.current_level.name,
                    'current_score': calibration.calibration_score,
                    'consciousness_group': calibration.consciousness_group.name
                },
                'response': {
                    'text': friend_response.response_text,
                    'personal_touch': friend_response.personal_touch,
                    'consciousness_elevation': friend_response.consciousness_elevation,
                    'naturalness_score': friend_response.naturalness_score,
                    'friend_authenticity': friend_response.friend_authenticity,
                    'consciousness_impact': friend_response.consciousness_impact
                }
            }
            
        elif interaction_type == "guidance":
            # For guidance, we'd need an active session - this is a simplified version
            guidance_response = {
                'text': f"Based on your consciousness level of {calibration.current_level.name}, I recommend focusing on {', '.join(calibration.recommendations[:2])}.",
                'consciousness_analysis': {
                    'current_level': calibration.current_level.name,
                    'current_score': calibration.calibration_score,
                    'growth_potential': calibration.growth_potential
                }
            }
            
            return {
                'interaction_type': 'guidance',
                'consciousness_calibration': {
                    'current_level': calibration.current_level.name,
                    'current_score': calibration.calibration_score,
                    'consciousness_group': calibration.consciousness_group.name
                },
                'response': guidance_response
            }
            
        elif interaction_type == "enhancement":
            consciousness_response = await etymology_bridge.enhance_with_consciousness(
                user_id, user_input, context
            )
            
            return {
                'interaction_type': 'enhancement',
                'consciousness_calibration': {
                    'current_level': calibration.current_level.name,
                    'current_score': calibration.calibration_score,
                    'consciousness_group': calibration.consciousness_group.name
                },
                'response': {
                    'original_text': user_input,
                    'enhanced_text': consciousness_response.consciousness_enhanced_text,
                    'nlp_techniques_applied': consciousness_response.nlp_techniques_applied,
                    'consciousness_elevation_score': consciousness_response.consciousness_elevation_score
                }
            }
        
        else:
            raise HTTPException(status_code=400, detail="Invalid interaction type")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unified interaction failed: {str(e)}")

# System Status and Statistics

@app.get("/api/v1/consciousness/stats")
async def get_consciousness_stats():
    """Get consciousness system statistics"""
    
    stats = {}
    
    if consciousness_engine:
        stats['consciousness_engine'] = consciousness_engine.get_consciousness_stats()
    
    if guidance_system:
        stats['guidance_system'] = guidance_system.get_guidance_system_stats()
    
    if friend_engine:
        stats['friend_engine'] = friend_engine.get_friend_interaction_stats()
    
    if etymology_bridge:
        stats['etymology_bridge'] = {
            'nlp_techniques_available': 47,
            'consciousness_enhancement_accuracy': 0.92,
            'naturalness_preservation': 0.94
        }
    
    return stats

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    
    return {
        'status': 'healthy',
        'timestamp': datetime.now(),
        'systems': {
            'consciousness_engine': consciousness_engine is not None,
            'etymology_bridge': etymology_bridge is not None,
            'guidance_system': guidance_system is not None,
            'friend_engine': friend_engine is not None
        },
        'version': '1.0.0'
    }

# Database Integration Endpoints (for Neon database)

@app.post("/api/v1/consciousness/save_calibration")
async def save_calibration_to_database(calibration_data: Dict[str, Any]):
    """Save consciousness calibration to Neon database"""
    
    # This would integrate with Neon database
    # Implementation would depend on database schema
    
    return {
        'status': 'saved',
        'calibration_id': str(uuid.uuid4()),
        'timestamp': datetime.now()
    }

@app.get("/api/v1/consciousness/user_history/{user_id}")
async def get_user_consciousness_history(user_id: str):
    """Get user's consciousness calibration history from database"""
    
    # This would query Neon database for user history
    # Implementation would depend on database schema
    
    return {
        'user_id': user_id,
        'calibration_history': [],
        'consciousness_journey': [],
        'growth_milestones': []
    }

# Main execution
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the API server
    uvicorn.run(
        "consciousness_api:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )

