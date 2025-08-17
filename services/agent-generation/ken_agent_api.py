#!/usr/bin/env python3
"""
K.E.N. Agent Generation API v1.0
Unified API for K.E.N.'s Autonomous AI Agent Generation System
Complete integration with K.E.N. core systems
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from agent_generation_engine import (
    AgentGenerationEngine, AgentType, ArchetypeSpecialty, IntelligenceLevel
)
from tiny_model_generator import (
    TinyModelGenerator, ModelArchitecture, ModelSpecialty
)
from agent_lifecycle_manager import AgentLifecycleManager
from specialized_archetypes import SpecializedArchetypeSystem, DeploymentMode

# Pydantic models for API
class AgentCreationRequest(BaseModel):
    agent_type: str = Field(..., description="Type of agent (permanent, temporary, phantom, single_use)")
    archetype_specialty: str = Field(..., description="Archetype specialty")
    purpose: str = Field(..., description="Purpose of the agent")
    duration_hours: Optional[int] = Field(None, description="Duration in hours for temporary agents")
    intelligence_level: Optional[str] = Field(None, description="Intelligence level override")
    custom_config: Optional[Dict[str, Any]] = Field(None, description="Custom configuration")

class TaskAssignmentRequest(BaseModel):
    agent_id: str = Field(..., description="Agent ID")
    task: Dict[str, Any] = Field(..., description="Task details")

class SpecializedAgentRequest(BaseModel):
    template_id: str = Field(..., description="Archetype template ID")
    agent_type: str = Field(..., description="Type of agent")
    purpose: str = Field(..., description="Purpose of the agent")
    duration_hours: Optional[int] = Field(None, description="Duration in hours")
    custom_config: Optional[Dict[str, Any]] = Field(None, description="Custom configuration")

class CrisisResponseRequest(BaseModel):
    crisis_type: str = Field(..., description="Type of crisis")
    severity: str = Field(..., description="Crisis severity (low, medium, high, critical)")
    description: Optional[str] = Field(None, description="Crisis description")

class LegalOptimizationRequest(BaseModel):
    revenue_threshold: float = Field(..., description="Revenue threshold for optimization")
    current_structure: Optional[str] = Field(None, description="Current legal structure")
    target_jurisdictions: Optional[List[str]] = Field(None, description="Target jurisdictions")

class CompetitiveIntelRequest(BaseModel):
    target_domains: List[str] = Field(..., description="Target domains for intelligence")
    monitoring_duration_days: Optional[int] = Field(30, description="Monitoring duration in days")
    intelligence_depth: Optional[str] = Field("standard", description="Intelligence depth (basic, standard, deep)")

class ModelGenerationRequest(BaseModel):
    architecture: str = Field(..., description="Model architecture")
    specialty: str = Field(..., description="Model specialty")
    custom_config: Optional[Dict[str, Any]] = Field(None, description="Custom model configuration")

class KENAgentAPI:
    """
    K.E.N. Agent Generation API
    Unified interface for all agent generation and management operations
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("KENAgentAPI")
        
        # Initialize core components
        self.agent_engine = AgentGenerationEngine(config.get('agent_engine', {}))
        self.model_generator = TinyModelGenerator(config.get('model_generator', {}))
        self.lifecycle_manager = AgentLifecycleManager(
            config.get('lifecycle_manager', {}), self.agent_engine
        )
        self.archetype_system = SpecializedArchetypeSystem(
            config.get('archetype_system', {}), self.agent_engine, self.model_generator
        )
        
        # FastAPI app
        self.app = FastAPI(
            title="K.E.N. Agent Generation API",
            description="Autonomous AI Agent Generation System with Elite Intelligence",
            version="1.0.0"
        )
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        self.logger.info("K.E.N. Agent Generation API initialized")

    def _setup_routes(self):
        """Setup API routes"""
        
        # Health and status endpoints
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "components": {
                    "agent_engine": "operational",
                    "model_generator": "operational",
                    "lifecycle_manager": "operational",
                    "archetype_system": "operational"
                }
            }
        
        @self.app.get("/api/status")
        async def system_status():
            """Get comprehensive system status"""
            return {
                "agent_engine": self.agent_engine.get_system_stats(),
                "model_generator": self.model_generator.get_generation_stats(),
                "lifecycle_manager": self.lifecycle_manager.get_lifecycle_stats(),
                "archetype_system": self.archetype_system.get_deployment_status()
            }
        
        # Agent management endpoints
        @self.app.post("/api/agents/create")
        async def create_agent(request: AgentCreationRequest):
            """Create new autonomous agent"""
            try:
                # Parse enums
                agent_type = AgentType(request.agent_type)
                archetype_specialty = ArchetypeSpecialty(request.archetype_specialty)
                intelligence_level = IntelligenceLevel(request.intelligence_level) if request.intelligence_level else None
                
                # Calculate duration
                duration = timedelta(hours=request.duration_hours) if request.duration_hours else None
                
                # Create agent
                agent = await self.agent_engine.create_agent(
                    agent_type=agent_type,
                    archetype_specialty=archetype_specialty,
                    purpose=request.purpose,
                    duration=duration,
                    intelligence_level=intelligence_level,
                    custom_config=request.custom_config
                )
                
                return {
                    "success": True,
                    "agent_id": agent.agent_id,
                    "agent_type": agent.configuration.agent_type.value,
                    "archetype_specialty": agent.configuration.archetype_specialty.value,
                    "intelligence_level": agent.current_intelligence_level.value,
                    "created_at": agent.created_at.isoformat(),
                    "expires_at": agent.expires_at.isoformat() if agent.expires_at else None
                }
                
            except Exception as e:
                self.logger.error(f"Error creating agent: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/agents")
        async def list_agents():
            """List all active agents"""
            return {
                "agents": self.agent_engine.get_active_agents(),
                "total_count": len(self.agent_engine.active_agents)
            }
        
        @self.app.get("/api/agents/{agent_id}")
        async def get_agent_details(agent_id: str):
            """Get detailed information about specific agent"""
            agent_details = self.agent_engine.get_agent_details(agent_id)
            if not agent_details:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            # Add lifecycle information
            lifecycle_details = self.lifecycle_manager.get_agent_lifecycle_details(agent_id)
            if lifecycle_details:
                agent_details["lifecycle"] = lifecycle_details
            
            return agent_details
        
        @self.app.post("/api/agents/{agent_id}/tasks")
        async def assign_task(agent_id: str, request: TaskAssignmentRequest):
            """Assign task to agent"""
            try:
                success = await self.agent_engine.assign_task(agent_id, request.task)
                if not success:
                    raise HTTPException(status_code=400, detail="Failed to assign task")
                
                return {
                    "success": True,
                    "agent_id": agent_id,
                    "task_assigned": True,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Error assigning task: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/api/agents/{agent_id}")
        async def terminate_agent(agent_id: str, reason: str = "Manual termination"):
            """Terminate specific agent"""
            try:
                success = await self.agent_engine.terminate_agent(agent_id, reason)
                if not success:
                    raise HTTPException(status_code=404, detail="Agent not found")
                
                return {
                    "success": True,
                    "agent_id": agent_id,
                    "terminated": True,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Error terminating agent: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Specialized agent endpoints
        @self.app.get("/api/archetypes/templates")
        async def list_archetype_templates():
            """List available archetype templates"""
            return {
                "templates": self.archetype_system.get_archetype_templates(),
                "total_count": len(self.archetype_system.archetype_templates)
            }
        
        @self.app.post("/api/archetypes/create")
        async def create_specialized_agent(request: SpecializedAgentRequest):
            """Create specialized agent from archetype template"""
            try:
                agent_type = AgentType(request.agent_type)
                duration = timedelta(hours=request.duration_hours) if request.duration_hours else None
                
                agent = await self.archetype_system.create_specialized_agent(
                    template_id=request.template_id,
                    agent_type=agent_type,
                    purpose=request.purpose,
                    duration=duration,
                    custom_config=request.custom_config
                )
                
                return {
                    "success": True,
                    "agent_id": agent.agent_id,
                    "template_id": request.template_id,
                    "agent_type": agent.configuration.agent_type.value,
                    "intelligence_level": agent.current_intelligence_level.value,
                    "created_at": agent.created_at.isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Error creating specialized agent: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/archetypes/{template_id}/performance")
        async def get_archetype_performance(template_id: str):
            """Get performance data for archetype template"""
            performance = self.archetype_system.get_archetype_performance(template_id)
            if not performance:
                raise HTTPException(status_code=404, detail="Archetype template not found")
            
            return performance
        
        # Crisis response endpoints
        @self.app.post("/api/crisis/deploy")
        async def deploy_crisis_response(request: CrisisResponseRequest):
            """Deploy crisis response team"""
            try:
                team = await self.archetype_system.deploy_crisis_response_team(
                    crisis_type=request.crisis_type,
                    severity=request.severity
                )
                
                return {
                    "success": True,
                    "crisis_type": request.crisis_type,
                    "severity": request.severity,
                    "team_deployed": True,
                    "agents": [
                        {
                            "agent_id": agent.agent_id,
                            "archetype": agent.configuration.archetype_specialty.value,
                            "intelligence_level": agent.current_intelligence_level.value
                        }
                        for agent in team
                    ],
                    "team_size": len(team),
                    "deployment_time": datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Error deploying crisis response: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Legal optimization endpoints
        @self.app.post("/api/legal/optimize")
        async def deploy_legal_optimization(request: LegalOptimizationRequest):
            """Deploy legal optimization team"""
            try:
                team = await self.archetype_system.deploy_legal_optimization_team(
                    revenue_threshold=request.revenue_threshold
                )
                
                return {
                    "success": True,
                    "revenue_threshold": request.revenue_threshold,
                    "optimization_team_deployed": True,
                    "agents": [
                        {
                            "agent_id": agent.agent_id,
                            "archetype": agent.configuration.archetype_specialty.value,
                            "intelligence_level": agent.current_intelligence_level.value,
                            "purpose": agent.configuration.purpose
                        }
                        for agent in team
                    ],
                    "team_size": len(team),
                    "deployment_time": datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Error deploying legal optimization: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Competitive intelligence endpoints
        @self.app.post("/api/intelligence/deploy")
        async def deploy_competitive_intelligence(request: CompetitiveIntelRequest):
            """Deploy competitive intelligence network"""
            try:
                network = await self.archetype_system.deploy_competitive_intelligence_network(
                    target_domains=request.target_domains
                )
                
                return {
                    "success": True,
                    "target_domains": request.target_domains,
                    "intelligence_network_deployed": True,
                    "agents": [
                        {
                            "agent_id": agent.agent_id,
                            "archetype": agent.configuration.archetype_specialty.value,
                            "agent_type": agent.configuration.agent_type.value,
                            "intelligence_level": agent.current_intelligence_level.value
                        }
                        for agent in network
                    ],
                    "network_size": len(network),
                    "monitoring_duration_days": request.monitoring_duration_days,
                    "deployment_time": datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Error deploying competitive intelligence: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Model generation endpoints
        @self.app.get("/api/models")
        async def list_models():
            """List available tiny models"""
            return {
                "models": self.model_generator.get_available_models(),
                "generation_stats": self.model_generator.get_generation_stats()
            }
        
        @self.app.post("/api/models/generate")
        async def generate_model(request: ModelGenerationRequest):
            """Generate new tiny model"""
            try:
                architecture = ModelArchitecture(request.architecture)
                specialty = ModelSpecialty(request.specialty)
                
                model = await self.model_generator.generate_tiny_model(
                    architecture=architecture,
                    specialty=specialty,
                    custom_config=request.custom_config
                )
                
                return {
                    "success": True,
                    "model_id": model.model_id,
                    "architecture": model.configuration.architecture.value,
                    "specialty": model.configuration.specialty.value,
                    "parameter_count": model.parameter_count,
                    "model_size_mb": model.model_size_mb,
                    "accuracy_score": model.accuracy_score,
                    "created_at": model.created_at.isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Error generating model: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/models/{model_id}/inference")
        async def model_inference(model_id: str, input_text: str, task_type: Optional[str] = None):
            """Perform inference with tiny model"""
            try:
                result = await self.model_generator.inference(
                    model_id=model_id,
                    input_text=input_text,
                    task_type=task_type
                )
                
                return {
                    "success": True,
                    "model_id": model_id,
                    "input_text": input_text,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Error in model inference: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Integration endpoints
        @self.app.post("/api/integration/2fa/generate")
        async def generate_2fa_code(service_name: str):
            """Generate 2FA code for service"""
            try:
                # Import 2FA integration
                import sys
                sys.path.append('/home/ubuntu/knowledge-evolution-nexus/services/2fauth-integration')
                from ken_2fa_manager import KEN2FAManager
                
                manager = KEN2FAManager()
                result = await manager.autonomous_2fa_handler(service_name)
                
                return {
                    "success": True,
                    "service_name": service_name,
                    "2fa_code": result.get('otp_code'),
                    "expires_in": result.get('expires_in', 30),
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Error generating 2FA code: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/integration/vaultwarden/store")
        async def store_credentials(service_name: str, username: str, password: str, notes: Optional[str] = None):
            """Store credentials in Vaultwarden"""
            try:
                # Import Vaultwarden integration
                import sys
                sys.path.append('/home/ubuntu/knowledge-evolution-nexus/services/2fauth-integration')
                from ken_2fa_manager import KEN2FAManager
                
                manager = KEN2FAManager()
                result = await manager.store_credentials_vaultwarden(
                    service_name=service_name,
                    username=username,
                    password=password,
                    notes=notes
                )
                
                return {
                    "success": True,
                    "service_name": service_name,
                    "stored": result.get('success', False),
                    "vault_id": result.get('vault_id'),
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Error storing credentials: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Autonomous operations endpoints
        @self.app.post("/api/autonomous/enable")
        async def enable_autonomous_mode(archetype_specialty: Optional[str] = None):
            """Enable autonomous mode for agents"""
            try:
                # Enable autonomous operations
                enabled_agents = []
                
                for agent_id, agent in self.agent_engine.active_agents.items():
                    if (archetype_specialty is None or 
                        agent.configuration.archetype_specialty.value == archetype_specialty):
                        
                        # Enable autonomous decision making
                        agent.knowledge_base['autonomous_mode'] = True
                        agent.knowledge_base['autonomous_enabled_at'] = datetime.now().isoformat()
                        enabled_agents.append(agent_id)
                
                return {
                    "success": True,
                    "autonomous_mode": "enabled",
                    "affected_agents": enabled_agents,
                    "archetype_filter": archetype_specialty,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Error enabling autonomous mode: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/autonomous/disable")
        async def disable_autonomous_mode(archetype_specialty: Optional[str] = None):
            """Disable autonomous mode for agents"""
            try:
                # Disable autonomous operations
                disabled_agents = []
                
                for agent_id, agent in self.agent_engine.active_agents.items():
                    if (archetype_specialty is None or 
                        agent.configuration.archetype_specialty.value == archetype_specialty):
                        
                        # Disable autonomous decision making
                        agent.knowledge_base['autonomous_mode'] = False
                        agent.knowledge_base['autonomous_disabled_at'] = datetime.now().isoformat()
                        disabled_agents.append(agent_id)
                
                return {
                    "success": True,
                    "autonomous_mode": "disabled",
                    "affected_agents": disabled_agents,
                    "archetype_filter": archetype_specialty,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Error disabling autonomous mode: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Emergency endpoints
        @self.app.post("/api/emergency/shutdown")
        async def emergency_shutdown():
            """Emergency system shutdown"""
            try:
                # Terminate all agents
                terminated_agents = []
                for agent_id in list(self.agent_engine.active_agents.keys()):
                    await self.agent_engine.terminate_agent(agent_id, "Emergency shutdown")
                    terminated_agents.append(agent_id)
                
                # Shutdown lifecycle manager
                self.lifecycle_manager.shutdown()
                
                return {
                    "success": True,
                    "emergency_shutdown": True,
                    "terminated_agents": terminated_agents,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Error in emergency shutdown: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    async def start_server(self, host: str = "0.0.0.0", port: int = 8004):
        """Start the API server"""
        
        self.logger.info(f"Starting K.E.N. Agent Generation API on {host}:{port}")
        
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(config)
        await server.serve()

    def run_server(self, host: str = "0.0.0.0", port: int = 8004):
        """Run the API server (blocking)"""
        
        uvicorn.run(
            app=self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )

# Main execution
async def main():
    """Main execution function"""
    
    # Configuration
    config = {
        'agent_engine': {
            'max_concurrent_agents': 1000,
            'agent_data_dir': '/app/data/agents'
        },
        'model_generator': {
            'models_dir': '/app/data/models',
            'training_data_dir': '/app/data/training'
        },
        'lifecycle_manager': {
            'lifecycle_db_path': '/app/data/lifecycle.db'
        },
        'archetype_system': {}
    }
    
    # Initialize API
    api = KENAgentAPI(config)
    
    # Start server
    await api.start_server(host="0.0.0.0", port=8004)

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run server
    asyncio.run(main())

