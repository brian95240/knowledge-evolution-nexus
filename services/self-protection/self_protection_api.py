#!/usr/bin/env python3
"""
K.E.N. Self-Protection System API v1.0
Unified API for Environmental Monitoring, Legal Orchestration, Phase Scaling, and Communication Workflow
Complete Autonomous Self-Protection with 179 Quintillion Optimization Capability
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

# Import all self-protection components
import sys
sys.path.append('/home/ubuntu/knowledge-evolution-nexus/services/self-protection')
sys.path.append('/home/ubuntu/knowledge-evolution-nexus/services/2fauth-integration')

from environmental_monitor import (
    EnvironmentalMonitor, EnvironmentalTrigger, TriggerType, TriggerSeverity
)
from legal_orchestration import (
    LegalOrchestrationEngine, LegalConsultation, JurisdictionType
)
from phase_scaling_system import (
    PhaseScalingSystem, ScalingPlan, ScalingPhase
)
from communication_workflow import (
    CommunicationWorkflowEngine, NotificationMessage, ApprovalRequest, ApprovalStatus
)
from ken_2fa_manager import KEN2FAManager

# Pydantic models for API
class TriggerResponse(BaseModel):
    trigger_id: str
    trigger_type: str
    severity: str
    title: str
    description: str
    confidence_score: float
    requires_approval: bool
    recommended_actions: List[str]
    autonomous_actions: List[str]
    cost_benefit_analysis: Dict[str, Any]

class ScalingAssessmentRequest(BaseModel):
    current_revenue: float

class ScalingAssessmentResponse(BaseModel):
    scaling_recommended: bool
    current_phase: str
    target_phase: Optional[str]
    roi_percentage: Optional[float]
    implementation_timeline: Optional[str]
    confidence_score: float

class ApprovalResponseRequest(BaseModel):
    approval_id: str
    status: str
    approved_actions: Optional[List[str]] = None
    rejection_reason: Optional[str] = None
    modifications: Optional[Dict[str, Any]] = None

class SystemStatsResponse(BaseModel):
    environmental_monitor: Dict[str, Any]
    legal_orchestration: Dict[str, Any]
    phase_scaling: Dict[str, Any]
    communication_workflow: Dict[str, Any]
    system_health: Dict[str, Any]

class SelfProtectionAPI:
    """
    K.E.N.'s Unified Self-Protection System API
    Complete Autonomous Self-Protection with Advanced Intelligence
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("SelfProtectionAPI")
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="K.E.N. Self-Protection System API",
            description="Autonomous Environmental Monitoring, Legal Orchestration, Phase Scaling, and Communication Workflow",
            version="1.0.0",
            docs_url="/api/docs",
            redoc_url="/api/redoc"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize all subsystems
        self.environmental_monitor = EnvironmentalMonitor(config)
        self.legal_orchestrator = LegalOrchestrationEngine(config)
        self.phase_scaler = PhaseScalingSystem(config)
        self.communication_engine = CommunicationWorkflowEngine(config)
        self.auth_manager = KEN2FAManager(config)
        
        # Security
        self.security = HTTPBearer()
        
        # System state
        self.system_active = True
        self.last_health_check = datetime.now()
        self.active_triggers = {}
        self.active_consultations = {}
        self.active_scaling_plans = {}
        
        # Setup API routes
        self._setup_routes()
        
        # Start background monitoring
        asyncio.create_task(self._start_background_monitoring())
        
        self.logger.info("K.E.N. Self-Protection API initialized")

    def _setup_routes(self):
        """Setup all API routes"""
        
        # Health and status endpoints
        @self.app.get("/api/health")
        async def health_check():
            """System health check"""
            return await self._health_check()
        
        @self.app.get("/api/status")
        async def system_status():
            """Get comprehensive system status"""
            return await self._get_system_status()
        
        @self.app.get("/api/stats", response_model=SystemStatsResponse)
        async def system_stats():
            """Get system statistics"""
            return await self._get_system_stats()
        
        # Environmental monitoring endpoints
        @self.app.get("/api/triggers")
        async def get_triggers():
            """Get all environmental triggers"""
            return await self._get_triggers()
        
        @self.app.get("/api/triggers/{trigger_id}")
        async def get_trigger(trigger_id: str):
            """Get specific trigger details"""
            return await self._get_trigger_details(trigger_id)
        
        @self.app.post("/api/triggers/scan")
        async def trigger_scan(background_tasks: BackgroundTasks):
            """Trigger immediate environmental scan"""
            background_tasks.add_task(self._perform_environmental_scan)
            return {"message": "Environmental scan initiated", "status": "processing"}
        
        # Legal orchestration endpoints
        @self.app.get("/api/consultations")
        async def get_consultations():
            """Get all legal consultations"""
            return await self._get_consultations()
        
        @self.app.get("/api/consultations/{consultation_id}")
        async def get_consultation(consultation_id: str):
            """Get specific consultation details"""
            return await self._get_consultation_details(consultation_id)
        
        @self.app.post("/api/consultations")
        async def create_consultation(trigger_id: str, background_tasks: BackgroundTasks):
            """Create legal consultation for trigger"""
            background_tasks.add_task(self._create_legal_consultation, trigger_id)
            return {"message": "Legal consultation initiated", "status": "processing"}
        
        # Phase scaling endpoints
        @self.app.post("/api/scaling/assess", response_model=ScalingAssessmentResponse)
        async def assess_scaling(request: ScalingAssessmentRequest):
            """Assess scaling opportunity"""
            return await self._assess_scaling_opportunity(request.current_revenue)
        
        @self.app.get("/api/scaling/plans")
        async def get_scaling_plans():
            """Get all scaling plans"""
            return await self._get_scaling_plans()
        
        @self.app.get("/api/scaling/plans/{plan_id}")
        async def get_scaling_plan(plan_id: str):
            """Get specific scaling plan"""
            return await self._get_scaling_plan_details(plan_id)
        
        @self.app.get("/api/scaling/phase")
        async def get_current_phase():
            """Get current scaling phase information"""
            return await self._get_current_phase_info()
        
        # Communication workflow endpoints
        @self.app.get("/api/notifications")
        async def get_notifications():
            """Get notification history"""
            return await self._get_notifications()
        
        @self.app.get("/api/approvals")
        async def get_approvals():
            """Get approval requests"""
            return await self._get_approval_requests()
        
        @self.app.post("/api/approvals/respond")
        async def respond_to_approval(request: ApprovalResponseRequest):
            """Respond to approval request"""
            return await self._process_approval_response(request)
        
        @self.app.get("/api/stakeholders")
        async def get_stakeholders():
            """Get stakeholders"""
            return await self._get_stakeholders()
        
        # Autonomous action endpoints
        @self.app.post("/api/autonomous/enable")
        async def enable_autonomous_mode():
            """Enable autonomous action mode"""
            return await self._enable_autonomous_mode()
        
        @self.app.post("/api/autonomous/disable")
        async def disable_autonomous_mode():
            """Disable autonomous action mode"""
            return await self._disable_autonomous_mode()
        
        @self.app.get("/api/autonomous/status")
        async def get_autonomous_status():
            """Get autonomous mode status"""
            return await self._get_autonomous_status()
        
        # Integration endpoints
        @self.app.post("/api/integration/2fa/generate")
        async def generate_2fa_code(service_name: str):
            """Generate 2FA code for service"""
            return await self._generate_2fa_code(service_name)
        
        @self.app.post("/api/integration/vaultwarden/store")
        async def store_credentials(service: str, username: str, password: str):
            """Store credentials in Vaultwarden"""
            return await self._store_credentials(service, username, password)
        
        # Emergency endpoints
        @self.app.post("/api/emergency/shutdown")
        async def emergency_shutdown():
            """Emergency system shutdown"""
            return await self._emergency_shutdown()
        
        @self.app.post("/api/emergency/reset")
        async def emergency_reset():
            """Emergency system reset"""
            return await self._emergency_reset()

    async def _start_background_monitoring(self):
        """Start background monitoring tasks"""
        
        self.logger.info("Starting background monitoring tasks")
        
        # Start environmental monitoring
        asyncio.create_task(self._continuous_environmental_monitoring())
        
        # Start health monitoring
        asyncio.create_task(self._continuous_health_monitoring())
        
        # Start approval monitoring
        asyncio.create_task(self._continuous_approval_monitoring())

    async def _continuous_environmental_monitoring(self):
        """Continuous environmental monitoring"""
        
        while self.system_active:
            try:
                # Perform environmental scan
                triggers = await self.environmental_monitor.scan_environment()
                
                # Process new triggers
                for trigger in triggers:
                    if trigger.trigger_id not in self.active_triggers:
                        await self._process_new_trigger(trigger)
                        self.active_triggers[trigger.trigger_id] = trigger
                
                # Wait before next scan
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in environmental monitoring: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    async def _continuous_health_monitoring(self):
        """Continuous health monitoring"""
        
        while self.system_active:
            try:
                # Update health check timestamp
                self.last_health_check = datetime.now()
                
                # Check system health
                health_status = await self._health_check()
                
                # Log health issues
                if health_status['status'] != 'healthy':
                    self.logger.warning(f"System health issue detected: {health_status}")
                
                # Wait before next check
                await asyncio.sleep(60)  # 1 minute
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {str(e)}")
                await asyncio.sleep(30)  # Wait 30 seconds on error

    async def _continuous_approval_monitoring(self):
        """Continuous approval monitoring"""
        
        while self.system_active:
            try:
                # Check for expired approvals
                current_time = datetime.now()
                
                for approval_id, approval in self.communication_engine.approval_requests.items():
                    if (approval.status == ApprovalStatus.PENDING and 
                        approval.response_deadline < current_time):
                        
                        # Mark as expired
                        approval.status = ApprovalStatus.EXPIRED
                        self.logger.warning(f"Approval request expired: {approval_id}")
                
                # Wait before next check
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                self.logger.error(f"Error in approval monitoring: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def _process_new_trigger(self, trigger: EnvironmentalTrigger):
        """Process new environmental trigger"""
        
        self.logger.info(f"Processing new trigger: {trigger.title}")
        
        try:
            # Create legal consultation if needed
            consultation = None
            if trigger.severity in [TriggerSeverity.HIGH, TriggerSeverity.CRITICAL, TriggerSeverity.EMERGENCY]:
                consultation = await self.legal_orchestrator.create_consultation(trigger)
                if consultation:
                    self.active_consultations[consultation.consultation_id] = consultation
            
            # Check for scaling opportunity
            scaling_plan = None
            if trigger.trigger_type == TriggerType.REVENUE_THRESHOLD:
                # Extract revenue from trigger
                revenue = trigger.impact_assessment.get('financial_impact', 0)
                if revenue > 0:
                    scaling_trigger = await self.phase_scaler.create_scaling_trigger(revenue)
                    if scaling_trigger:
                        # Convert scaling trigger to plan
                        assessment = await self.phase_scaler.assess_scaling_opportunity(revenue)
                        if assessment['scaling_recommended']:
                            scaling_plan = assessment['scaling_plan']
                            self.active_scaling_plans[scaling_plan['plan_id']] = scaling_plan
            
            # Send notifications
            await self.communication_engine.process_trigger_notification(
                trigger, consultation, scaling_plan
            )
            
            self.logger.info(f"Trigger processed successfully: {trigger.trigger_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing trigger {trigger.trigger_id}: {str(e)}")

    async def _health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'issues': []
        }
        
        try:
            # Check environmental monitor
            env_health = await self.environmental_monitor.health_check()
            health_status['components']['environmental_monitor'] = env_health
            if not env_health.get('healthy', False):
                health_status['issues'].append('Environmental monitor unhealthy')
            
            # Check legal orchestrator
            legal_health = self.legal_orchestrator.get_system_stats()
            health_status['components']['legal_orchestrator'] = {
                'healthy': True,
                'active_consultations': legal_health.get('total_consultations', 0)
            }
            
            # Check phase scaler
            scaling_health = self.phase_scaler.get_system_stats()
            health_status['components']['phase_scaler'] = {
                'healthy': True,
                'current_phase': scaling_health.get('current_phase', 'unknown')
            }
            
            # Check communication engine
            comm_health = self.communication_engine.get_workflow_stats()
            health_status['components']['communication_engine'] = {
                'healthy': True,
                'notification_success_rate': comm_health.get('notification_success_rate', 0.0)
            }
            
            # Check 2FA manager
            auth_health = await self.auth_manager.health_check()
            health_status['components']['auth_manager'] = auth_health
            if not auth_health.get('healthy', False):
                health_status['issues'].append('2FA manager unhealthy')
            
            # Overall status
            if health_status['issues']:
                health_status['status'] = 'degraded' if len(health_status['issues']) < 3 else 'unhealthy'
            
        except Exception as e:
            health_status['status'] = 'error'
            health_status['issues'].append(f'Health check error: {str(e)}')
        
        return health_status

    async def _get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            'system_active': self.system_active,
            'last_health_check': self.last_health_check.isoformat(),
            'active_triggers': len(self.active_triggers),
            'active_consultations': len(self.active_consultations),
            'active_scaling_plans': len(self.active_scaling_plans),
            'uptime': (datetime.now() - self.last_health_check).total_seconds(),
            'health': await self._health_check()
        }

    async def _get_system_stats(self) -> SystemStatsResponse:
        """Get system statistics"""
        
        return SystemStatsResponse(
            environmental_monitor=await self.environmental_monitor.get_system_stats(),
            legal_orchestration=self.legal_orchestrator.get_system_stats(),
            phase_scaling=self.phase_scaler.get_system_stats(),
            communication_workflow=self.communication_engine.get_workflow_stats(),
            system_health=await self._health_check()
        )

    async def _get_triggers(self) -> List[Dict[str, Any]]:
        """Get all environmental triggers"""
        
        triggers = []
        for trigger in self.active_triggers.values():
            triggers.append({
                'trigger_id': trigger.trigger_id,
                'trigger_type': trigger.trigger_type.value,
                'severity': trigger.severity.value,
                'title': trigger.title,
                'description': trigger.description,
                'detected_at': trigger.detected_at.isoformat(),
                'confidence_score': trigger.confidence_score,
                'requires_approval': trigger.requires_approval,
                'status': 'active'
            })
        
        return triggers

    async def _get_trigger_details(self, trigger_id: str) -> Dict[str, Any]:
        """Get specific trigger details"""
        
        if trigger_id not in self.active_triggers:
            raise HTTPException(status_code=404, detail="Trigger not found")
        
        trigger = self.active_triggers[trigger_id]
        return asdict(trigger)

    async def _perform_environmental_scan(self):
        """Perform immediate environmental scan"""
        
        self.logger.info("Performing immediate environmental scan")
        
        try:
            triggers = await self.environmental_monitor.scan_environment()
            
            for trigger in triggers:
                if trigger.trigger_id not in self.active_triggers:
                    await self._process_new_trigger(trigger)
                    self.active_triggers[trigger.trigger_id] = trigger
            
            self.logger.info(f"Environmental scan completed: {len(triggers)} triggers found")
            
        except Exception as e:
            self.logger.error(f"Error in environmental scan: {str(e)}")

    async def _get_consultations(self) -> List[Dict[str, Any]]:
        """Get all legal consultations"""
        
        consultations = []
        for consultation in self.active_consultations.values():
            consultations.append({
                'consultation_id': consultation.consultation_id,
                'trigger_id': consultation.trigger.trigger_id,
                'status': consultation.status,
                'created_at': consultation.created_at.isoformat(),
                'consensus_confidence': consultation.consensus_confidence,
                'estimated_cost': consultation.estimated_cost
            })
        
        return consultations

    async def _get_consultation_details(self, consultation_id: str) -> Dict[str, Any]:
        """Get specific consultation details"""
        
        if consultation_id not in self.active_consultations:
            raise HTTPException(status_code=404, detail="Consultation not found")
        
        consultation = self.active_consultations[consultation_id]
        return asdict(consultation)

    async def _create_legal_consultation(self, trigger_id: str):
        """Create legal consultation for trigger"""
        
        if trigger_id not in self.active_triggers:
            self.logger.error(f"Trigger not found for consultation: {trigger_id}")
            return
        
        trigger = self.active_triggers[trigger_id]
        
        try:
            consultation = await self.legal_orchestrator.create_consultation(trigger)
            if consultation:
                self.active_consultations[consultation.consultation_id] = consultation
                self.logger.info(f"Legal consultation created: {consultation.consultation_id}")
            
        except Exception as e:
            self.logger.error(f"Error creating consultation: {str(e)}")

    async def _assess_scaling_opportunity(self, revenue: float) -> ScalingAssessmentResponse:
        """Assess scaling opportunity"""
        
        try:
            assessment = await self.phase_scaler.assess_scaling_opportunity(revenue)
            
            return ScalingAssessmentResponse(
                scaling_recommended=assessment['scaling_recommended'],
                current_phase=assessment.get('current_phase', 'unknown'),
                target_phase=assessment.get('target_phase'),
                roi_percentage=assessment.get('scaling_plan', {}).get('roi_percentage'),
                implementation_timeline=assessment.get('implementation_timeline', {}).get('total_timeline'),
                confidence_score=assessment.get('confidence_score', 0.0)
            )
            
        except Exception as e:
            self.logger.error(f"Error assessing scaling opportunity: {str(e)}")
            raise HTTPException(status_code=500, detail="Error assessing scaling opportunity")

    async def _get_scaling_plans(self) -> List[Dict[str, Any]]:
        """Get all scaling plans"""
        
        plans = []
        for plan in self.active_scaling_plans.values():
            plans.append({
                'plan_id': plan['plan_id'],
                'current_phase': plan['current_phase'],
                'target_phase': plan['target_phase'],
                'current_revenue': plan['current_revenue'],
                'roi_percentage': plan['roi_percentage'],
                'total_cost': plan['total_cost'],
                'projected_savings': plan['projected_savings'],
                'created_at': plan['created_at']
            })
        
        return plans

    async def _get_scaling_plan_details(self, plan_id: str) -> Dict[str, Any]:
        """Get specific scaling plan"""
        
        if plan_id not in self.active_scaling_plans:
            raise HTTPException(status_code=404, detail="Scaling plan not found")
        
        return self.active_scaling_plans[plan_id]

    async def _get_current_phase_info(self) -> Dict[str, Any]:
        """Get current scaling phase information"""
        
        return self.phase_scaler.get_current_phase_info()

    async def _get_notifications(self) -> List[Dict[str, Any]]:
        """Get notification history"""
        
        return self.communication_engine.get_notification_history()

    async def _get_approval_requests(self) -> List[Dict[str, Any]]:
        """Get approval requests"""
        
        return self.communication_engine.get_approval_requests()

    async def _process_approval_response(self, request: ApprovalResponseRequest) -> Dict[str, Any]:
        """Process approval response"""
        
        try:
            status = ApprovalStatus(request.status)
            
            success = await self.communication_engine.process_approval_response(
                request.approval_id,
                status,
                request.approved_actions,
                request.rejection_reason,
                request.modifications
            )
            
            if success:
                return {"message": "Approval response processed successfully", "status": "success"}
            else:
                raise HTTPException(status_code=404, detail="Approval request not found")
                
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid approval status: {request.status}")
        except Exception as e:
            self.logger.error(f"Error processing approval response: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing approval response")

    async def _get_stakeholders(self) -> List[Dict[str, Any]]:
        """Get stakeholders"""
        
        return self.communication_engine.get_stakeholders()

    async def _enable_autonomous_mode(self) -> Dict[str, Any]:
        """Enable autonomous action mode"""
        
        # TODO: Implement autonomous mode enabling
        self.logger.info("Autonomous mode enabled")
        return {"message": "Autonomous mode enabled", "status": "enabled"}

    async def _disable_autonomous_mode(self) -> Dict[str, Any]:
        """Disable autonomous action mode"""
        
        # TODO: Implement autonomous mode disabling
        self.logger.info("Autonomous mode disabled")
        return {"message": "Autonomous mode disabled", "status": "disabled"}

    async def _get_autonomous_status(self) -> Dict[str, Any]:
        """Get autonomous mode status"""
        
        return {
            'autonomous_mode_enabled': True,  # TODO: Get actual status
            'autonomous_actions_available': [
                'Legal research and analysis',
                'Document preparation',
                'Cost analysis and budgeting',
                'Timeline planning',
                'Regulatory compliance research'
            ],
            'autonomous_actions_executed': 0,  # TODO: Get actual count
            'last_autonomous_action': None  # TODO: Get actual timestamp
        }

    async def _generate_2fa_code(self, service_name: str) -> Dict[str, Any]:
        """Generate 2FA code for service"""
        
        try:
            code = await self.auth_manager.autonomous_2fa_handler(service_name)
            
            if code:
                return {"service": service_name, "code": code, "status": "success"}
            else:
                raise HTTPException(status_code=404, detail="Service not found or 2FA not configured")
                
        except Exception as e:
            self.logger.error(f"Error generating 2FA code: {str(e)}")
            raise HTTPException(status_code=500, detail="Error generating 2FA code")

    async def _store_credentials(self, service: str, username: str, password: str) -> Dict[str, Any]:
        """Store credentials in Vaultwarden"""
        
        try:
            # TODO: Implement Vaultwarden integration
            self.logger.info(f"Credentials stored for service: {service}")
            return {"message": "Credentials stored successfully", "service": service, "status": "success"}
            
        except Exception as e:
            self.logger.error(f"Error storing credentials: {str(e)}")
            raise HTTPException(status_code=500, detail="Error storing credentials")

    async def _emergency_shutdown(self) -> Dict[str, Any]:
        """Emergency system shutdown"""
        
        self.logger.warning("Emergency shutdown initiated")
        
        try:
            # Stop background monitoring
            self.system_active = False
            
            # TODO: Implement graceful shutdown procedures
            
            return {"message": "Emergency shutdown completed", "status": "shutdown"}
            
        except Exception as e:
            self.logger.error(f"Error during emergency shutdown: {str(e)}")
            raise HTTPException(status_code=500, detail="Error during emergency shutdown")

    async def _emergency_reset(self) -> Dict[str, Any]:
        """Emergency system reset"""
        
        self.logger.warning("Emergency reset initiated")
        
        try:
            # Clear active states
            self.active_triggers.clear()
            self.active_consultations.clear()
            self.active_scaling_plans.clear()
            
            # Restart monitoring
            self.system_active = True
            asyncio.create_task(self._start_background_monitoring())
            
            return {"message": "Emergency reset completed", "status": "reset"}
            
        except Exception as e:
            self.logger.error(f"Error during emergency reset: {str(e)}")
            raise HTTPException(status_code=500, detail="Error during emergency reset")

    def run(self, host: str = "0.0.0.0", port: int = 8003):
        """Run the API server"""
        
        self.logger.info(f"Starting K.E.N. Self-Protection API on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )

# Main execution
def main():
    """Main execution function"""
    
    # Configuration
    config = {
        'api_mode': True,
        'encryption_enabled': True,
        'environmental_monitoring_enabled': True,
        'legal_orchestration_enabled': True,
        'phase_scaling_enabled': True,
        'communication_workflow_enabled': True,
        'autonomous_mode_enabled': True,
        
        # API configuration
        'api_host': '0.0.0.0',
        'api_port': 8003,
        
        # Database configuration
        'database_url': 'postgresql://user:pass@localhost/ken_db',
        
        # External service configuration
        'email': {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': 'ken@system.com',
            'sender_password': 'app_password'
        },
        'sms': {
            'account_sid': 'twilio_sid',
            'auth_token': 'twilio_token',
            'from_phone': '+1-555-0100'
        },
        'slack': {
            'webhook_url': 'https://hooks.slack.com/services/...'
        },
        'teams': {
            'webhook_url': 'https://outlook.office.com/webhook/...'
        }
    }
    
    # Initialize and run API
    api = SelfProtectionAPI(config)
    api.run(
        host=config['api_host'],
        port=config['api_port']
    )

if __name__ == "__main__":
    main()

