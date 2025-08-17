#!/usr/bin/env python3
"""
K.E.N. Archetype Integration Bridge v1.0
Seamless integration between new Agent Generation System and original Archetype System
Unified intelligence orchestration with MENSA + Vertex Expert + Chess Grandmaster capabilities
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid

# Import new agent generation system
from agent_generation_engine import AgentGenerationEngine, AgentType, ArchetypeSpecialty
from tiny_model_generator import TinyModelGenerator, ModelArchitecture, ModelSpecialty
from self_contained_model_engine import SelfContainedModelEngine, SelfContainedArchitecture, IntelligenceSpecialty
from nlp_mastery_engine import NLPMasteryEngine, NLPModel, CommunicationContext

# Import original archetype system
sys.path.append('/home/ubuntu/knowledge-evolution-nexus/services/self-protection')
from legal_orchestration import LegalOrchestrationEngine, ArchetypeExpertise, ExpertArchetype

class IntegratedArchetypeType(Enum):
    # Original archetype types
    LEGAL_EXPERT = "legal_expert"
    FINANCIAL_EXPERT = "financial_expert"
    STRATEGIC_EXPERT = "strategic_expert"
    CRISIS_EXPERT = "crisis_expert"
    TECHNICAL_EXPERT = "technical_expert"
    COMMUNICATION_EXPERT = "communication_expert"
    
    # New agent generation types
    PERMANENT_AGENT = "permanent_agent"
    TEMPORARY_AGENT = "temporary_agent"
    PHANTOM_AGENT = "phantom_agent"
    SINGLE_USE_AGENT = "single_use_agent"
    
    # Hybrid types
    LEGAL_AI_HYBRID = "legal_ai_hybrid"
    FINANCIAL_AI_HYBRID = "financial_ai_hybrid"
    STRATEGIC_AI_HYBRID = "strategic_ai_hybrid"
    CRISIS_AI_HYBRID = "crisis_ai_hybrid"

class IntelligenceLevel(Enum):
    MENSA_BASE = "mensa_base"           # IQ 160+ (.01% MENSA)
    VERTEX_EXPERT = "vertex_expert"     # .01% domain expertise
    CHESS_GRANDMASTER = "chess_grandmaster"  # 2600+ ELO strategic thinking
    TRANSCENDENT = "transcendent"       # Beyond human capability
    OMNISCIENT = "omniscient"          # Maximum intelligence level

@dataclass
class IntegratedArchetype:
    """Unified archetype combining original and new systems"""
    archetype_id: str
    archetype_type: IntegratedArchetypeType
    intelligence_level: IntelligenceLevel
    
    # Original archetype properties
    expert_archetype: Optional[ExpertArchetype] = None
    
    # New agent properties
    agent_id: Optional[str] = None
    model_id: Optional[str] = None
    
    # Unified properties
    name: str = ""
    specializations: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    
    # Intelligence metrics
    mensa_iq_level: int = 180
    vertex_expertise_depth: float = 0.99
    chess_strategic_depth: int = 20
    transcendence_multiplier: float = 1.0
    
    # Performance metrics
    success_rate: float = 0.95
    response_time_ms: float = 3000.0
    cost_per_hour: float = 1.0
    
    # NLP capabilities
    nlp_models: List[NLPModel] = field(default_factory=list)
    communication_effectiveness: float = 0.90
    
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    usage_count: int = 0

@dataclass
class ArchetypeRequest:
    """Request for archetype creation or consultation"""
    request_id: str
    request_type: str  # "create", "consult", "hybrid"
    
    # Requirements
    expertise_required: List[str]
    intelligence_level_required: IntelligenceLevel
    duration: Optional[str] = None  # "permanent", "temporary", "single_use"
    
    # Context
    problem_description: str
    urgency_level: str = "normal"
    budget_constraints: Optional[float] = None
    
    # Communication requirements
    communication_style: str = "professional"
    target_audience: str = "general"
    nlp_requirements: List[NLPModel] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ArchetypeResponse:
    """Response from integrated archetype system"""
    request_id: str
    archetype: IntegratedArchetype
    
    # Analysis results
    analysis: str
    recommendations: List[str]
    implementation_plan: Dict[str, Any]
    
    # Performance predictions
    success_probability: float
    estimated_completion_time: str
    cost_estimate: float
    
    # Communication enhancement
    enhanced_communication: str
    nlp_patterns_used: List[str]
    
    created_at: datetime = field(default_factory=datetime.now)

class ArchetypeIntegrationBridge:
    """
    K.E.N.'s Archetype Integration Bridge
    Seamless integration between original and new archetype systems
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("ArchetypeIntegrationBridge")
        
        # Initialize subsystems
        self.agent_generation_engine = AgentGenerationEngine(config)
        self.tiny_model_generator = TinyModelGenerator(config)
        self.self_contained_engine = SelfContainedModelEngine(config)
        self.nlp_mastery_engine = NLPMasteryEngine(config)
        self.legal_orchestration_engine = LegalOrchestrationEngine(config)
        
        # Integrated archetype registry
        self.integrated_archetypes: Dict[str, IntegratedArchetype] = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_archetypes_created': 0,
            'hybrid_archetypes_created': 0,
            'average_success_rate': 0.0,
            'average_response_time': 0.0,
            'total_consultations': 0
        }
        
        # Archetype mapping
        self.archetype_mappings = self._initialize_archetype_mappings()
        
        self.logger.info("K.E.N. Archetype Integration Bridge initialized")

    def _initialize_archetype_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize mappings between different archetype systems"""
        
        return {
            # Legal expertise mappings
            'legal_expert': {
                'original_expertise': ArchetypeExpertise.TAX_OPTIMIZATION,
                'agent_specialty': ArchetypeSpecialty.LEGAL_INTELLIGENCE,
                'model_specialty': ModelSpecialty.LEGAL_REASONING,
                'intelligence_specialty': IntelligenceSpecialty.LEGAL_REASONING,
                'nlp_models': [NLPModel.META_MODEL, NLPModel.PRESUPPOSITIONS, NLPModel.AUTHORITY_PRINCIPLE],
                'intelligence_level': IntelligenceLevel.TRANSCENDENT
            },
            
            # Financial expertise mappings
            'financial_expert': {
                'original_expertise': ArchetypeExpertise.TRANSFER_PRICING,
                'agent_specialty': ArchetypeSpecialty.FINANCIAL_INTELLIGENCE,
                'model_specialty': ModelSpecialty.FINANCIAL_ANALYSIS,
                'intelligence_specialty': IntelligenceSpecialty.FINANCIAL_ANALYSIS,
                'nlp_models': [NLPModel.CIALDINI_INFLUENCE, NLPModel.SOCIAL_PROOF, NLPModel.SCARCITY_PRINCIPLE],
                'intelligence_level': IntelligenceLevel.OMNISCIENT
            },
            
            # Strategic expertise mappings
            'strategic_expert': {
                'original_expertise': ArchetypeExpertise.COMPETITIVE_INTELLIGENCE,
                'agent_specialty': ArchetypeSpecialty.STRATEGIC_INTELLIGENCE,
                'model_specialty': ModelSpecialty.STRATEGIC_PLANNING,
                'intelligence_specialty': IntelligenceSpecialty.STRATEGIC_PLANNING,
                'nlp_models': [NLPModel.SLEIGHT_OF_MOUTH, NLPModel.FUTURE_PACING, NLPModel.CHESS_STRATEGY],
                'intelligence_level': IntelligenceLevel.TRANSCENDENT
            },
            
            # Crisis expertise mappings
            'crisis_expert': {
                'original_expertise': ArchetypeExpertise.GOVERNMENT_AFFAIRS,
                'agent_specialty': ArchetypeSpecialty.CRISIS_INTELLIGENCE,
                'model_specialty': ModelSpecialty.CRISIS_RESPONSE,
                'intelligence_specialty': IntelligenceSpecialty.CRISIS_RESPONSE,
                'nlp_models': [NLPModel.RAPPORT_BUILDING, NLPModel.REFRAMING, NLPModel.PARTS_INTEGRATION],
                'intelligence_level': IntelligenceLevel.OMNISCIENT
            },
            
            # Technical expertise mappings
            'technical_expert': {
                'original_expertise': ArchetypeExpertise.PATENT_ATTORNEY,
                'agent_specialty': ArchetypeSpecialty.TECHNICAL_INTELLIGENCE,
                'model_specialty': ModelSpecialty.TECHNICAL_ANALYSIS,
                'intelligence_specialty': IntelligenceSpecialty.TECHNICAL_ANALYSIS,
                'nlp_models': [NLPModel.META_MODEL, NLPModel.REPRESENTATIONAL_SYSTEMS],
                'intelligence_level': IntelligenceLevel.VERTEX_EXPERT
            },
            
            # Communication expertise mappings
            'communication_expert': {
                'original_expertise': ArchetypeExpertise.GOVERNMENT_AFFAIRS,
                'agent_specialty': ArchetypeSpecialty.COMMUNICATION_INTELLIGENCE,
                'model_specialty': ModelSpecialty.COMMUNICATION,
                'intelligence_specialty': IntelligenceSpecialty.COMMUNICATION,
                'nlp_models': [NLPModel.MILTON_ERICKSON_HYPNOTIC, NLPModel.EMBEDDED_COMMANDS, NLPModel.ANCHORING],
                'intelligence_level': IntelligenceLevel.TRANSCENDENT
            }
        }

    async def create_integrated_archetype(
        self,
        archetype_request: ArchetypeRequest
    ) -> IntegratedArchetype:
        """Create integrated archetype combining original and new systems"""
        
        archetype_id = str(uuid.uuid4())
        
        self.logger.info(f"Creating integrated archetype: {archetype_request.request_type}")
        
        # Determine archetype type based on requirements
        archetype_type = self._determine_archetype_type(archetype_request)
        
        # Get mapping for this archetype type
        mapping = self.archetype_mappings.get(archetype_type.value.replace('_hybrid', '').replace('_agent', ''))
        
        if not mapping:
            mapping = self.archetype_mappings['strategic_expert']  # Default fallback
        
        # Create original expert archetype if needed
        expert_archetype = None
        if archetype_request.request_type in ["create", "hybrid"]:
            expert_archetype = await self._create_original_archetype(mapping, archetype_request)
        
        # Create new AI agent if needed
        agent_id = None
        model_id = None
        if archetype_request.request_type in ["create", "hybrid"] or "agent" in archetype_type.value:
            agent_id, model_id = await self._create_ai_agent(mapping, archetype_request)
        
        # Determine intelligence level
        intelligence_level = archetype_request.intelligence_level_required or mapping['intelligence_level']
        
        # Calculate intelligence parameters
        intelligence_params = self._calculate_intelligence_parameters(intelligence_level)
        
        # Select NLP models
        nlp_models = archetype_request.nlp_requirements or mapping['nlp_models']
        
        # Create integrated archetype
        integrated_archetype = IntegratedArchetype(
            archetype_id=archetype_id,
            archetype_type=archetype_type,
            intelligence_level=intelligence_level,
            expert_archetype=expert_archetype,
            agent_id=agent_id,
            model_id=model_id,
            name=f"{archetype_type.value.replace('_', ' ').title()} Archetype",
            specializations=archetype_request.expertise_required,
            capabilities=self._determine_capabilities(archetype_type, mapping),
            mensa_iq_level=intelligence_params['iq_level'],
            vertex_expertise_depth=intelligence_params['expertise_depth'],
            chess_strategic_depth=intelligence_params['strategic_depth'],
            transcendence_multiplier=intelligence_params['transcendence_multiplier'],
            nlp_models=nlp_models,
            communication_effectiveness=self._calculate_communication_effectiveness(nlp_models)
        )
        
        # Store archetype
        self.integrated_archetypes[archetype_id] = integrated_archetype
        
        # Update performance metrics
        self._update_performance_metrics(integrated_archetype)
        
        self.logger.info(f"Integrated archetype created: {archetype_id}")
        
        return integrated_archetype

    def _determine_archetype_type(self, request: ArchetypeRequest) -> IntegratedArchetypeType:
        """Determine archetype type based on request"""
        
        # Check for specific expertise requirements
        expertise_keywords = {
            'legal': IntegratedArchetypeType.LEGAL_AI_HYBRID,
            'financial': IntegratedArchetypeType.FINANCIAL_AI_HYBRID,
            'strategic': IntegratedArchetypeType.STRATEGIC_AI_HYBRID,
            'crisis': IntegratedArchetypeType.CRISIS_AI_HYBRID,
            'technical': IntegratedArchetypeType.TECHNICAL_EXPERT,
            'communication': IntegratedArchetypeType.COMMUNICATION_EXPERT
        }
        
        for keyword, archetype_type in expertise_keywords.items():
            if any(keyword in expertise.lower() for expertise in request.expertise_required):
                return archetype_type
        
        # Check duration for agent types
        if request.duration == "permanent":
            return IntegratedArchetypeType.PERMANENT_AGENT
        elif request.duration == "temporary":
            return IntegratedArchetypeType.TEMPORARY_AGENT
        elif request.duration == "single_use":
            return IntegratedArchetypeType.SINGLE_USE_AGENT
        
        # Default to strategic hybrid
        return IntegratedArchetypeType.STRATEGIC_AI_HYBRID

    async def _create_original_archetype(
        self,
        mapping: Dict[str, Any],
        request: ArchetypeRequest
    ) -> ExpertArchetype:
        """Create original expert archetype"""
        
        expert_archetype = ExpertArchetype(
            archetype_id=str(uuid.uuid4()),
            expertise=mapping['original_expertise'],
            name=f"Expert {mapping['original_expertise'].value}",
            credentials=["PhD", "20+ years experience", "Industry recognition"],
            specializations=request.expertise_required,
            jurisdictions=[],  # Will be populated based on requirements
            mensa_percentile=99.99,  # .01% MENSA Society member
            vertex_expertise_level=99.99,  # .01% vertex expert
            chess_grandmaster_rating=2600,  # Chess Grandmaster level
            consultation_rate=500.0,  # USD per hour
            availability_score=1.0,
            success_rate=0.95,
            created_at=datetime.now()
        )
        
        return expert_archetype

    async def _create_ai_agent(
        self,
        mapping: Dict[str, Any],
        request: ArchetypeRequest
    ) -> Tuple[str, str]:
        """Create AI agent using new generation system"""
        
        # Create agent
        agent = await self.agent_generation_engine.create_agent(
            agent_type=AgentType.PERMANENT if request.duration == "permanent" else AgentType.PHANTOM,
            archetype_specialty=mapping['agent_specialty'],
            purpose=request.problem_description,
            intelligence_level=request.intelligence_level_required.value if request.intelligence_level_required else "transcendent"
        )
        
        # Create tiny model
        tiny_model = await self.tiny_model_generator.generate_tiny_model(
            architecture=ModelArchitecture.SMALL,
            specialty=mapping['model_specialty']
        )
        
        return agent.agent_id, tiny_model.model_id

    def _calculate_intelligence_parameters(self, intelligence_level: IntelligenceLevel) -> Dict[str, Any]:
        """Calculate intelligence parameters based on level"""
        
        intelligence_configs = {
            IntelligenceLevel.MENSA_BASE: {
                'iq_level': 160,
                'expertise_depth': 0.90,
                'strategic_depth': 15,
                'transcendence_multiplier': 1.0
            },
            IntelligenceLevel.VERTEX_EXPERT: {
                'iq_level': 170,
                'expertise_depth': 0.95,
                'strategic_depth': 20,
                'transcendence_multiplier': 1.2
            },
            IntelligenceLevel.CHESS_GRANDMASTER: {
                'iq_level': 180,
                'expertise_depth': 0.97,
                'strategic_depth': 30,
                'transcendence_multiplier': 1.5
            },
            IntelligenceLevel.TRANSCENDENT: {
                'iq_level': 190,
                'expertise_depth': 0.99,
                'strategic_depth': 40,
                'transcendence_multiplier': 2.0
            },
            IntelligenceLevel.OMNISCIENT: {
                'iq_level': 200,
                'expertise_depth': 0.999,
                'strategic_depth': 50,
                'transcendence_multiplier': 3.0
            }
        }
        
        return intelligence_configs.get(intelligence_level, intelligence_configs[IntelligenceLevel.TRANSCENDENT])

    def _determine_capabilities(
        self,
        archetype_type: IntegratedArchetypeType,
        mapping: Dict[str, Any]
    ) -> List[str]:
        """Determine capabilities based on archetype type"""
        
        base_capabilities = [
            "MENSA-level reasoning",
            "Vertex expert knowledge",
            "Chess Grandmaster strategic thinking",
            "Advanced NLP communication",
            "Real-time analysis",
            "Autonomous decision making"
        ]
        
        specialty_capabilities = {
            IntegratedArchetypeType.LEGAL_AI_HYBRID: [
                "Legal research and analysis",
                "Regulatory compliance assessment",
                "Contract optimization",
                "Risk assessment",
                "Jurisdiction analysis"
            ],
            IntegratedArchetypeType.FINANCIAL_AI_HYBRID: [
                "Financial modeling",
                "Investment analysis",
                "Tax optimization",
                "Risk management",
                "Portfolio optimization"
            ],
            IntegratedArchetypeType.STRATEGIC_AI_HYBRID: [
                "Strategic planning",
                "Competitive analysis",
                "Market research",
                "Resource optimization",
                "Scenario planning"
            ],
            IntegratedArchetypeType.CRISIS_AI_HYBRID: [
                "Crisis assessment",
                "Emergency response planning",
                "Stakeholder communication",
                "Risk mitigation",
                "Recovery planning"
            ]
        }
        
        return base_capabilities + specialty_capabilities.get(archetype_type, [])

    def _calculate_communication_effectiveness(self, nlp_models: List[NLPModel]) -> float:
        """Calculate communication effectiveness based on NLP models"""
        
        base_effectiveness = 0.70
        
        # Add effectiveness for each NLP model
        model_bonuses = {
            NLPModel.MILTON_ERICKSON_HYPNOTIC: 0.15,
            NLPModel.CIALDINI_INFLUENCE: 0.12,
            NLPModel.RAPPORT_BUILDING: 0.10,
            NLPModel.REPRESENTATIONAL_SYSTEMS: 0.08,
            NLPModel.PRESUPPOSITIONS: 0.07,
            NLPModel.EMBEDDED_COMMANDS: 0.06,
            NLPModel.REFRAMING: 0.05
        }
        
        total_bonus = sum(model_bonuses.get(model, 0.03) for model in nlp_models)
        
        return min(base_effectiveness + total_bonus, 0.98)

    async def consult_integrated_archetype(
        self,
        archetype_request: ArchetypeRequest
    ) -> ArchetypeResponse:
        """Consult with integrated archetype system"""
        
        # Find or create appropriate archetype
        archetype = await self._find_or_create_archetype(archetype_request)
        
        # Perform analysis using both systems
        analysis_results = await self._perform_integrated_analysis(archetype, archetype_request)
        
        # Enhance communication using NLP
        enhanced_communication = await self._enhance_communication(
            analysis_results['analysis'],
            archetype.nlp_models,
            archetype_request
        )
        
        # Create response
        response = ArchetypeResponse(
            request_id=archetype_request.request_id,
            archetype=archetype,
            analysis=analysis_results['analysis'],
            recommendations=analysis_results['recommendations'],
            implementation_plan=analysis_results['implementation_plan'],
            success_probability=analysis_results['success_probability'],
            estimated_completion_time=analysis_results['estimated_completion_time'],
            cost_estimate=analysis_results['cost_estimate'],
            enhanced_communication=enhanced_communication['enhanced_text'],
            nlp_patterns_used=enhanced_communication['patterns_used']
        )
        
        # Update archetype usage
        archetype.last_used = datetime.now()
        archetype.usage_count += 1
        
        # Update performance metrics
        self.performance_metrics['total_consultations'] += 1
        
        self.logger.info(f"Consultation completed: {archetype_request.request_id}")
        
        return response

    async def _find_or_create_archetype(
        self,
        request: ArchetypeRequest
    ) -> IntegratedArchetype:
        """Find existing or create new archetype for request"""
        
        # Look for existing archetype that matches requirements
        for archetype in self.integrated_archetypes.values():
            if self._archetype_matches_requirements(archetype, request):
                return archetype
        
        # Create new archetype if none found
        return await self.create_integrated_archetype(request)

    def _archetype_matches_requirements(
        self,
        archetype: IntegratedArchetype,
        request: ArchetypeRequest
    ) -> bool:
        """Check if archetype matches request requirements"""
        
        # Check expertise overlap
        expertise_overlap = any(
            expertise.lower() in [spec.lower() for spec in archetype.specializations]
            for expertise in request.expertise_required
        )
        
        # Check intelligence level
        intelligence_match = (
            not request.intelligence_level_required or
            archetype.intelligence_level == request.intelligence_level_required
        )
        
        return expertise_overlap and intelligence_match

    async def _perform_integrated_analysis(
        self,
        archetype: IntegratedArchetype,
        request: ArchetypeRequest
    ) -> Dict[str, Any]:
        """Perform analysis using integrated archetype capabilities"""
        
        analysis_results = {
            'analysis': f"Integrated analysis using {archetype.intelligence_level.value} intelligence level.",
            'recommendations': [
                "Leverage MENSA-level reasoning for complex problem solving",
                "Apply vertex expertise for domain-specific insights",
                "Use Chess Grandmaster strategic thinking for multi-move planning",
                "Implement NLP techniques for enhanced communication"
            ],
            'implementation_plan': {
                'phase_1': 'Initial assessment and planning',
                'phase_2': 'Strategy development and resource allocation',
                'phase_3': 'Implementation and monitoring',
                'phase_4': 'Optimization and continuous improvement'
            },
            'success_probability': archetype.success_rate,
            'estimated_completion_time': self._estimate_completion_time(request),
            'cost_estimate': self._estimate_cost(archetype, request)
        }
        
        # Use original archetype if available
        if archetype.expert_archetype:
            original_analysis = f"Expert consultation from {archetype.expert_archetype.name} "
            original_analysis += f"with {archetype.expert_archetype.success_rate:.1%} success rate."
            analysis_results['analysis'] += f" {original_analysis}"
        
        # Use AI agent if available
        if archetype.agent_id and archetype.model_id:
            ai_analysis = await self.tiny_model_generator.inference(
                archetype.model_id,
                request.problem_description,
                "analysis"
            )
            analysis_results['analysis'] += f" AI Agent Analysis: {ai_analysis}"
        
        return analysis_results

    async def _enhance_communication(
        self,
        text: str,
        nlp_models: List[NLPModel],
        request: ArchetypeRequest
    ) -> Dict[str, Any]:
        """Enhance communication using NLP models"""
        
        # Create communication context
        context = CommunicationContext(
            target_audience=request.target_audience,
            communication_goal="consultation",
            emotional_state="professional",
            relationship_level="expert",
            cultural_context="business",
            time_constraints=request.urgency_level,
            resistance_level="low"
        )
        
        # Enhance communication
        nlp_response = await self.nlp_mastery_engine.enhance_communication(
            text,
            context,
            nlp_models
        )
        
        return {
            'enhanced_text': nlp_response.enhanced_text,
            'patterns_used': nlp_response.embedded_patterns,
            'effectiveness_prediction': nlp_response.effectiveness_prediction
        }

    def _estimate_completion_time(self, request: ArchetypeRequest) -> str:
        """Estimate completion time based on request complexity"""
        
        complexity_factors = {
            'simple': '1-2 hours',
            'moderate': '4-8 hours',
            'complex': '1-3 days',
            'very_complex': '1-2 weeks'
        }
        
        # Simple complexity assessment
        if len(request.expertise_required) <= 1:
            complexity = 'simple'
        elif len(request.expertise_required) <= 3:
            complexity = 'moderate'
        elif len(request.expertise_required) <= 5:
            complexity = 'complex'
        else:
            complexity = 'very_complex'
        
        return complexity_factors[complexity]

    def _estimate_cost(self, archetype: IntegratedArchetype, request: ArchetypeRequest) -> float:
        """Estimate cost based on archetype and request"""
        
        base_cost = archetype.cost_per_hour
        
        # Adjust for intelligence level
        intelligence_multipliers = {
            IntelligenceLevel.MENSA_BASE: 1.0,
            IntelligenceLevel.VERTEX_EXPERT: 1.2,
            IntelligenceLevel.CHESS_GRANDMASTER: 1.5,
            IntelligenceLevel.TRANSCENDENT: 2.0,
            IntelligenceLevel.OMNISCIENT: 3.0
        }
        
        multiplier = intelligence_multipliers.get(archetype.intelligence_level, 1.0)
        
        # Estimate hours based on complexity
        estimated_hours = len(request.expertise_required) * 2  # 2 hours per expertise area
        
        return base_cost * multiplier * estimated_hours

    def _update_performance_metrics(self, archetype: IntegratedArchetype):
        """Update performance metrics"""
        
        self.performance_metrics['total_archetypes_created'] += 1
        
        if 'hybrid' in archetype.archetype_type.value:
            self.performance_metrics['hybrid_archetypes_created'] += 1
        
        # Update averages
        total_archetypes = len(self.integrated_archetypes)
        if total_archetypes > 0:
            self.performance_metrics['average_success_rate'] = sum(
                a.success_rate for a in self.integrated_archetypes.values()
            ) / total_archetypes
            
            self.performance_metrics['average_response_time'] = sum(
                a.response_time_ms for a in self.integrated_archetypes.values()
            ) / total_archetypes

    def get_available_archetypes(self) -> List[Dict[str, Any]]:
        """Get list of available integrated archetypes"""
        
        archetypes = []
        for archetype in self.integrated_archetypes.values():
            archetypes.append({
                'archetype_id': archetype.archetype_id,
                'archetype_type': archetype.archetype_type.value,
                'intelligence_level': archetype.intelligence_level.value,
                'name': archetype.name,
                'specializations': archetype.specializations,
                'capabilities': archetype.capabilities,
                'success_rate': archetype.success_rate,
                'communication_effectiveness': archetype.communication_effectiveness,
                'usage_count': archetype.usage_count,
                'last_used': archetype.last_used.isoformat() if archetype.last_used else None,
                'has_expert_archetype': archetype.expert_archetype is not None,
                'has_ai_agent': archetype.agent_id is not None,
                'nlp_models_count': len(archetype.nlp_models)
            })
        
        return archetypes

    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics"""
        
        return {
            'performance_metrics': self.performance_metrics,
            'total_integrated_archetypes': len(self.integrated_archetypes),
            'archetype_types_distribution': {
                archetype_type.value: sum(
                    1 for a in self.integrated_archetypes.values()
                    if a.archetype_type == archetype_type
                )
                for archetype_type in IntegratedArchetypeType
            },
            'intelligence_levels_distribution': {
                level.value: sum(
                    1 for a in self.integrated_archetypes.values()
                    if a.intelligence_level == level
                )
                for level in IntelligenceLevel
            },
            'subsystems_status': {
                'agent_generation_engine': 'active',
                'tiny_model_generator': 'active',
                'self_contained_engine': 'active',
                'nlp_mastery_engine': 'active',
                'legal_orchestration_engine': 'active'
            }
        }

# Main execution for testing
async def main():
    """Main execution function for testing"""
    
    # Configuration
    config = {
        'models_dir': '/app/data/models',
        'training_data_dir': '/app/data/training',
        'nlp_models_dir': '/app/data/nlp_models'
    }
    
    # Initialize integration bridge
    bridge = ArchetypeIntegrationBridge(config)
    
    # Create test request
    request = ArchetypeRequest(
        request_id=str(uuid.uuid4()),
        request_type="hybrid",
        expertise_required=["legal analysis", "financial optimization"],
        intelligence_level_required=IntelligenceLevel.TRANSCENDENT,
        duration="permanent",
        problem_description="Optimize corporate structure for tax efficiency and legal compliance",
        urgency_level="normal",
        communication_style="professional",
        target_audience="business_executives"
    )
    
    # Create integrated archetype
    archetype = await bridge.create_integrated_archetype(request)
    
    # Perform consultation
    response = await bridge.consult_integrated_archetype(request)
    
    # Display results
    print(f"Created archetype: {archetype.name}")
    print(f"Intelligence level: {archetype.intelligence_level.value}")
    print(f"Capabilities: {len(archetype.capabilities)}")
    print(f"NLP models: {len(archetype.nlp_models)}")
    print(f"Analysis: {response.analysis[:200]}...")
    print(f"Enhanced communication: {response.enhanced_communication[:200]}...")
    
    # Get statistics
    available_archetypes = bridge.get_available_archetypes()
    integration_stats = bridge.get_integration_stats()
    
    print(f"Available archetypes: {len(available_archetypes)}")
    print(f"Integration stats: {integration_stats['performance_metrics']}")

if __name__ == "__main__":
    asyncio.run(main())

