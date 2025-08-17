#!/usr/bin/env python3
"""
K.E.N. Agent Lifecycle Manager v1.0
Advanced lifecycle management for permanent, temporary, purpose-based, and phantom agents
Intelligent resource allocation, performance optimization, and autonomous scaling
"""

import asyncio
import json
import logging
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import psutil
import pickle
import hashlib

from agent_generation_engine import (
    AgentType, AgentStatus, ArchetypeSpecialty, IntelligenceLevel,
    AutonomousAgent, AgentConfiguration, AgentGenerationEngine
)

class LifecycleEvent(Enum):
    CREATED = "created"
    ACTIVATED = "activated"
    TASK_ASSIGNED = "task_assigned"
    TASK_COMPLETED = "task_completed"
    PERFORMANCE_UPDATED = "performance_updated"
    SUSPENDED = "suspended"
    RESUMED = "resumed"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    PHANTOM_DISSOLVED = "phantom_dissolved"

class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"

class ScalingTrigger(Enum):
    PERFORMANCE_DEGRADATION = "performance_degradation"
    HIGH_DEMAND = "high_demand"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    COST_OPTIMIZATION = "cost_optimization"
    STRATEGIC_SCALING = "strategic_scaling"

@dataclass
class LifecyclePolicy:
    """Policy for agent lifecycle management"""
    policy_id: str
    name: str
    description: str
    
    # Agent type policies
    permanent_max_idle_hours: int = 168  # 1 week
    temporary_max_duration_hours: int = 24
    phantom_max_idle_minutes: int = 30
    single_use_auto_terminate: bool = True
    
    # Performance policies
    min_success_rate: float = 0.7
    max_response_time_ms: float = 5000.0
    performance_review_interval_hours: int = 6
    
    # Resource policies
    max_cpu_percent: float = 80.0
    max_memory_mb: int = 1024
    max_storage_mb: int = 512
    resource_check_interval_minutes: int = 5
    
    # Scaling policies
    auto_scaling_enabled: bool = True
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    min_agents_per_archetype: int = 1
    max_agents_per_archetype: int = 10
    
    # Cost policies
    max_cost_per_hour: float = 1.0
    cost_optimization_enabled: bool = True
    budget_alert_threshold: float = 0.8
    
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ResourceUsage:
    """Resource usage tracking for agents"""
    agent_id: str
    timestamp: datetime
    
    # System resources
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    storage_mb: float = 0.0
    network_bytes: int = 0
    
    # Performance metrics
    response_time_ms: float = 0.0
    throughput_per_second: float = 0.0
    error_rate: float = 0.0
    
    # Cost metrics
    cost_per_hour: float = 0.0
    total_cost: float = 0.0

@dataclass
class LifecycleEvent:
    """Lifecycle event record"""
    event_id: str
    agent_id: str
    event_type: LifecycleEvent
    timestamp: datetime
    
    # Event details
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Context
    triggered_by: str = "system"
    policy_id: Optional[str] = None
    
    # Impact
    resource_impact: Dict[str, float] = field(default_factory=dict)
    performance_impact: Dict[str, float] = field(default_factory=dict)

class AgentLifecycleManager:
    """
    K.E.N.'s Advanced Agent Lifecycle Manager
    Manages permanent, temporary, purpose-based, and phantom agents
    """
    
    def __init__(self, config: Dict[str, Any], agent_engine: AgentGenerationEngine):
        self.config = config
        self.agent_engine = agent_engine
        self.logger = logging.getLogger("AgentLifecycleManager")
        
        # Database setup
        self.db_path = Path(config.get('lifecycle_db_path', '/app/data/lifecycle.db'))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Lifecycle policies
        self.policies: Dict[str, LifecyclePolicy] = {}
        self.default_policy_id = "default"
        
        # Resource monitoring
        self.resource_usage: Dict[str, List[ResourceUsage]] = {}
        self.system_resources = {
            'cpu_cores': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'storage_gb': psutil.disk_usage('/').total / (1024**3)
        }
        
        # Lifecycle events
        self.lifecycle_events: List[LifecycleEvent] = []
        self.event_handlers: Dict[LifecycleEvent, List[callable]] = {}
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.scaling_decisions: List[Dict[str, Any]] = []
        
        # Background tasks
        self.monitoring_active = False
        self.monitoring_tasks: List[asyncio.Task] = []
        
        # Initialize system
        self._initialize_database()
        self._create_default_policies()
        self._start_monitoring()
        
        self.logger.info("K.E.N. Agent Lifecycle Manager initialized")

    def _initialize_database(self):
        """Initialize SQLite database for lifecycle management"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Lifecycle policies table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lifecycle_policies (
                    policy_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    policy_data TEXT NOT NULL,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            
            # Lifecycle events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lifecycle_events (
                    event_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    description TEXT,
                    metadata TEXT,
                    triggered_by TEXT,
                    policy_id TEXT
                )
            """)
            
            # Resource usage table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS resource_usage (
                    usage_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    cpu_percent REAL,
                    memory_mb REAL,
                    storage_mb REAL,
                    network_bytes INTEGER,
                    response_time_ms REAL,
                    throughput_per_second REAL,
                    error_rate REAL,
                    cost_per_hour REAL,
                    total_cost REAL
                )
            """)
            
            # Performance history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_history (
                    record_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    success_rate REAL,
                    average_response_time REAL,
                    efficiency_score REAL,
                    tasks_completed INTEGER,
                    cost_efficiency REAL
                )
            """)
            
            # Scaling decisions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scaling_decisions (
                    decision_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    archetype_specialty TEXT NOT NULL,
                    action_taken TEXT NOT NULL,
                    agents_before INTEGER,
                    agents_after INTEGER,
                    expected_impact TEXT,
                    actual_impact TEXT
                )
            """)
            
            conn.commit()

    def _create_default_policies(self):
        """Create default lifecycle policies"""
        
        # Default policy
        default_policy = LifecyclePolicy(
            policy_id="default",
            name="Default Lifecycle Policy",
            description="Standard lifecycle management for all agent types"
        )
        
        # High-performance policy
        high_performance_policy = LifecyclePolicy(
            policy_id="high_performance",
            name="High Performance Policy",
            description="Optimized for maximum performance and responsiveness",
            min_success_rate=0.9,
            max_response_time_ms=2000.0,
            performance_review_interval_hours=1,
            max_cpu_percent=95.0,
            max_memory_mb=2048,
            max_agents_per_archetype=20
        )
        
        # Cost-optimized policy
        cost_optimized_policy = LifecyclePolicy(
            policy_id="cost_optimized",
            name="Cost Optimized Policy",
            description="Optimized for cost efficiency",
            permanent_max_idle_hours=24,
            temporary_max_duration_hours=8,
            phantom_max_idle_minutes=15,
            max_cost_per_hour=0.5,
            max_agents_per_archetype=5,
            scale_down_threshold=0.2
        )
        
        # Crisis response policy
        crisis_policy = LifecyclePolicy(
            policy_id="crisis_response",
            name="Crisis Response Policy",
            description="Emergency scaling and performance for crisis situations",
            min_success_rate=0.95,
            max_response_time_ms=1000.0,
            performance_review_interval_hours=0.5,
            max_agents_per_archetype=50,
            auto_scaling_enabled=True,
            scale_up_threshold=0.6
        )
        
        # Store policies
        self.policies[default_policy.policy_id] = default_policy
        self.policies[high_performance_policy.policy_id] = high_performance_policy
        self.policies[cost_optimized_policy.policy_id] = cost_optimized_policy
        self.policies[crisis_policy.policy_id] = crisis_policy
        
        # Save to database
        for policy in self.policies.values():
            self._save_policy_to_database(policy)

    def _save_policy_to_database(self, policy: LifecyclePolicy):
        """Save lifecycle policy to database"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO lifecycle_policies 
                (policy_id, name, description, policy_data, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                policy.policy_id,
                policy.name,
                policy.description,
                json.dumps(asdict(policy), default=str),
                policy.created_at.isoformat(),
                datetime.now().isoformat()
            ))
            conn.commit()

    def _start_monitoring(self):
        """Start background monitoring tasks"""
        
        self.monitoring_active = True
        
        # Start monitoring threads
        monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        resource_thread = threading.Thread(target=self._resource_monitoring_loop, daemon=True)
        resource_thread.start()
        
        performance_thread = threading.Thread(target=self._performance_monitoring_loop, daemon=True)
        performance_thread.start()
        
        scaling_thread = threading.Thread(target=self._scaling_monitoring_loop, daemon=True)
        scaling_thread.start()
        
        self.logger.info("Lifecycle monitoring started")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Check agent lifecycles
                self._check_agent_lifecycles()
                
                # Process lifecycle events
                self._process_lifecycle_events()
                
                # Cleanup expired data
                self._cleanup_expired_data()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(10)

    def _resource_monitoring_loop(self):
        """Resource monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Monitor agent resource usage
                self._monitor_agent_resources()
                
                # Check resource limits
                self._check_resource_limits()
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {str(e)}")
                time.sleep(30)

    def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Monitor agent performance
                self._monitor_agent_performance()
                
                # Analyze performance trends
                self._analyze_performance_trends()
                
                time.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {str(e)}")
                time.sleep(60)

    def _scaling_monitoring_loop(self):
        """Scaling monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Check scaling triggers
                self._check_scaling_triggers()
                
                # Execute scaling decisions
                self._execute_scaling_decisions()
                
                time.sleep(600)  # Check every 10 minutes
                
            except Exception as e:
                self.logger.error(f"Error in scaling monitoring: {str(e)}")
                time.sleep(60)

    def _check_agent_lifecycles(self):
        """Check agent lifecycles and apply policies"""
        
        current_time = datetime.now()
        
        for agent_id, agent in self.agent_engine.active_agents.items():
            policy = self._get_agent_policy(agent)
            
            # Check expiration based on agent type
            if agent.configuration.agent_type == AgentType.TEMPORARY:
                if agent.expires_at and current_time >= agent.expires_at:
                    asyncio.create_task(self._expire_agent(agent_id, "Temporary agent expired"))
                    continue
            
            elif agent.configuration.agent_type == AgentType.PHANTOM:
                idle_time = current_time - agent.last_active
                if (agent.status == AgentStatus.IDLE and 
                    idle_time > timedelta(minutes=policy.phantom_max_idle_minutes)):
                    asyncio.create_task(self._dissolve_phantom_agent(agent_id))
                    continue
            
            elif agent.configuration.agent_type == AgentType.PERMANENT:
                idle_time = current_time - agent.last_active
                if (agent.status == AgentStatus.IDLE and 
                    idle_time > timedelta(hours=policy.permanent_max_idle_hours)):
                    asyncio.create_task(self._suspend_agent(agent_id, "Permanent agent idle too long"))
                    continue
            
            elif agent.configuration.agent_type == AgentType.SINGLE_USE:
                if (policy.single_use_auto_terminate and 
                    agent.total_tasks_completed > 0 and 
                    agent.status == AgentStatus.IDLE):
                    asyncio.create_task(self._terminate_agent(agent_id, "Single-use task completed"))
                    continue
            
            # Check performance thresholds
            if (agent.success_rate < policy.min_success_rate and 
                agent.total_tasks_completed > 5):
                asyncio.create_task(self._handle_poor_performance(agent_id))
                continue

    def _get_agent_policy(self, agent: AutonomousAgent) -> LifecyclePolicy:
        """Get applicable policy for agent"""
        
        # Check for agent-specific policy
        agent_policy_id = agent.configuration.__dict__.get('policy_id', self.default_policy_id)
        
        # Check for archetype-specific policy
        if agent.configuration.archetype_specialty == ArchetypeSpecialty.CRISIS_MANAGEMENT:
            return self.policies.get('crisis_response', self.policies[self.default_policy_id])
        
        return self.policies.get(agent_policy_id, self.policies[self.default_policy_id])

    async def _expire_agent(self, agent_id: str, reason: str):
        """Expire agent due to lifecycle policy"""
        
        await self._record_lifecycle_event(
            agent_id, LifecycleEvent.EXPIRED, reason, {"expiration_reason": reason}
        )
        
        await self.agent_engine.terminate_agent(agent_id, reason)
        
        self.logger.info(f"Agent expired: {agent_id} - {reason}")

    async def _dissolve_phantom_agent(self, agent_id: str):
        """Dissolve phantom agent"""
        
        await self._record_lifecycle_event(
            agent_id, LifecycleEvent.PHANTOM_DISSOLVED, "Phantom agent dissolved due to inactivity"
        )
        
        await self.agent_engine.terminate_agent(agent_id, "Phantom agent dissolved")
        
        self.logger.info(f"Phantom agent dissolved: {agent_id}")

    async def _suspend_agent(self, agent_id: str, reason: str):
        """Suspend agent temporarily"""
        
        if agent_id in self.agent_engine.active_agents:
            agent = self.agent_engine.active_agents[agent_id]
            agent.status = AgentStatus.SUSPENDED
            
            await self._record_lifecycle_event(
                agent_id, LifecycleEvent.SUSPENDED, reason, {"suspension_reason": reason}
            )
            
            self.logger.info(f"Agent suspended: {agent_id} - {reason}")

    async def _terminate_agent(self, agent_id: str, reason: str):
        """Terminate agent"""
        
        await self._record_lifecycle_event(
            agent_id, LifecycleEvent.TERMINATED, reason, {"termination_reason": reason}
        )
        
        await self.agent_engine.terminate_agent(agent_id, reason)
        
        self.logger.info(f"Agent terminated: {agent_id} - {reason}")

    async def _handle_poor_performance(self, agent_id: str):
        """Handle agent with poor performance"""
        
        agent = self.agent_engine.active_agents.get(agent_id)
        if not agent:
            return
        
        # Try to improve performance first
        await self._optimize_agent_performance(agent_id)
        
        # If still poor performance, consider replacement
        if agent.success_rate < 0.5:
            await self._replace_poor_performing_agent(agent_id)

    async def _optimize_agent_performance(self, agent_id: str):
        """Optimize agent performance"""
        
        # Record optimization attempt
        await self._record_lifecycle_event(
            agent_id, LifecycleEvent.PERFORMANCE_UPDATED, 
            "Performance optimization attempted"
        )
        
        # In production, implement performance optimization strategies
        self.logger.info(f"Optimizing performance for agent: {agent_id}")

    async def _replace_poor_performing_agent(self, agent_id: str):
        """Replace poor performing agent"""
        
        agent = self.agent_engine.active_agents.get(agent_id)
        if not agent:
            return
        
        # Create replacement agent
        replacement_agent = await self.agent_engine.create_agent(
            agent.configuration.agent_type,
            agent.configuration.archetype_specialty,
            purpose=f"Replacement for poor performing agent {agent_id}"
        )
        
        # Transfer any pending tasks
        for task in agent.assigned_tasks:
            await self.agent_engine.assign_task(replacement_agent.agent_id, task)
        
        # Terminate poor performing agent
        await self._terminate_agent(agent_id, "Replaced due to poor performance")
        
        self.logger.info(f"Replaced poor performing agent {agent_id} with {replacement_agent.agent_id}")

    def _monitor_agent_resources(self):
        """Monitor resource usage for all agents"""
        
        current_time = datetime.now()
        
        for agent_id, agent in self.agent_engine.active_agents.items():
            # Simulate resource monitoring (in production, get actual metrics)
            resource_usage = ResourceUsage(
                agent_id=agent_id,
                timestamp=current_time,
                cpu_percent=min(100.0, max(0.0, 
                    agent.efficiency_score * 50 + (1 - agent.success_rate) * 30)),
                memory_mb=min(2048.0, max(128.0,
                    len(agent.assigned_tasks) * 64 + agent.total_tasks_completed * 2)),
                storage_mb=min(1024.0, max(32.0,
                    len(agent.completed_tasks) * 8 + len(agent.learned_patterns) * 4)),
                response_time_ms=agent.average_response_time * 1000,
                throughput_per_second=1.0 / max(agent.average_response_time, 0.1),
                error_rate=1.0 - agent.success_rate,
                cost_per_hour=self._calculate_agent_cost_per_hour(agent),
                total_cost=agent.cost_incurred
            )
            
            # Store resource usage
            if agent_id not in self.resource_usage:
                self.resource_usage[agent_id] = []
            
            self.resource_usage[agent_id].append(resource_usage)
            
            # Keep only recent data (last 24 hours)
            cutoff_time = current_time - timedelta(hours=24)
            self.resource_usage[agent_id] = [
                usage for usage in self.resource_usage[agent_id] 
                if usage.timestamp > cutoff_time
            ]
            
            # Save to database
            self._save_resource_usage_to_database(resource_usage)

    def _calculate_agent_cost_per_hour(self, agent: AutonomousAgent) -> float:
        """Calculate estimated cost per hour for agent"""
        
        # Base cost by intelligence level
        intelligence_costs = {
            IntelligenceLevel.MENSA_BASE: 0.10,
            IntelligenceLevel.VERTEX_ENHANCED: 0.25,
            IntelligenceLevel.GRANDMASTER: 0.50,
            IntelligenceLevel.TRANSCENDENT: 1.00,
            IntelligenceLevel.OMNISCIENT: 2.00
        }
        
        base_cost = intelligence_costs.get(agent.current_intelligence_level, 0.25)
        
        # Adjust for usage
        usage_multiplier = 1.0 + (len(agent.assigned_tasks) * 0.1)
        
        # Adjust for performance (better performance = higher cost)
        performance_multiplier = 0.5 + (agent.efficiency_score * 0.5)
        
        return base_cost * usage_multiplier * performance_multiplier

    def _save_resource_usage_to_database(self, usage: ResourceUsage):
        """Save resource usage to database"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO resource_usage 
                (usage_id, agent_id, timestamp, cpu_percent, memory_mb, storage_mb,
                 network_bytes, response_time_ms, throughput_per_second, error_rate,
                 cost_per_hour, total_cost)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                usage.agent_id,
                usage.timestamp.isoformat(),
                usage.cpu_percent,
                usage.memory_mb,
                usage.storage_mb,
                usage.network_bytes,
                usage.response_time_ms,
                usage.throughput_per_second,
                usage.error_rate,
                usage.cost_per_hour,
                usage.total_cost
            ))
            conn.commit()

    def _check_resource_limits(self):
        """Check if agents are exceeding resource limits"""
        
        for agent_id, usage_history in self.resource_usage.items():
            if not usage_history:
                continue
            
            latest_usage = usage_history[-1]
            agent = self.agent_engine.active_agents.get(agent_id)
            
            if not agent:
                continue
            
            policy = self._get_agent_policy(agent)
            
            # Check CPU limit
            if latest_usage.cpu_percent > policy.max_cpu_percent:
                asyncio.create_task(self._handle_resource_violation(
                    agent_id, ResourceType.CPU, latest_usage.cpu_percent, policy.max_cpu_percent
                ))
            
            # Check memory limit
            if latest_usage.memory_mb > policy.max_memory_mb:
                asyncio.create_task(self._handle_resource_violation(
                    agent_id, ResourceType.MEMORY, latest_usage.memory_mb, policy.max_memory_mb
                ))
            
            # Check cost limit
            if latest_usage.cost_per_hour > policy.max_cost_per_hour:
                asyncio.create_task(self._handle_resource_violation(
                    agent_id, ResourceType.CPU, latest_usage.cost_per_hour, policy.max_cost_per_hour
                ))

    async def _handle_resource_violation(self, agent_id: str, resource_type: ResourceType, 
                                       current_value: float, limit_value: float):
        """Handle resource limit violation"""
        
        violation_details = {
            'resource_type': resource_type.value,
            'current_value': current_value,
            'limit_value': limit_value,
            'violation_percentage': (current_value / limit_value - 1.0) * 100
        }
        
        await self._record_lifecycle_event(
            agent_id, LifecycleEvent.PERFORMANCE_UPDATED,
            f"Resource violation: {resource_type.value}",
            violation_details
        )
        
        # Take corrective action
        if violation_details['violation_percentage'] > 50:  # Severe violation
            await self._suspend_agent(agent_id, f"Severe {resource_type.value} violation")
        else:  # Minor violation - optimize
            await self._optimize_agent_performance(agent_id)
        
        self.logger.warning(f"Resource violation for agent {agent_id}: {resource_type.value} = {current_value} (limit: {limit_value})")

    def _monitor_agent_performance(self):
        """Monitor agent performance metrics"""
        
        current_time = datetime.now()
        
        for agent_id, agent in self.agent_engine.active_agents.items():
            # Calculate cost efficiency
            cost_efficiency = 0.0
            if agent.cost_incurred > 0:
                cost_efficiency = (agent.total_tasks_completed * agent.success_rate) / agent.cost_incurred
            
            performance_record = {
                'record_id': str(uuid.uuid4()),
                'agent_id': agent_id,
                'timestamp': current_time.isoformat(),
                'success_rate': agent.success_rate,
                'average_response_time': agent.average_response_time,
                'efficiency_score': agent.efficiency_score,
                'tasks_completed': agent.total_tasks_completed,
                'cost_efficiency': cost_efficiency
            }
            
            # Store performance history
            if agent_id not in self.performance_history:
                self.performance_history[agent_id] = []
            
            self.performance_history[agent_id].append(performance_record)
            
            # Keep only recent data (last 7 days)
            cutoff_time = current_time - timedelta(days=7)
            self.performance_history[agent_id] = [
                record for record in self.performance_history[agent_id]
                if datetime.fromisoformat(record['timestamp']) > cutoff_time
            ]
            
            # Save to database
            self._save_performance_to_database(performance_record)

    def _save_performance_to_database(self, record: Dict[str, Any]):
        """Save performance record to database"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO performance_history 
                (record_id, agent_id, timestamp, success_rate, average_response_time,
                 efficiency_score, tasks_completed, cost_efficiency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record['record_id'],
                record['agent_id'],
                record['timestamp'],
                record['success_rate'],
                record['average_response_time'],
                record['efficiency_score'],
                record['tasks_completed'],
                record['cost_efficiency']
            ))
            conn.commit()

    def _analyze_performance_trends(self):
        """Analyze performance trends and identify issues"""
        
        for agent_id, history in self.performance_history.items():
            if len(history) < 5:  # Need minimum data for trend analysis
                continue
            
            # Analyze trends
            recent_records = history[-5:]
            older_records = history[-10:-5] if len(history) >= 10 else []
            
            if older_records:
                # Calculate trend changes
                recent_avg_success = sum(r['success_rate'] for r in recent_records) / len(recent_records)
                older_avg_success = sum(r['success_rate'] for r in older_records) / len(older_records)
                
                recent_avg_response = sum(r['average_response_time'] for r in recent_records) / len(recent_records)
                older_avg_response = sum(r['average_response_time'] for r in older_records) / len(older_records)
                
                # Check for performance degradation
                success_decline = older_avg_success - recent_avg_success
                response_increase = recent_avg_response - older_avg_response
                
                if success_decline > 0.1 or response_increase > 1.0:  # Significant degradation
                    asyncio.create_task(self._handle_performance_degradation(
                        agent_id, success_decline, response_increase
                    ))

    async def _handle_performance_degradation(self, agent_id: str, success_decline: float, response_increase: float):
        """Handle performance degradation"""
        
        degradation_details = {
            'success_rate_decline': success_decline,
            'response_time_increase': response_increase,
            'degradation_severity': 'high' if success_decline > 0.2 or response_increase > 2.0 else 'medium'
        }
        
        await self._record_lifecycle_event(
            agent_id, LifecycleEvent.PERFORMANCE_UPDATED,
            "Performance degradation detected",
            degradation_details
        )
        
        # Take corrective action based on severity
        if degradation_details['degradation_severity'] == 'high':
            await self._replace_poor_performing_agent(agent_id)
        else:
            await self._optimize_agent_performance(agent_id)
        
        self.logger.warning(f"Performance degradation detected for agent {agent_id}: {degradation_details}")

    def _check_scaling_triggers(self):
        """Check for scaling triggers"""
        
        # Analyze demand by archetype
        archetype_demand = {}
        archetype_agents = {}
        
        for agent in self.agent_engine.active_agents.values():
            archetype = agent.configuration.archetype_specialty.value
            
            if archetype not in archetype_demand:
                archetype_demand[archetype] = 0
                archetype_agents[archetype] = 0
            
            archetype_agents[archetype] += 1
            archetype_demand[archetype] += len(agent.assigned_tasks)
        
        # Check for scaling opportunities
        for archetype, demand in archetype_demand.items():
            agent_count = archetype_agents.get(archetype, 0)
            
            if agent_count == 0:
                continue
            
            avg_demand_per_agent = demand / agent_count
            
            # Scale up if high demand
            if avg_demand_per_agent > 3 and agent_count < 10:  # High demand threshold
                asyncio.create_task(self._scale_up_archetype(
                    ArchetypeSpecialty(archetype), ScalingTrigger.HIGH_DEMAND
                ))
            
            # Scale down if low demand
            elif avg_demand_per_agent < 0.5 and agent_count > 1:  # Low demand threshold
                asyncio.create_task(self._scale_down_archetype(
                    ArchetypeSpecialty(archetype), ScalingTrigger.COST_OPTIMIZATION
                ))

    async def _scale_up_archetype(self, archetype: ArchetypeSpecialty, trigger: ScalingTrigger):
        """Scale up agents for specific archetype"""
        
        current_count = len([
            agent for agent in self.agent_engine.active_agents.values()
            if agent.configuration.archetype_specialty == archetype
        ])
        
        # Create new agent
        new_agent = await self.agent_engine.create_agent(
            AgentType.TEMPORARY,  # Use temporary for scaling
            archetype,
            purpose=f"Scale-up agent for {archetype.value}",
            duration=timedelta(hours=4)  # 4-hour temporary agent
        )
        
        # Record scaling decision
        scaling_decision = {
            'decision_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'trigger_type': trigger.value,
            'archetype_specialty': archetype.value,
            'action_taken': 'scale_up',
            'agents_before': current_count,
            'agents_after': current_count + 1,
            'expected_impact': 'Increased capacity and reduced response time',
            'actual_impact': 'TBD'
        }
        
        self.scaling_decisions.append(scaling_decision)
        self._save_scaling_decision_to_database(scaling_decision)
        
        self.logger.info(f"Scaled up {archetype.value}: {current_count} -> {current_count + 1} agents")

    async def _scale_down_archetype(self, archetype: ArchetypeSpecialty, trigger: ScalingTrigger):
        """Scale down agents for specific archetype"""
        
        # Find least utilized agent of this archetype
        archetype_agents = [
            agent for agent in self.agent_engine.active_agents.values()
            if (agent.configuration.archetype_specialty == archetype and
                agent.configuration.agent_type != AgentType.PERMANENT)
        ]
        
        if not archetype_agents:
            return
        
        # Find agent with lowest utilization
        least_utilized = min(archetype_agents, key=lambda a: len(a.assigned_tasks))
        
        if len(least_utilized.assigned_tasks) == 0:  # Only scale down idle agents
            current_count = len(archetype_agents)
            
            await self._terminate_agent(least_utilized.agent_id, "Scale-down optimization")
            
            # Record scaling decision
            scaling_decision = {
                'decision_id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'trigger_type': trigger.value,
                'archetype_specialty': archetype.value,
                'action_taken': 'scale_down',
                'agents_before': current_count,
                'agents_after': current_count - 1,
                'expected_impact': 'Reduced costs while maintaining capacity',
                'actual_impact': 'TBD'
            }
            
            self.scaling_decisions.append(scaling_decision)
            self._save_scaling_decision_to_database(scaling_decision)
            
            self.logger.info(f"Scaled down {archetype.value}: {current_count} -> {current_count - 1} agents")

    def _save_scaling_decision_to_database(self, decision: Dict[str, Any]):
        """Save scaling decision to database"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO scaling_decisions 
                (decision_id, timestamp, trigger_type, archetype_specialty, action_taken,
                 agents_before, agents_after, expected_impact, actual_impact)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision['decision_id'],
                decision['timestamp'],
                decision['trigger_type'],
                decision['archetype_specialty'],
                decision['action_taken'],
                decision['agents_before'],
                decision['agents_after'],
                decision['expected_impact'],
                decision['actual_impact']
            ))
            conn.commit()

    def _execute_scaling_decisions(self):
        """Execute pending scaling decisions"""
        
        # In production, implement more sophisticated scaling logic
        # This could include:
        # - Predictive scaling based on historical patterns
        # - Cost-aware scaling decisions
        # - Performance-based scaling triggers
        # - Integration with cloud auto-scaling services
        
        pass

    async def _record_lifecycle_event(self, agent_id: str, event_type: LifecycleEvent, 
                                    description: str, metadata: Optional[Dict[str, Any]] = None):
        """Record lifecycle event"""
        
        event = LifecycleEvent(
            event_id=str(uuid.uuid4()),
            agent_id=agent_id,
            event_type=event_type,
            timestamp=datetime.now(),
            description=description,
            metadata=metadata or {}
        )
        
        self.lifecycle_events.append(event)
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO lifecycle_events 
                (event_id, agent_id, event_type, timestamp, description, metadata, triggered_by)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.agent_id,
                event.event_type.value,
                event.timestamp.isoformat(),
                event.description,
                json.dumps(event.metadata, default=str),
                event.triggered_by
            ))
            conn.commit()

    def _process_lifecycle_events(self):
        """Process pending lifecycle events"""
        
        # Process event handlers
        for event in self.lifecycle_events[-100:]:  # Process recent events
            handlers = self.event_handlers.get(event.event_type, [])
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(f"Error in event handler: {str(e)}")

    def _cleanup_expired_data(self):
        """Cleanup expired data"""
        
        cutoff_time = datetime.now() - timedelta(days=30)
        
        # Cleanup old lifecycle events
        self.lifecycle_events = [
            event for event in self.lifecycle_events
            if event.timestamp > cutoff_time
        ]
        
        # Cleanup old resource usage data
        for agent_id in list(self.resource_usage.keys()):
            self.resource_usage[agent_id] = [
                usage for usage in self.resource_usage[agent_id]
                if usage.timestamp > cutoff_time
            ]
            
            if not self.resource_usage[agent_id]:
                del self.resource_usage[agent_id]
        
        # Cleanup old performance history
        for agent_id in list(self.performance_history.keys()):
            self.performance_history[agent_id] = [
                record for record in self.performance_history[agent_id]
                if datetime.fromisoformat(record['timestamp']) > cutoff_time
            ]
            
            if not self.performance_history[agent_id]:
                del self.performance_history[agent_id]

    def get_lifecycle_stats(self) -> Dict[str, Any]:
        """Get lifecycle management statistics"""
        
        # Agent statistics by type
        agent_stats_by_type = {}
        agent_stats_by_archetype = {}
        agent_stats_by_status = {}
        
        for agent in self.agent_engine.active_agents.values():
            # By type
            agent_type = agent.configuration.agent_type.value
            agent_stats_by_type[agent_type] = agent_stats_by_type.get(agent_type, 0) + 1
            
            # By archetype
            archetype = agent.configuration.archetype_specialty.value
            agent_stats_by_archetype[archetype] = agent_stats_by_archetype.get(archetype, 0) + 1
            
            # By status
            status = agent.status.value
            agent_stats_by_status[status] = agent_stats_by_status.get(status, 0) + 1
        
        # Resource utilization
        total_cpu = sum(
            usage[-1].cpu_percent for usage in self.resource_usage.values() if usage
        )
        total_memory = sum(
            usage[-1].memory_mb for usage in self.resource_usage.values() if usage
        )
        total_cost = sum(
            usage[-1].cost_per_hour for usage in self.resource_usage.values() if usage
        )
        
        # Performance metrics
        avg_success_rate = 0.0
        avg_response_time = 0.0
        if self.agent_engine.active_agents:
            avg_success_rate = sum(a.success_rate for a in self.agent_engine.active_agents.values()) / len(self.agent_engine.active_agents)
            avg_response_time = sum(a.average_response_time for a in self.agent_engine.active_agents.values()) / len(self.agent_engine.active_agents)
        
        return {
            'total_active_agents': len(self.agent_engine.active_agents),
            'agents_by_type': agent_stats_by_type,
            'agents_by_archetype': agent_stats_by_archetype,
            'agents_by_status': agent_stats_by_status,
            'resource_utilization': {
                'total_cpu_percent': total_cpu,
                'total_memory_mb': total_memory,
                'total_cost_per_hour': total_cost
            },
            'performance_metrics': {
                'average_success_rate': avg_success_rate,
                'average_response_time_seconds': avg_response_time
            },
            'lifecycle_events_count': len(self.lifecycle_events),
            'scaling_decisions_count': len(self.scaling_decisions),
            'policies_count': len(self.policies)
        }

    def get_agent_lifecycle_details(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed lifecycle information for specific agent"""
        
        if agent_id not in self.agent_engine.active_agents:
            return None
        
        agent = self.agent_engine.active_agents[agent_id]
        
        # Get resource usage history
        resource_history = self.resource_usage.get(agent_id, [])
        
        # Get performance history
        performance_history = self.performance_history.get(agent_id, [])
        
        # Get lifecycle events
        agent_events = [
            asdict(event) for event in self.lifecycle_events
            if event.agent_id == agent_id
        ]
        
        # Get applicable policy
        policy = self._get_agent_policy(agent)
        
        return {
            'agent_id': agent_id,
            'current_status': agent.status.value,
            'agent_type': agent.configuration.agent_type.value,
            'archetype_specialty': agent.configuration.archetype_specialty.value,
            'intelligence_level': agent.current_intelligence_level.value,
            'lifecycle_policy': asdict(policy),
            'resource_usage_history': [asdict(usage) for usage in resource_history[-24:]],  # Last 24 records
            'performance_history': performance_history[-24:],  # Last 24 records
            'lifecycle_events': agent_events[-50:],  # Last 50 events
            'current_performance': {
                'success_rate': agent.success_rate,
                'average_response_time': agent.average_response_time,
                'efficiency_score': agent.efficiency_score,
                'total_tasks_completed': agent.total_tasks_completed
            },
            'lifecycle_timestamps': {
                'created_at': agent.created_at.isoformat(),
                'last_active': agent.last_active.isoformat(),
                'expires_at': agent.expires_at.isoformat() if agent.expires_at else None
            }
        }

    def shutdown(self):
        """Shutdown lifecycle manager"""
        
        self.monitoring_active = False
        
        # Wait for monitoring tasks to complete
        time.sleep(2)
        
        self.logger.info("Agent Lifecycle Manager shutdown complete")

# Main execution for testing
async def main():
    """Main execution function for testing"""
    from agent_generation_engine import AgentGenerationEngine
    
    # Initialize agent engine
    agent_config = {
        'max_concurrent_agents': 100,
        'agent_data_dir': '/app/data/agents'
    }
    agent_engine = AgentGenerationEngine(agent_config)
    
    # Initialize lifecycle manager
    lifecycle_config = {
        'lifecycle_db_path': '/app/data/lifecycle.db'
    }
    lifecycle_manager = AgentLifecycleManager(lifecycle_config, agent_engine)
    
    # Create test agents
    legal_agent = await agent_engine.create_agent(
        AgentType.PERMANENT,
        ArchetypeSpecialty.LEGAL_INTELLIGENCE,
        purpose="Legal analysis and strategy development"
    )
    
    phantom_agent = await agent_engine.create_agent(
        AgentType.PHANTOM,
        ArchetypeSpecialty.CRISIS_MANAGEMENT,
        purpose="Emergency response analysis"
    )
    
    # Wait for monitoring
    await asyncio.sleep(5)
    
    # Get statistics
    stats = lifecycle_manager.get_lifecycle_stats()
    agent_details = lifecycle_manager.get_agent_lifecycle_details(legal_agent.agent_id)
    
    print(f"Lifecycle stats: {stats}")
    print(f"Legal agent details: {agent_details}")
    
    # Shutdown
    lifecycle_manager.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

