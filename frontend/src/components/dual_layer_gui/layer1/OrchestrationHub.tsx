import React, { useState, useEffect } from 'react';
import './OrchestrationHub.css';

interface SystemStatus {
  enhancement_factor: number;
  consciousness_level: number;
  algorithm_42_active: boolean;
  uptime: number;
}

interface ResourceMonitoring {
  cpu_usage: number;
  memory_usage: number;
  cache_hit_rate: number;
  auto_scale_status: string;
}

interface CostTracking {
  current_monthly: number;
  target_monthly: number;
  enhancement_per_euro: number;
}

interface FibonacciScaling {
  adherence_ratio: number;
  growth_pattern: string;
  scaling_efficiency: number;
}

interface OrchestrationData {
  system_status: SystemStatus;
  resource_monitoring: ResourceMonitoring;
  cost_tracking: CostTracking;
  fibonacci_scaling: FibonacciScaling;
}

export const OrchestrationHub: React.FC = () => {
  const [data, setData] = useState<OrchestrationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchOrchestrationData = async () => {
      try {
        const response = await fetch('/api/gui/layer1/orchestration');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const orchestrationData = await response.json();
        setData(orchestrationData);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchOrchestrationData();
    
    // Update every 5 seconds
    const interval = setInterval(fetchOrchestrationData, 5000);
    
    return () => clearInterval(interval);
  }, []);

  const formatNumber = (num: number): string => {
    if (num >= 1e6) {
      return (num / 1e6).toFixed(1) + 'M';
    } else if (num >= 1e3) {
      return (num / 1e3).toFixed(1) + 'K';
    }
    return num.toFixed(1);
  };

  const formatUptime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  if (loading) {
    return (
      <div className="orchestration-hub loading">
        <div className="loading-spinner"></div>
        <p>Loading K.E.N. Orchestration Hub...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="orchestration-hub error">
        <div className="error-icon">⚠️</div>
        <p>Error loading orchestration data: {error}</p>
        <button onClick={() => window.location.reload()}>Retry</button>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="orchestration-hub no-data">
        <p>No orchestration data available</p>
      </div>
    );
  }

  return (
    <div className="orchestration-hub">
      <div className="hub-header">
        <h2>🧠 K.E.N. Orchestration Hub</h2>
        <div className="status-indicator">
          <span className={`status-dot ${data.system_status.algorithm_42_active ? 'active' : 'inactive'}`}></span>
          <span className="status-text">
            {data.system_status.algorithm_42_active ? 'Consciousness Active' : 'System Operational'}
          </span>
        </div>
      </div>

      <div className="hub-grid">
        {/* System Status Panel */}
        <div className="panel system-status">
          <div className="panel-header">
            <h3>🎯 System Status</h3>
            <div className="uptime">Uptime: {formatUptime(data.system_status.uptime)}</div>
          </div>
          <div className="panel-content">
            <div className="metric-row">
              <div className="metric">
                <div className="metric-label">Enhancement Factor</div>
                <div className="metric-value enhancement">
                  {formatNumber(data.system_status.enhancement_factor)}x
                </div>
                <div className="metric-target">Target: 2.1M</div>
              </div>
              <div className="metric">
                <div className="metric-label">Consciousness Level</div>
                <div className="metric-value consciousness">
                  {(data.system_status.consciousness_level * 100).toFixed(1)}%
                </div>
                <div className="metric-target">Target: 94.3%</div>
              </div>
            </div>
            <div className="algorithm-42-status">
              <div className={`algorithm-42 ${data.system_status.algorithm_42_active ? 'active' : 'inactive'}`}>
                <span className="algorithm-icon">✨</span>
                <span className="algorithm-text">
                  Algorithm 42: {data.system_status.algorithm_42_active ? 'TRANSCENDENT' : 'PREPARING'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Resource Monitoring Panel */}
        <div className="panel resource-monitoring">
          <div className="panel-header">
            <h3>📊 Resource Monitoring</h3>
            <div className="auto-scale-status">
              Auto-Scale: {data.resource_monitoring.auto_scale_status}
            </div>
          </div>
          <div className="panel-content">
            <div className="resource-meters">
              <div className="resource-meter">
                <div className="meter-label">CPU Usage</div>
                <div className="meter-bar">
                  <div 
                    className="meter-fill cpu"
                    style={{ width: `${data.resource_monitoring.cpu_usage}%` }}
                  ></div>
                </div>
                <div className="meter-value">{data.resource_monitoring.cpu_usage.toFixed(1)}%</div>
              </div>
              
              <div className="resource-meter">
                <div className="meter-label">Memory Usage</div>
                <div className="meter-bar">
                  <div 
                    className="meter-fill memory"
                    style={{ width: `${data.resource_monitoring.memory_usage}%` }}
                  ></div>
                </div>
                <div className="meter-value">{data.resource_monitoring.memory_usage.toFixed(1)}%</div>
              </div>
              
              <div className="resource-meter">
                <div className="meter-label">Cache Hit Rate</div>
                <div className="meter-bar">
                  <div 
                    className="meter-fill cache"
                    style={{ width: `${data.resource_monitoring.cache_hit_rate}%` }}
                  ></div>
                </div>
                <div className="meter-value">{data.resource_monitoring.cache_hit_rate.toFixed(1)}%</div>
              </div>
            </div>
          </div>
        </div>

        {/* Cost Tracking Panel */}
        <div className="panel cost-tracking">
          <div className="panel-header">
            <h3>💰 Cost Tracking</h3>
            <div className="cost-status">
              {data.cost_tracking.current_monthly <= data.cost_tracking.target_monthly ? '✅' : '⚠️'}
            </div>
          </div>
          <div className="panel-content">
            <div className="cost-metrics">
              <div className="cost-metric">
                <div className="cost-label">Monthly Cost</div>
                <div className="cost-value current">
                  €{data.cost_tracking.current_monthly.toFixed(2)}
                </div>
                <div className="cost-target">Target: €{data.cost_tracking.target_monthly.toFixed(2)}</div>
              </div>
              <div className="cost-metric">
                <div className="cost-label">Enhancement per €</div>
                <div className="cost-value efficiency">
                  {formatNumber(data.cost_tracking.enhancement_per_euro)}x
                </div>
                <div className="cost-efficiency">Cost Efficiency</div>
              </div>
            </div>
          </div>
        </div>

        {/* Fibonacci Scaling Panel */}
        <div className="panel fibonacci-scaling">
          <div className="panel-header">
            <h3>🌀 Fibonacci Scaling</h3>
            <div className="scaling-pattern">
              Pattern: {data.fibonacci_scaling.growth_pattern}
            </div>
          </div>
          <div className="panel-content">
            <div className="fibonacci-metrics">
              <div className="fibonacci-metric">
                <div className="fibonacci-label">Adherence Ratio</div>
                <div className="fibonacci-value">
                  {(data.fibonacci_scaling.adherence_ratio * 100).toFixed(1)}%
                </div>
                <div className="fibonacci-bar">
                  <div 
                    className="fibonacci-fill"
                    style={{ width: `${data.fibonacci_scaling.adherence_ratio * 100}%` }}
                  ></div>
                </div>
              </div>
              <div className="fibonacci-metric">
                <div className="fibonacci-label">Scaling Efficiency</div>
                <div className="fibonacci-value">
                  {data.fibonacci_scaling.scaling_efficiency.toFixed(1)}%
                </div>
                <div className="fibonacci-sequence">1, 1, 2, 3, 5, 8, 13...</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="quick-actions">
        <button className="action-button primary">
          🚀 Execute Full System
        </button>
        <button className="action-button secondary">
          🔧 Optimize Resources
        </button>
        <button className="action-button tertiary">
          📊 View Analytics
        </button>
      </div>
    </div>
  );
};

