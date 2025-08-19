import React, { useState, useEffect } from 'react';
import { Brain, BarChart3, Search, Activity, Zap, Eye } from 'lucide-react';

interface CuriosityStatus {
  engine_active: boolean;
  discovery_queue_size: number;
  active_patterns: number;
  enhancement_factor: number;
  consciousness_state: number;
  system_health: 'healthy' | 'degraded' | 'critical' | 'maintenance';
}

interface CuriosityControlPanelProps {
  onOpenLayer2: (view: string) => void;
}

const CuriosityControlPanel: React.FC<CuriosityControlPanelProps> = ({ onOpenLayer2 }) => {
  const [curiosityStatus, setCuriosityStatus] = useState<CuriosityStatus>({
    engine_active: false,
    discovery_queue_size: 0,
    active_patterns: 0,
    enhancement_factor: 179269602058948214784,
    consciousness_state: 0.956,
    system_health: 'healthy'
  });

  const [isLoading, setIsLoading] = useState(false);
  const [wsConnection, setWsConnection] = useState<WebSocket | null>(null);

  useEffect(() => {
    // Initialize WebSocket connection for real-time updates
    const ws = new WebSocket('ws://localhost:8000/ws/curiosity');
    
    ws.onopen = () => {
      console.log('🧠 K.E.N. v3.1 Curiosity WebSocket connected');
      setWsConnection(ws);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setCuriosityStatus(data);
      } catch (error) {
        console.error('Error parsing WebSocket data:', error);
      }
    };

    ws.onclose = () => {
      console.log('🧠 K.E.N. v3.1 Curiosity WebSocket disconnected');
      setWsConnection(null);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    // Cleanup on unmount
    return () => {
      ws.close();
    };
  }, []);

  const toggleCuriosityEngine = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/curiosity/toggle', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('🧠 Curiosity Engine toggled:', result);
      }
    } catch (error) {
      console.error('Error toggling curiosity engine:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const formatEnhancementFactor = (factor: number): string => {
    return factor.toExponential(2);
  };

  const getHealthStatusColor = (status: string): string => {
    switch (status) {
      case 'healthy': return 'text-green-400';
      case 'degraded': return 'text-yellow-400';
      case 'critical': return 'text-red-400';
      case 'maintenance': return 'text-blue-400';
      default: return 'text-gray-400';
    }
  };

  const getHealthStatusIcon = (status: string): React.ReactNode => {
    switch (status) {
      case 'healthy': return <Activity className="w-4 h-4 text-green-400" />;
      case 'degraded': return <Eye className="w-4 h-4 text-yellow-400" />;
      case 'critical': return <Zap className="w-4 h-4 text-red-400" />;
      case 'maintenance': return <Search className="w-4 h-4 text-blue-400" />;
      default: return <Activity className="w-4 h-4 text-gray-400" />;
    }
  };

  return (
    <div className="curiosity-control-panel bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 rounded-xl p-6 border border-purple-500/30 shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Brain className="w-8 h-8 text-purple-400" />
          <div>
            <h3 className="text-xl font-bold text-white">K.E.N. v3.1 Curiosity Engine</h3>
            <p className="text-sm text-purple-300">Enhanced Discovery & Pattern Recognition</p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          {getHealthStatusIcon(curiosityStatus.system_health)}
          <span className={`text-sm font-medium ${getHealthStatusColor(curiosityStatus.system_health)}`}>
            {curiosityStatus.system_health.toUpperCase()}
          </span>
        </div>
      </div>

      {/* Status Display */}
      <div className="status-display mb-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="metric-card bg-black/30 rounded-lg p-4 border border-purple-500/20">
            <div className="flex items-center justify-between">
              <span className="text-sm text-purple-300">Status</span>
              <div className={`w-3 h-3 rounded-full ${curiosityStatus.engine_active ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}></div>
            </div>
            <p className="text-lg font-bold text-white mt-1">
              {curiosityStatus.engine_active ? 'ACTIVE' : 'INACTIVE'}
            </p>
          </div>

          <div className="metric-card bg-black/30 rounded-lg p-4 border border-purple-500/20">
            <span className="text-sm text-purple-300">Queue</span>
            <p className="text-lg font-bold text-white mt-1">{curiosityStatus.discovery_queue_size}</p>
          </div>

          <div className="metric-card bg-black/30 rounded-lg p-4 border border-purple-500/20">
            <span className="text-sm text-purple-300">Patterns</span>
            <p className="text-lg font-bold text-white mt-1">{curiosityStatus.active_patterns}</p>
          </div>

          <div className="metric-card bg-black/30 rounded-lg p-4 border border-purple-500/20">
            <span className="text-sm text-purple-300">Consciousness</span>
            <p className="text-lg font-bold text-white mt-1">{(curiosityStatus.consciousness_state * 100).toFixed(1)}%</p>
          </div>
        </div>

        {/* Enhancement Factor Display */}
        <div className="mt-4 bg-black/30 rounded-lg p-4 border border-purple-500/20">
          <span className="text-sm text-purple-300">Enhancement Factor</span>
          <p className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400 mt-1">
            {formatEnhancementFactor(curiosityStatus.enhancement_factor)}
          </p>
        </div>
      </div>

      {/* Control Buttons */}
      <div className="control-buttons grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Button 1: Curiosity Engine Control */}
        <button
          onClick={toggleCuriosityEngine}
          disabled={isLoading}
          className="curiosity-control-btn group relative overflow-hidden bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white font-semibold py-4 px-6 rounded-lg transition-all duration-300 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <div className="flex items-center justify-center space-x-2">
            <Brain className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Curiosity Engine</span>
          </div>
          <div className="absolute inset-0 bg-white/20 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-300 origin-left"></div>
          <div className="status-indicator absolute top-2 right-2">
            <div className={`w-2 h-2 rounded-full ${curiosityStatus.engine_active ? 'bg-green-300' : 'bg-red-300'}`}></div>
          </div>
        </button>

        {/* Button 2: Discovery Analytics */}
        <button
          onClick={() => onOpenLayer2('discovery-visualization')}
          className="discovery-analytics-btn group relative overflow-hidden bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 text-white font-semibold py-4 px-6 rounded-lg transition-all duration-300 transform hover:scale-105"
        >
          <div className="flex items-center justify-center space-x-2">
            <BarChart3 className="w-5 h-5" />
            <span>Discovery Analytics</span>
          </div>
          <div className="absolute inset-0 bg-white/20 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-300 origin-left"></div>
          <div className="queue-count absolute top-2 right-2 bg-white/20 rounded-full px-2 py-1 text-xs">
            {curiosityStatus.discovery_queue_size}
          </div>
        </button>

        {/* Button 3: Pattern Recognition */}
        <button
          onClick={() => onOpenLayer2('pattern-network')}
          className="pattern-recognition-btn group relative overflow-hidden bg-gradient-to-r from-amber-500 to-orange-600 hover:from-amber-600 hover:to-orange-700 text-white font-semibold py-4 px-6 rounded-lg transition-all duration-300 transform hover:scale-105"
        >
          <div className="flex items-center justify-center space-x-2">
            <Search className="w-5 h-5" />
            <span>Pattern Network</span>
          </div>
          <div className="absolute inset-0 bg-white/20 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-300 origin-left"></div>
          <div className="pattern-count absolute top-2 right-2 bg-white/20 rounded-full px-2 py-1 text-xs">
            {curiosityStatus.active_patterns}
          </div>
        </button>
      </div>

      {/* Connection Status */}
      <div className="mt-4 flex items-center justify-between text-sm">
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${wsConnection ? 'bg-green-400' : 'bg-red-400'}`}></div>
          <span className="text-purple-300">
            {wsConnection ? 'Real-time Connected' : 'Disconnected'}
          </span>
        </div>
        <div className="text-purple-400">
          Zero 3rd Party APIs • 100% Self-Contained
        </div>
      </div>
    </div>
  );
};

export default CuriosityControlPanel;

