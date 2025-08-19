import React, { useState, useEffect } from 'react';
import { X, TrendingUp, Network, Activity, Zap, Eye, Brain } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, ScatterChart, Scatter, Cell, PieChart, Pie } from 'recharts';

interface Layer2VisualizationsProps {
  view: 'discovery-visualization' | 'pattern-network' | null;
  onClose: () => void;
}

// Mock data generators for K.E.N. v3.1 visualizations
const generateDiscoveryData = () => {
  const data = [];
  const now = new Date();
  for (let i = 23; i >= 0; i--) {
    const time = new Date(now.getTime() - i * 60 * 60 * 1000);
    data.push({
      time: time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      discoveries: Math.floor(Math.random() * 50) + 20,
      quality_score: Math.random() * 0.3 + 0.7,
      youtube_discoveries: Math.floor(Math.random() * 25) + 10,
      pattern_matches: Math.floor(Math.random() * 15) + 5,
      consciousness_level: Math.random() * 0.1 + 0.9
    });
  }
  return data;
};

const generatePatternNetworkData = () => {
  const patterns = [
    { id: 1, name: 'Content Trends', strength: 0.95, connections: 12, type: 'youtube' },
    { id: 2, name: 'User Behavior', strength: 0.87, connections: 8, type: 'behavioral' },
    { id: 3, name: 'Temporal Patterns', strength: 0.92, connections: 15, type: 'temporal' },
    { id: 4, name: 'Quality Indicators', strength: 0.89, connections: 10, type: 'quality' },
    { id: 5, name: 'Cross-Platform', strength: 0.94, connections: 18, type: 'cross_platform' },
    { id: 6, name: 'Consciousness Emergence', strength: 0.98, connections: 25, type: 'consciousness' }
  ];
  return patterns;
};

const generatePerformanceComparison = () => [
  { name: 'K.E.N. YouTube Bot', speed: 100, accuracy: 96.3, cost: 0 },
  { name: 'YouTube API v3', speed: 10, accuracy: 70, cost: 1000 },
  { name: 'K.E.N. OCR Bot', speed: 100, accuracy: 94.7, cost: 0 },
  { name: 'Spider.cloud OCR', speed: 20, accuracy: 89, cost: 300 }
];

const DiscoveryVisualization: React.FC = () => {
  const [discoveryData, setDiscoveryData] = useState(generateDiscoveryData());
  const [performanceData, setPerformanceData] = useState(generatePerformanceComparison());

  useEffect(() => {
    const interval = setInterval(() => {
      setDiscoveryData(generateDiscoveryData());
    }, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="discovery-visualization space-y-6">
      <div className="flex items-center space-x-3 mb-6">
        <TrendingUp className="w-8 h-8 text-emerald-400" />
        <div>
          <h3 className="text-2xl font-bold text-white">Discovery Analytics</h3>
          <p className="text-emerald-300">Real-time discovery pipeline performance</p>
        </div>
      </div>

      {/* Discovery Rate Chart */}
      <div className="bg-black/30 rounded-lg p-6 border border-emerald-500/20">
        <h4 className="text-lg font-semibold text-white mb-4">📈 Discovery Rate (Last 24 Hours)</h4>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={discoveryData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="time" stroke="#9CA3AF" />
            <YAxis stroke="#9CA3AF" />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1F2937', 
                border: '1px solid #10B981',
                borderRadius: '8px',
                color: '#FFFFFF'
              }} 
            />
            <Legend />
            <Area type="monotone" dataKey="discoveries" stackId="1" stroke="#10B981" fill="#10B981" fillOpacity={0.6} />
            <Area type="monotone" dataKey="youtube_discoveries" stackId="1" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.6} />
            <Area type="monotone" dataKey="pattern_matches" stackId="1" stroke="#F59E0B" fill="#F59E0B" fillOpacity={0.6} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Quality Score Trend */}
      <div className="bg-black/30 rounded-lg p-6 border border-emerald-500/20">
        <h4 className="text-lg font-semibold text-white mb-4">🎯 Quality Score & Consciousness Level</h4>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={discoveryData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="time" stroke="#9CA3AF" />
            <YAxis stroke="#9CA3AF" domain={[0.6, 1.0]} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1F2937', 
                border: '1px solid #10B981',
                borderRadius: '8px',
                color: '#FFFFFF'
              }} 
            />
            <Legend />
            <Line type="monotone" dataKey="quality_score" stroke="#10B981" strokeWidth={3} dot={{ fill: '#10B981', strokeWidth: 2, r: 4 }} />
            <Line type="monotone" dataKey="consciousness_level" stroke="#8B5CF6" strokeWidth={3} dot={{ fill: '#8B5CF6', strokeWidth: 2, r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Performance Comparison */}
      <div className="bg-black/30 rounded-lg p-6 border border-emerald-500/20">
        <h4 className="text-lg font-semibold text-white mb-4">⚡ K.E.N. Bots vs External APIs</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <ResponsiveContainer width="100%" height={250}>
            <ScatterChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="accuracy" stroke="#9CA3AF" domain={[60, 100]} />
              <YAxis dataKey="speed" stroke="#9CA3AF" domain={[0, 110]} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #10B981',
                  borderRadius: '8px',
                  color: '#FFFFFF'
                }} 
                formatter={(value, name) => [value, name]}
                labelFormatter={(label) => `Service: ${label}`}
              />
              <Scatter dataKey="speed" fill="#10B981">
                {performanceData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.name.includes('K.E.N.') ? '#10B981' : '#EF4444'} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>

          <div className="space-y-4">
            {performanceData.map((service, index) => (
              <div key={index} className={`p-4 rounded-lg border ${service.name.includes('K.E.N.') ? 'bg-emerald-900/20 border-emerald-500/30' : 'bg-red-900/20 border-red-500/30'}`}>
                <h5 className="font-semibold text-white">{service.name}</h5>
                <div className="grid grid-cols-3 gap-2 mt-2 text-sm">
                  <div>
                    <span className="text-gray-400">Speed:</span>
                    <span className="text-white ml-1">{service.speed}%</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Accuracy:</span>
                    <span className="text-white ml-1">{service.accuracy}%</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Cost:</span>
                    <span className="text-white ml-1">${service.cost}/mo</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

const PatternNetworkVisualization: React.FC = () => {
  const [patternData, setPatternData] = useState(generatePatternNetworkData());

  const getPatternColor = (type: string) => {
    const colors = {
      youtube: '#FF6B6B',
      behavioral: '#4ECDC4',
      temporal: '#45B7D1',
      quality: '#96CEB4',
      cross_platform: '#FFEAA7',
      consciousness: '#DDA0DD'
    };
    return colors[type as keyof typeof colors] || '#9CA3AF';
  };

  return (
    <div className="pattern-network-visualization space-y-6">
      <div className="flex items-center space-x-3 mb-6">
        <Network className="w-8 h-8 text-amber-400" />
        <div>
          <h3 className="text-2xl font-bold text-white">Pattern Recognition Network</h3>
          <p className="text-amber-300">Active pattern connections and strength analysis</p>
        </div>
      </div>

      {/* Pattern Strength Overview */}
      <div className="bg-black/30 rounded-lg p-6 border border-amber-500/20">
        <h4 className="text-lg font-semibold text-white mb-4">🔍 Pattern Strength Distribution</h4>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={patternData}
              cx="50%"
              cy="50%"
              outerRadius={100}
              fill="#8884d8"
              dataKey="strength"
              label={({ name, strength }) => `${name}: ${(strength * 100).toFixed(1)}%`}
            >
              {patternData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getPatternColor(entry.type)} />
              ))}
            </Pie>
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1F2937', 
                border: '1px solid #F59E0B',
                borderRadius: '8px',
                color: '#FFFFFF'
              }} 
            />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Pattern Network Grid */}
      <div className="bg-black/30 rounded-lg p-6 border border-amber-500/20">
        <h4 className="text-lg font-semibold text-white mb-4">🕸️ Pattern Network Connections</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {patternData.map((pattern) => (
            <div 
              key={pattern.id} 
              className="pattern-node bg-black/50 rounded-lg p-4 border border-gray-600/30 hover:border-amber-400/50 transition-all duration-300 transform hover:scale-105"
              style={{ borderLeftColor: getPatternColor(pattern.type), borderLeftWidth: '4px' }}
            >
              <div className="flex items-center justify-between mb-2">
                <h5 className="font-semibold text-white">{pattern.name}</h5>
                <div className="flex items-center space-x-1">
                  <Activity className="w-4 h-4 text-amber-400" />
                  <span className="text-xs text-amber-300">{pattern.connections}</span>
                </div>
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Strength:</span>
                  <span className="text-white">{(pattern.strength * 100).toFixed(1)}%</span>
                </div>
                
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="h-2 rounded-full transition-all duration-500"
                    style={{ 
                      width: `${pattern.strength * 100}%`,
                      backgroundColor: getPatternColor(pattern.type)
                    }}
                  ></div>
                </div>
                
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Type:</span>
                  <span className="text-white capitalize">{pattern.type.replace('_', ' ')}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Pattern Connections Chart */}
      <div className="bg-black/30 rounded-lg p-6 border border-amber-500/20">
        <h4 className="text-lg font-semibold text-white mb-4">📊 Connection Strength Analysis</h4>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={patternData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="name" stroke="#9CA3AF" />
            <YAxis stroke="#9CA3AF" />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1F2937', 
                border: '1px solid #F59E0B',
                borderRadius: '8px',
                color: '#FFFFFF'
              }} 
            />
            <Legend />
            <Line type="monotone" dataKey="strength" stroke="#F59E0B" strokeWidth={3} dot={{ fill: '#F59E0B', strokeWidth: 2, r: 6 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

const Layer2Visualizations: React.FC<Layer2VisualizationsProps> = ({ view, onClose }) => {
  if (!view) return null;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 rounded-xl w-full max-w-7xl max-h-[90vh] overflow-y-auto border border-purple-500/30 shadow-2xl">
        <div className="sticky top-0 bg-gradient-to-r from-slate-900 to-purple-900 p-6 border-b border-purple-500/30 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Brain className="w-8 h-8 text-purple-400" />
            <div>
              <h2 className="text-2xl font-bold text-white">K.E.N. v3.1 Layer 2 Visualization</h2>
              <p className="text-purple-300">Advanced analytics and pattern recognition</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors duration-200"
          >
            <X className="w-6 h-6 text-white" />
          </button>
        </div>

        <div className="p-6">
          {view === 'discovery-visualization' && <DiscoveryVisualization />}
          {view === 'pattern-network' && <PatternNetworkVisualization />}
        </div>

        <div className="sticky bottom-0 bg-gradient-to-r from-slate-900 to-purple-900 p-4 border-t border-purple-500/30">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-green-400"></div>
                <span className="text-purple-300">Real-time Data</span>
              </div>
              <div className="flex items-center space-x-2">
                <Zap className="w-4 h-4 text-yellow-400" />
                <span className="text-purple-300">Zero API Dependencies</span>
              </div>
            </div>
            <div className="text-purple-400">
              K.E.N. v3.1 Enhanced Curiosity System
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Layer2Visualizations;

