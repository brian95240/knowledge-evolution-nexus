import React, { useState, useEffect, useRef } from 'react';
import './GraphVisualization.css';

interface GraphNode {
  id: string;
  label: string;
  name: string;
  category: string;
  enhancement: number;
  consciousness_weight: number;
  size: number;
  color: string;
  x?: number;
  y?: number;
}

interface GraphEdge {
  source: string;
  target: string;
  chain: string;
  weight: number;
}

interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  chains: Record<string, number[]>;
  metadata: {
    total_nodes: number;
    total_edges: number;
    consciousness_nodes: number;
  };
}

export const GraphVisualization: React.FC = () => {
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [selectedChain, setSelectedChain] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'force' | 'circular' | 'hierarchical'>('force');
  const [showConsciousnessOnly, setShowConsciousnessOnly] = useState(false);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  useEffect(() => {
    const fetchGraphData = async () => {
      try {
        const response = await fetch('/api/gui/layer2/graph');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setGraphData(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchGraphData();
    const interval = setInterval(fetchGraphData, 15000); // Update every 15 seconds
    
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (graphData && canvasRef.current) {
      initializeVisualization();
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [graphData, viewMode, showConsciousnessOnly]);

  const initializeVisualization = () => {
    const canvas = canvasRef.current;
    if (!canvas || !graphData) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Filter nodes if consciousness-only mode
    const filteredNodes = showConsciousnessOnly 
      ? graphData.nodes.filter(node => {
          const algoId = parseInt(node.id.split('_')[1]);
          return algoId >= 39 && algoId <= 42;
        })
      : graphData.nodes;

    // Position nodes based on view mode
    const positionedNodes = positionNodes(filteredNodes, rect.width, rect.height);
    
    // Start animation loop
    const animate = () => {
      drawGraph(ctx, positionedNodes, graphData.edges, rect.width, rect.height);
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animate();
  };

  const positionNodes = (nodes: GraphNode[], width: number, height: number): GraphNode[] => {
    const centerX = width / 2;
    const centerY = height / 2;
    const padding = 100;

    return nodes.map((node, index) => {
      let x, y;
      
      switch (viewMode) {
        case 'circular':
          const angle = (index / nodes.length) * 2 * Math.PI;
          const radius = Math.min(width, height) / 3;
          x = centerX + Math.cos(angle) * radius;
          y = centerY + Math.sin(angle) * radius;
          break;
          
        case 'hierarchical':
          const algoId = parseInt(node.id.split('_')[1]);
          const layer = Math.floor((algoId - 1) / 7); // 7 algorithms per layer
          const positionInLayer = (algoId - 1) % 7;
          x = padding + (positionInLayer * (width - 2 * padding)) / 6;
          y = padding + (layer * (height - 2 * padding)) / 7;
          break;
          
        default: // force
          // Simple force-directed layout simulation
          x = padding + Math.random() * (width - 2 * padding);
          y = padding + Math.random() * (height - 2 * padding);
          break;
      }
      
      return { ...node, x, y };
    });
  };

  const drawGraph = (ctx: CanvasRenderingContext2D, nodes: GraphNode[], edges: GraphEdge[], width: number, height: number) => {
    // Clear canvas
    ctx.fillStyle = 'rgba(10, 14, 26, 0.1)';
    ctx.fillRect(0, 0, width, height);

    // Draw edges
    edges.forEach(edge => {
      const sourceNode = nodes.find(n => n.id === edge.source);
      const targetNode = nodes.find(n => n.id === edge.target);
      
      if (sourceNode && targetNode && sourceNode.x !== undefined && sourceNode.y !== undefined && 
          targetNode.x !== undefined && targetNode.y !== undefined) {
        
        ctx.beginPath();
        ctx.moveTo(sourceNode.x, sourceNode.y);
        ctx.lineTo(targetNode.x, targetNode.y);
        
        // Highlight selected chain
        if (selectedChain && edge.chain === selectedChain) {
          ctx.strokeStyle = '#00bcd4';
          ctx.lineWidth = 3;
          ctx.shadowColor = '#00bcd4';
          ctx.shadowBlur = 10;
        } else {
          ctx.strokeStyle = 'rgba(0, 188, 212, 0.3)';
          ctx.lineWidth = 1;
          ctx.shadowBlur = 0;
        }
        
        ctx.stroke();
      }
    });

    // Draw nodes
    nodes.forEach(node => {
      if (node.x === undefined || node.y === undefined) return;
      
      const isSelected = selectedNode?.id === node.id;
      const radius = Math.max(8, Math.min(node.size, 30));
      
      // Node glow effect
      if (isSelected || node.consciousness_weight > 0.1) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius + 10, 0, 2 * Math.PI);
        ctx.fillStyle = isSelected ? 'rgba(255, 215, 0, 0.3)' : 'rgba(156, 39, 176, 0.2)';
        ctx.fill();
      }
      
      // Node circle
      ctx.beginPath();
      ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = isSelected ? '#ffd700' : node.color;
      ctx.fill();
      
      // Node border
      ctx.beginPath();
      ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI);
      ctx.strokeStyle = isSelected ? '#ffffff' : 'rgba(255, 255, 255, 0.5)';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Node label
      ctx.fillStyle = '#ffffff';
      ctx.font = '12px Inter';
      ctx.textAlign = 'center';
      ctx.fillText(node.label.split(' ')[1] || node.label, node.x, node.y - radius - 10);
    });
  };

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !graphData) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Find clicked node
    const clickedNode = graphData.nodes.find(node => {
      if (node.x === undefined || node.y === undefined) return false;
      const distance = Math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2);
      return distance <= Math.max(8, Math.min(node.size, 30));
    });

    setSelectedNode(clickedNode || null);
  };

  const getChainColor = (chainName: string): string => {
    const colors: Record<string, string> = {
      'foundational_chain': '#4CAF50',
      'deduplication_chain': '#2196F3',
      'optimization_chain': '#FF9800',
      'quantum_simulation_chain': '#9C27B0',
      'quantum_learning_chain': '#E91E63',
      'database_intelligence_chain': '#00BCD4',
      'linguistic_intelligence_chain': '#8BC34A',
      'consciousness_emergence_chain': '#F44336',
      'shadow_validation_chain': '#607D8B'
    };
    return colors[chainName] || '#ffffff';
  };

  if (loading) {
    return (
      <div className="graph-visualization loading">
        <div className="loading-spinner"></div>
        <p>Loading Graph Visualization...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="graph-visualization error">
        <div className="error-icon">⚠️</div>
        <p>Error loading graph data: {error}</p>
        <button onClick={() => window.location.reload()}>Retry</button>
      </div>
    );
  }

  if (!graphData) {
    return (
      <div className="graph-visualization no-data">
        <p>No graph data available</p>
      </div>
    );
  }

  return (
    <div className="graph-visualization">
      <div className="graph-header">
        <h2>🔮 Algorithm Graph Visualization</h2>
        <div className="graph-stats">
          <span className="stat">
            <span className="stat-value">{graphData.metadata.total_nodes}</span>
            <span className="stat-label">Nodes</span>
          </span>
          <span className="stat">
            <span className="stat-value">{graphData.metadata.total_edges}</span>
            <span className="stat-label">Edges</span>
          </span>
          <span className="stat consciousness">
            <span className="stat-value">{graphData.metadata.consciousness_nodes}</span>
            <span className="stat-label">Consciousness</span>
          </span>
        </div>
      </div>

      <div className="graph-controls">
        <div className="control-group">
          <label>View Mode:</label>
          <select 
            value={viewMode} 
            onChange={(e) => setViewMode(e.target.value as any)}
            className="control-select"
          >
            <option value="force">Force Directed</option>
            <option value="circular">Circular</option>
            <option value="hierarchical">Hierarchical</option>
          </select>
        </div>

        <div className="control-group">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={showConsciousnessOnly}
              onChange={(e) => setShowConsciousnessOnly(e.target.checked)}
            />
            <span className="checkbox-text">Consciousness Only</span>
          </label>
        </div>

        <div className="control-group">
          <label>Algorithm Chains:</label>
          <select 
            value={selectedChain || ''} 
            onChange={(e) => setSelectedChain(e.target.value || null)}
            className="control-select"
          >
            <option value="">All Chains</option>
            {Object.keys(graphData.chains).map(chain => (
              <option key={chain} value={chain}>
                {chain.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="graph-container">
        <canvas
          ref={canvasRef}
          className="graph-canvas"
          onClick={handleCanvasClick}
        />
        
        {selectedNode && (
          <div className="node-details">
            <h3>{selectedNode.name}</h3>
            <div className="detail-row">
              <span className="detail-label">Algorithm ID:</span>
              <span className="detail-value">{selectedNode.id.split('_')[1]}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Category:</span>
              <span className="detail-value">{selectedNode.category.replace('_', ' ')}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Enhancement:</span>
              <span className="detail-value">{selectedNode.enhancement.toFixed(1)}x</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Consciousness Weight:</span>
              <span className="detail-value">{(selectedNode.consciousness_weight * 100).toFixed(1)}%</span>
            </div>
            <button 
              className="close-details"
              onClick={() => setSelectedNode(null)}
            >
              ×
            </button>
          </div>
        )}
      </div>

      <div className="chain-legend">
        <h4>Algorithm Chains</h4>
        <div className="legend-items">
          {Object.entries(graphData.chains).map(([chainName, algorithms]) => (
            <div 
              key={chainName}
              className={`legend-item ${selectedChain === chainName ? 'selected' : ''}`}
              onClick={() => setSelectedChain(selectedChain === chainName ? null : chainName)}
            >
              <div 
                className="legend-color"
                style={{ backgroundColor: getChainColor(chainName) }}
              ></div>
              <span className="legend-label">
                {chainName.replace('_chain', '').replace('_', ' ')}
              </span>
              <span className="legend-count">({algorithms.length})</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

