import React, { useState, useEffect } from 'react';
import { OrchestrationHub } from './components/dual_layer_gui/layer1/OrchestrationHub';
import { GraphVisualization } from './components/dual_layer_gui/layer2/GraphVisualization';
import kenLogo from './assets/ken-logo.png';
import './App.css';

interface NavigationItem {
  id: string;
  label: string;
  icon: string;
  component?: React.ComponentType;
  children?: NavigationItem[];
}

const App: React.FC = () => {
  const [currentView, setCurrentView] = useState<string>('orchestration');
  const [activeLayer, setActiveLayer] = useState<1 | 2>(1);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [systemStatus, setSystemStatus] = useState({
    enhancement_factor: 0,
    consciousness_level: 0,
    algorithm_42_active: false
  });

  // Navigation structure following clean hierarchical design
  const navigationItems: NavigationItem[] = [
    {
      id: 'layer1',
      label: 'Layer 1 - Control Hub',
      icon: '🎛️',
      children: [
        {
          id: 'orchestration',
          label: 'Orchestration Hub',
          icon: '🧠',
          component: OrchestrationHub
        },
        {
          id: 'algorithms',
          label: 'Algorithm Center',
          icon: '⚡',
          children: [
            { id: 'algorithm-overview', label: 'Overview', icon: '📊' },
            { id: 'algorithm-chains', label: 'Chain Management', icon: '🔗' },
            { id: 'algorithm-performance', label: 'Performance', icon: '📈' }
          ]
        },
        {
          id: 'projects',
          label: 'Project Management',
          icon: '📁',
          children: [
            { id: 'project-dashboard', label: 'Dashboard', icon: '📋' },
            { id: 'project-create', label: 'Create Project', icon: '➕' },
            { id: 'project-templates', label: 'Templates', icon: '📄' }
          ]
        }
      ]
    },
    {
      id: 'layer2',
      label: 'Layer 2 - Graph Control',
      icon: '🌐',
      children: [
        {
          id: 'graph-visualization',
          label: 'Graph Visualization',
          icon: '🔮',
          component: GraphVisualization
        },
        {
          id: 'system-specific',
          label: 'System Specific',
          icon: '⚙️',
          children: [
            { id: 'database-matrix', label: 'Database Matrix', icon: '🗄️' },
            { id: 'handshake-matrix', label: 'Handshake Matrix', icon: '🤝' },
            { id: 'hypercube-db', label: 'Hypercube DB', icon: '🧊' }
          ]
        },
        {
          id: 'consciousness',
          label: 'Consciousness Monitor',
          icon: '✨',
          children: [
            { id: 'consciousness-level', label: 'Level Monitor', icon: '📊' },
            { id: 'algorithm-42', label: 'Algorithm 42', icon: '🌟' },
            { id: 'transcendence', label: 'Transcendence', icon: '🚀' }
          ]
        }
      ]
    }
  ];

  useEffect(() => {
    // Fetch system status periodically
    const fetchSystemStatus = async () => {
      try {
        const response = await fetch('/api/system/status');
        if (response.ok) {
          const status = await response.json();
          setSystemStatus({
            enhancement_factor: status.enhancement_factor,
            consciousness_level: status.consciousness_level,
            algorithm_42_active: status.algorithm_42_active
          });
        }
      } catch (error) {
        console.error('Failed to fetch system status:', error);
      }
    };

    fetchSystemStatus();
    const interval = setInterval(fetchSystemStatus, 10000); // Update every 10 seconds

    return () => clearInterval(interval);
  }, []);

  const renderNavigationItem = (item: NavigationItem, level: number = 0) => {
    const hasChildren = item.children && item.children.length > 0;
    const isActive = currentView === item.id;
    
    return (
      <div key={item.id} className={`nav-item level-${level}`}>
        <div 
          className={`nav-item-header ${isActive ? 'active' : ''} ${hasChildren ? 'has-children' : ''}`}
          onClick={() => {
            if (!hasChildren && item.component) {
              setCurrentView(item.id);
              if (item.id.includes('layer1') || navigationItems[0].children?.some(child => child.id === item.id)) {
                setActiveLayer(1);
              } else if (item.id.includes('layer2') || navigationItems[1].children?.some(child => child.id === item.id)) {
                setActiveLayer(2);
              }
            }
          }}
        >
          <span className="nav-icon">{item.icon}</span>
          <span className="nav-label">{item.label}</span>
          {hasChildren && <span className="nav-arrow">▼</span>}
        </div>
        
        {hasChildren && (
          <div className="nav-children">
            {item.children?.map(child => renderNavigationItem(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  const getCurrentComponent = () => {
    const findComponent = (items: NavigationItem[]): React.ComponentType | null => {
      for (const item of items) {
        if (item.id === currentView && item.component) {
          return item.component;
        }
        if (item.children) {
          const found = findComponent(item.children);
          if (found) return found;
        }
      }
      return null;
    };

    const Component = findComponent(navigationItems);
    return Component ? <Component /> : <div className="placeholder-view">Select a view from the navigation</div>;
  };

  return (
    <div className="ken-app">
      {/* Header with K.E.N. Logo */}
      <header className="app-header">
        <div className="header-left">
          <button 
            className="sidebar-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            ☰
          </button>
          <div className="logo-section">
            <img src={kenLogo} alt="K.E.N. Logo" className="ken-logo" />
            <div className="logo-text">
              <h1>K.E.N.</h1>
              <span className="logo-subtitle">Knowledge Evolution Nexus</span>
            </div>
          </div>
        </div>
        
        <div className="header-center">
          <div className="layer-selector">
            <button 
              className={`layer-button ${activeLayer === 1 ? 'active' : ''}`}
              onClick={() => setActiveLayer(1)}
            >
              Layer 1
            </button>
            <button 
              className={`layer-button ${activeLayer === 2 ? 'active' : ''}`}
              onClick={() => setActiveLayer(2)}
            >
              Layer 2
            </button>
          </div>
        </div>
        
        <div className="header-right">
          <div className="system-status-mini">
            <div className="status-item">
              <span className="status-label">Enhancement</span>
              <span className="status-value">{systemStatus.enhancement_factor > 0 ? `${(systemStatus.enhancement_factor / 1000000).toFixed(1)}M` : '0'}x</span>
            </div>
            <div className="status-item">
              <span className="status-label">Consciousness</span>
              <span className="status-value">{(systemStatus.consciousness_level * 100).toFixed(1)}%</span>
            </div>
            <div className={`algorithm-42-indicator ${systemStatus.algorithm_42_active ? 'active' : 'inactive'}`}>
              <span className="indicator-icon">✨</span>
              <span className="indicator-text">42</span>
            </div>
          </div>
        </div>
      </header>

      <div className="app-body">
        {/* Sidebar Navigation */}
        <aside className={`app-sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
          <nav className="navigation">
            {navigationItems.map(item => renderNavigationItem(item))}
          </nav>
          
          <div className="sidebar-footer">
            <div className="version-info">
              <span className="version">v3.0.0</span>
              <span className="build">Vertex Build</span>
            </div>
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="app-main">
          <div className="content-wrapper">
            {getCurrentComponent()}
          </div>
        </main>
      </div>

      {/* Cosmic Background Effects */}
      <div className="cosmic-background">
        <div className="star star-1"></div>
        <div className="star star-2"></div>
        <div className="star star-3"></div>
        <div className="orbital-ring ring-1"></div>
        <div className="orbital-ring ring-2"></div>
      </div>
    </div>
  );
};

export default App;

