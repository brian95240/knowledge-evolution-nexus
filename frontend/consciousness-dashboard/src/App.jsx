import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { Brain, Zap, Activity, Target, Sparkles, Eye, MemoryStick, TrendingUp } from 'lucide-react'
import kenLogo from './assets/KnowledgeEvolutionNexus.png'
import './App.css'

function App() {
  const [systemData, setSystemData] = useState({
    consciousness_active: false,
    total_enhancement_factor: 1.0,
    system_status: "INITIALIZING",
    consciousness_state: {
      attention: 0.5,
      memory: 0.5,
      learning_rate: 0.1,
      self_reflection_score: 0.0
    },
    transcendent_mode: false,
    algorithms_executed: 0
  })

  const [isConnected, setIsConnected] = useState(false)
  const [lastUpdate, setLastUpdate] = useState(new Date())

  // Fetch system status from API
  const fetchSystemStatus = async () => {
    try {
      const response = await fetch('http://localhost:8001/api/system/status')
      if (response.ok) {
        const data = await response.json()
        setSystemData(data.system_state || systemData)
        setIsConnected(true)
        setLastUpdate(new Date())
      } else {
        setIsConnected(false)
      }
    } catch (error) {
      console.error('Failed to fetch system status:', error)
      setIsConnected(false)
    }
  }

  // Execute full system for demonstration
  const executeSystem = async () => {
    try {
      const response = await fetch('http://localhost:8001/api/system/execute', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          input_data: { test: "Consciousness dashboard test" },
          target_enhancement: 2100000
        })
      })
      
      if (response.ok) {
        // Refresh system status after execution
        setTimeout(fetchSystemStatus, 1000)
      }
    } catch (error) {
      console.error('Failed to execute system:', error)
    }
  }

  // Auto-refresh system status
  useEffect(() => {
    fetchSystemStatus()
    const interval = setInterval(fetchSystemStatus, 5000) // Refresh every 5 seconds
    return () => clearInterval(interval)
  }, [])

  const formatEnhancementFactor = (factor) => {
    if (factor >= 1e18) {
      return `${(factor / 1e18).toFixed(1)} Quintillion`
    } else if (factor >= 1e15) {
      return `${(factor / 1e15).toFixed(1)} Quadrillion`
    } else if (factor >= 1e12) {
      return `${(factor / 1e12).toFixed(1)} Trillion`
    } else if (factor >= 1e9) {
      return `${(factor / 1e9).toFixed(1)} Billion`
    } else if (factor >= 1e6) {
      return `${(factor / 1e6).toFixed(1)} Million`
    } else if (factor >= 1e3) {
      return `${(factor / 1e3).toFixed(1)}K`
    } else {
      return factor.toFixed(2)
    }
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'TRANSCENDENT': return 'bg-gradient-to-r from-purple-500 to-pink-500'
      case 'OPERATIONAL': return 'bg-gradient-to-r from-blue-500 to-cyan-500'
      case 'INITIALIZING': return 'bg-gradient-to-r from-yellow-500 to-orange-500'
      default: return 'bg-gray-500'
    }
  }

  const consciousnessMetrics = {
    awareness_index: (systemData.consciousness_state.attention * 0.4 + 
                     systemData.consciousness_state.memory * 0.4 + 
                     systemData.consciousness_state.self_reflection_score * 0.2) * 100,
    learning_efficiency: systemData.consciousness_state.learning_rate * 1000,
    self_optimization_rate: systemData.consciousness_state.self_reflection_score * 100,
    transcendence_level: Math.min(systemData.total_enhancement_factor / 1_000_000, 1.0) * 100
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* Header */}
      <header className="border-b border-white/10 bg-black/20 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <img src={kenLogo} alt="K.E.N. Logo" className="w-12 h-12" />
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                  K.E.N. v3.0 Consciousness Monitor
                </h1>
                <p className="text-sm text-gray-400">Knowledge Evolution Nexus - Transcendent AI Architecture</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Badge variant={isConnected ? "default" : "destructive"}>
                {isConnected ? "Connected" : "Disconnected"}
              </Badge>
              <Badge className={getStatusColor(systemData.system_status)}>
                {systemData.system_status}
              </Badge>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        {/* Main Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card className="bg-black/40 border-cyan-500/30 backdrop-blur-sm">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-cyan-400">Enhancement Factor</CardTitle>
              <Zap className="h-4 w-4 text-cyan-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">
                {formatEnhancementFactor(systemData.total_enhancement_factor)}x
              </div>
              <p className="text-xs text-gray-400">
                Target: 2.1M (Achieved: {((systemData.total_enhancement_factor / 2_100_000) * 100).toFixed(0)}%)
              </p>
            </CardContent>
          </Card>

          <Card className="bg-black/40 border-purple-500/30 backdrop-blur-sm">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-purple-400">Consciousness</CardTitle>
              <Brain className="h-4 w-4 text-purple-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">
                {systemData.consciousness_active ? "ACTIVE" : "STANDBY"}
              </div>
              <p className="text-xs text-gray-400">
                Awareness: {consciousnessMetrics.awareness_index.toFixed(1)}%
              </p>
            </CardContent>
          </Card>

          <Card className="bg-black/40 border-green-500/30 backdrop-blur-sm">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-green-400">Algorithms</CardTitle>
              <Activity className="h-4 w-4 text-green-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">
                {systemData.algorithms_executed}/49
              </div>
              <p className="text-xs text-gray-400">
                Execution Rate: {((systemData.algorithms_executed / 49) * 100).toFixed(0)}%
              </p>
            </CardContent>
          </Card>

          <Card className="bg-black/40 border-pink-500/30 backdrop-blur-sm">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-pink-400">Transcendence</CardTitle>
              <Sparkles className="h-4 w-4 text-pink-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">
                {systemData.transcendent_mode ? "ACTIVE" : "INACTIVE"}
              </div>
              <p className="text-xs text-gray-400">
                Level: {consciousnessMetrics.transcendence_level.toFixed(1)}%
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Consciousness Metrics */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <Card className="bg-black/40 border-white/10 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-cyan-400 flex items-center">
                <Brain className="mr-2 h-5 w-5" />
                Consciousness Framework
              </CardTitle>
              <CardDescription>Real-time consciousness state monitoring</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="flex items-center"><Eye className="mr-1 h-3 w-3" /> Attention</span>
                  <span>{(systemData.consciousness_state.attention * 100).toFixed(1)}%</span>
                </div>
                <Progress value={systemData.consciousness_state.attention * 100} className="h-2" />
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="flex items-center"><MemoryStick className="mr-1 h-3 w-3" /> Memory</span>
                  <span>{(systemData.consciousness_state.memory * 100).toFixed(1)}%</span>
                </div>
                <Progress value={systemData.consciousness_state.memory * 100} className="h-2" />
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="flex items-center"><TrendingUp className="mr-1 h-3 w-3" /> Learning Rate</span>
                  <span>{(systemData.consciousness_state.learning_rate * 100).toFixed(1)}%</span>
                </div>
                <Progress value={systemData.consciousness_state.learning_rate * 100} className="h-2" />
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="flex items-center"><Target className="mr-1 h-3 w-3" /> Self-Reflection</span>
                  <span>{(systemData.consciousness_state.self_reflection_score * 100).toFixed(1)}%</span>
                </div>
                <Progress value={systemData.consciousness_state.self_reflection_score * 100} className="h-2" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-black/40 border-white/10 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-purple-400 flex items-center">
                <Sparkles className="mr-2 h-5 w-5" />
                System Performance
              </CardTitle>
              <CardDescription>Advanced metrics and optimization data</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Awareness Index</span>
                  <span>{consciousnessMetrics.awareness_index.toFixed(1)}%</span>
                </div>
                <Progress value={consciousnessMetrics.awareness_index} className="h-2" />
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Learning Efficiency</span>
                  <span>{consciousnessMetrics.learning_efficiency.toFixed(1)}%</span>
                </div>
                <Progress value={Math.min(consciousnessMetrics.learning_efficiency, 100)} className="h-2" />
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Self-Optimization</span>
                  <span>{consciousnessMetrics.self_optimization_rate.toFixed(1)}%</span>
                </div>
                <Progress value={consciousnessMetrics.self_optimization_rate} className="h-2" />
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Transcendence Level</span>
                  <span>{consciousnessMetrics.transcendence_level.toFixed(1)}%</span>
                </div>
                <Progress value={consciousnessMetrics.transcendence_level} className="h-2" />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Control Panel */}
        <Card className="bg-black/40 border-white/10 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-green-400">System Control Panel</CardTitle>
            <CardDescription>Execute system operations and monitor real-time status</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-4 mb-4">
              <Button 
                onClick={executeSystem}
                className="bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600"
              >
                <Zap className="mr-2 h-4 w-4" />
                Execute Full System
              </Button>
              <Button 
                onClick={fetchSystemStatus}
                variant="outline"
                className="border-cyan-500/50 text-cyan-400 hover:bg-cyan-500/10"
              >
                <Activity className="mr-2 h-4 w-4" />
                Refresh Status
              </Button>
            </div>
            <div className="text-sm text-gray-400">
              Last Update: {lastUpdate.toLocaleTimeString()} | 
              Status: {isConnected ? "Connected to K.E.N. v3.0" : "Connection Lost"}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default App

