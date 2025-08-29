import React, { useState } from 'react';
import URLAnalyzer from './components/URLAnalyzer';
import Dashboard from './components/Dashboard';
import SystemStatus from './components/SystemStatus';
import { Shield, BarChart3, Globe, Zap } from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState('analyzer');

  const tabs = [
    { id: 'analyzer', name: 'URL Analyzer', icon: Shield },
    { id: 'dashboard', name: 'Dashboard', icon: BarChart3 },
    { id: 'status', name: 'System Status', icon: Zap },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-lg border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center space-x-3">
              <Shield className="h-8 w-8 text-blue-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Phishing Triage System</h1>
                <p className="text-sm text-gray-600">Advanced Threat Intelligence & ML Detection</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 bg-green-50 px-3 py-1 rounded-full">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm font-medium text-green-700">System Online</span>
              </div>
              <Globe className="h-5 w-5 text-gray-400" />
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="h-5 w-5" />
                  <span>{tab.name}</span>
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        {activeTab === 'analyzer' && <URLAnalyzer />}
        {activeTab === 'dashboard' && <Dashboard />}
        {activeTab === 'status' && <SystemStatus />}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white mt-16">
        <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center">
            <div>
              <p className="text-lg font-semibold">Built by Huy Tran</p>
              <p className="text-gray-400">Advanced Cybersecurity & ML Engineering Project</p>
            </div>
            <div className="flex space-x-6">
              <a 
                href="https://github.com/itsnothuy/Phishing-Triage" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-white transition-colors"
              >
                GitHub Repository
              </a>
              <a 
                href="http://localhost:8001/docs" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-white transition-colors"
              >
                API Documentation
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;