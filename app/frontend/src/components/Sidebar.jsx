import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  Home, 
  Search, 
  Dna, 
  Box, 
  Share2, 
  ShieldAlert, 
  Info,
  Database,
  Activity,
  Bot,
  Globe
} from 'lucide-react';

const Sidebar = () => {
  const menuItems = [
    { name: 'Dashboard', icon: <Home size={18} />, path: '/' },
    { name: 'Interaction Predictor', icon: <Search size={18} />, path: '/predict' },
    { name: 'Mutation Scanner', icon: <Dna size={18} />, path: '/mutation' },
    { name: '3D Structure', icon: <Box size={18} />, path: '/structure' },
    { name: 'Interactome Graph', icon: <Share2 size={18} />, path: '/network' },
    { name: 'Drug Insights', icon: <ShieldAlert size={18} />, path: '/drug-targets' },
    { name: 'Bio-AI Assistant', icon: <Bot size={18} />, path: '/assistant' },
    { name: 'Zero-Shot Eval', icon: <Globe size={18} />, path: '/zero-shot' },
    { name: 'System Info', icon: <Info size={18} />, path: '/about' },
  ];

  return (
    <div className="scientific-sidebar">
      <div className="p-8 flex items-center gap-3">
        <div className="w-12 h-12 bg-gradient-to-br from-teal-400 to-teal-600 rounded-2xl flex items-center justify-center text-white shadow-lg shadow-teal-500/30">
          <Activity size={28} />
        </div>
        <div>
          <h1 className="text-2xl font-black tracking-tighter text-slate-800 leading-none">
            Trans<span className="text-scientific-primary">Graph</span>
          </h1>
          <p className="text-[10px] text-slate-400 font-bold tracking-[0.2em] uppercase mt-1">
            PPI FRAMEWORK
          </p>
        </div>
      </div>

      <nav className="flex-1 px-2 space-y-1">
        {menuItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => 
              `nav-item ${isActive ? 'active' : ''}`
            }
          >
            <div className="w-8 h-8 flex items-center justify-center rounded-lg">
              {item.icon}
            </div>
            <span className="text-sm font-bold tracking-tight">{item.name}</span>
          </NavLink>
        ))}
      </nav>

      <div className="p-6">
        <div className="p-5 glass-card bg-white/40 border-none shadow-none">
          <div className="flex items-center gap-2 text-scientific-primary mb-3">
            <Database size={16} />
            <span className="text-[10px] font-black uppercase tracking-[0.2em]">Live Data</span>
          </div>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-slate-500 font-bold uppercase">Backend</span>
              <div className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
                <span className="text-[10px] text-slate-700 font-bold uppercase">Ready</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-slate-500 font-bold uppercase">GPU Sync</span>
              <div className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
                <span className="text-[10px] text-slate-700 font-bold uppercase">Active</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
