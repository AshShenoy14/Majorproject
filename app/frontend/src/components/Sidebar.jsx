import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  Home, Search, Dna, Box, Share2, 
  ShieldAlert, Info, Database, Activity, 
  Bot, Globe, Settings, Cpu
} from 'lucide-react';

const Sidebar = () => {
  const menuItems = [
    { name: 'Home', icon: <Home size={16} />, path: '/' },
    { name: 'PPI Predictor', icon: <Search size={16} />, path: '/predict' },
    { name: 'Mutation Analysis', icon: <Dna size={16} />, path: '/mutation' },
    { name: 'Structure View', icon: <Box size={16} />, path: '/structure' },
    { name: 'Interactome 3D', icon: <Share2 size={16} />, path: '/network-3d' },
    { name: 'Drug Insights', icon: <ShieldAlert size={16} />, path: '/drug-targets' },
    { name: 'Bio-Assistant', icon: <Bot size={16} />, path: '/assistant' },
    { name: 'Zero-Shot Eval', icon: <Globe size={16} />, path: '/zero-shot' },
  ];

  return (
    <div className="scientific-sidebar bg-white/90 border-r border-slate-100 shadow-2xl flex flex-col h-full overflow-hidden">
      
      {/* Brand Section */}
      <div className="p-10 flex flex-col gap-1 items-center">
        <div className="p-3 bg-emerald-500/10 rounded-2xl text-emerald-600 mb-2">
          <Activity size={32} />
        </div>
        <h1 className="logo-cursive tracking-tight">
          Trans<span className="text-emerald-500">Graph</span>
        </h1>
        <p className="text-[9px] text-slate-400 font-black tracking-[0.4em] uppercase">
          PPI FRAMEWORK
        </p>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 space-y-1">
        <div className="px-6 mb-4">
          <p className="text-[9px] font-black text-slate-600 uppercase tracking-widest">Main Modules</p>
        </div>
        {menuItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => 
              `nav-item ${isActive ? 'active' : 'opacity-60 hover:opacity-100'}`
            }
          >
            <div className="w-5 h-5 flex items-center justify-center">
              {item.icon}
            </div>
            <span className="text-[11px] font-black uppercase tracking-widest">{item.name}</span>
          </NavLink>
        ))}
      </nav>

      {/* Hardware Status Panel */}
      <div className="p-6">
        <div className="p-6 bg-slate-50 rounded-[2rem] border border-slate-100 shadow-inner">
          <div className="flex items-center gap-2 text-emerald-600 mb-4">
            <Activity size={14} />
            <span className="text-[9px] font-black uppercase tracking-[0.2em]">Real-time Status</span>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">Engine Status</span>
              <div className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
                <span className="text-[9px] text-emerald-600 font-black uppercase">Online</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">GNN Core</span>
              <div className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" />
                <span className="text-[9px] text-emerald-600 font-black uppercase">Synced</span>
              </div>
            </div>
            <div className="mt-4 pt-4 border-t border-slate-200">
               <div className="flex justify-between items-center opacity-80">
                  <span className="text-[9px] text-slate-400 font-bold uppercase">Uptime</span>
                  <span className="text-[9px] text-slate-600 font-mono">14:22:04</span>
               </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
