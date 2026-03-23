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
  Activity
} from 'lucide-react';

const Sidebar = () => {
  const menuItems = [
    { name: 'Home', icon: <Home size={20} />, path: '/' },
    { name: 'Predict Interaction', icon: <Search size={20} />, path: '/predict' },
    { name: 'Mutation Analysis', icon: <Dna size={20} />, path: '/mutation' },
    { name: 'Protein Structure', icon: <Box size={20} />, path: '/structure' },
    { name: 'Network Explorer', icon: <Share2 size={20} />, path: '/network' },
    { name: 'Drug Target Insights', icon: <ShieldAlert size={20} />, path: '/drug-targets' },
    { name: 'About', icon: <Info size={20} />, path: '/about' },
  ];

  return (
    <div className="scientific-sidebar">
      <div className="p-6 flex items-center gap-3">
        <div className="w-10 h-10 bg-scientific-gradient rounded-xl flex items-center justify-center text-white shadow-lg">
          <Activity size={24} />
        </div>
        <div>
          <h1 className="text-xl font-bold bg-clip-text text-transparent bg-scientific-gradient">
            TransGraph
          </h1>
          <p className="text-[10px] text-slate-400 font-medium tracking-widest uppercase">
            PPI Platform
          </p>
        </div>
      </div>

      <nav className="flex-1 mt-4">
        {menuItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => 
              `nav-item ${isActive ? 'active' : ''}`
            }
          >
            {item.icon}
            <span className="text-sm font-medium">{item.name}</span>
          </NavLink>
        ))}
      </nav>

      <div className="p-4 mt-auto">
        <div className="p-4 bg-scientific-primary/5 rounded-2xl border border-scientific-primary/10">
          <div className="flex items-center gap-2 text-scientific-primary mb-2">
            <Database size={16} />
            <span className="text-xs font-bold uppercase tracking-wider">System Status</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-success rounded-full animate-pulse" />
            <span className="text-[10px] text-slate-500 font-medium">Backend Connected</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
