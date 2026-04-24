import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Menu, X, Search, Command, Activity, Zap } from 'lucide-react';
import Sidebar from './Sidebar';
import { AnimatePresence, motion } from 'framer-motion';

const PAGE_META = {
  '/':            { title: 'Dashboard',              subtitle: 'System Overview & Quick Actions' },
  '/predict':     { title: 'Interaction Prediction',  subtitle: 'Predict Protein-Protein Interactions' },
  '/mutation':    { title: 'Mutation Analysis',        subtitle: 'In-Silico Mutation Impact Scanner' },
  '/structure':   { title: 'Structure Viewer',         subtitle: '3D Protein Structure Visualization' },
  '/network':     { title: 'Network Explorer',         subtitle: 'Interactome Graph Analysis' },
  '/network-3d':  { title: 'Interactome 3D',           subtitle: 'Global Interaction Topography' },
  '/drug-targets':{ title: 'Drug Insights',            subtitle: 'Drug Target Discovery & ChEMBL Data' },
  '/assistant':   { title: 'Protein Assistant',        subtitle: 'AI-Powered Biological Query Engine' },
  '/about':       { title: 'About',                    subtitle: 'Project Overview & Model Details' },
  '/zero-shot':   { title: 'Cross-Species Testing',    subtitle: 'Zero-Shot Generalization Evaluation' },
};

const Layout = ({ children }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  
  const meta = PAGE_META[location.pathname] ?? { title: 'Bioinformatics Analysis', subtitle: '' };

  // Handle keyboard shortcut Ctrl+K / Cmd+K
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsSearchOpen(true);
      }
      if (e.key === 'Escape') {
        setIsSearchOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const filteredPages = Object.entries(PAGE_META).filter(([path, data]) => 
    data.title.toLowerCase().includes(searchQuery.toLowerCase()) || 
    data.subtitle.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="flex min-h-screen bg-slate-50 relative overflow-hidden font-inter">
      {/* Search Modal Overlay */}
      <AnimatePresence>
        {isSearchOpen && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] bg-slate-900/40 backdrop-blur-md flex items-start justify-center pt-32 px-4"
            onClick={() => setIsSearchOpen(false)}
          >
            <motion.div 
              initial={{ scale: 0.95, opacity: 0, y: -20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.95, opacity: 0, y: -20 }}
              className="bg-white w-full max-w-2xl rounded-[2rem] shadow-2xl overflow-hidden border border-slate-100"
              onClick={e => e.stopPropagation()}
            >
              <div className="p-6 border-b border-slate-100 flex items-center gap-4">
                <Search className="text-slate-400" size={24} />
                <input 
                  autoFocus
                  placeholder="Search interactions, proteins, or modules..."
                  className="flex-1 bg-transparent border-none outline-none text-xl font-medium text-slate-800 placeholder:text-slate-300"
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                />
                <div className="flex items-center gap-1.5 px-2 py-1 bg-slate-50 rounded-lg text-[10px] font-black text-slate-400 border border-slate-100">
                  <span className="text-xs">ESC</span>
                </div>
              </div>
              <div className="max-h-[400px] overflow-y-auto p-4">
                <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-4 px-4">Navigation Results</p>
                <div className="space-y-1">
                  {filteredPages.map(([path, data]) => (
                    <button
                      key={path}
                      onClick={() => {
                        navigate(path);
                        setIsSearchOpen(false);
                      }}
                      className="w-full text-left p-4 rounded-2xl hover:bg-emerald-50 transition-all group flex items-center justify-between"
                    >
                      <div className="flex items-center gap-4">
                        <div className="w-10 h-10 bg-slate-50 rounded-xl flex items-center justify-center text-slate-400 group-hover:bg-white group-hover:text-emerald-600 transition-all">
                          <Activity size={18} />
                        </div>
                        <div>
                          <p className="text-sm font-bold text-slate-700 group-hover:text-emerald-700">{data.title}</p>
                          <p className="text-[11px] text-slate-400 group-hover:text-emerald-600/60">{data.subtitle}</p>
                        </div>
                      </div>
                      <Zap size={14} className="text-slate-200 group-hover:text-emerald-400 opacity-0 group-hover:opacity-100 transition-all" />
                    </button>
                  ))}
                  {filteredPages.length === 0 && (
                    <div className="p-8 text-center text-slate-400">
                      <p className="font-bold">No results found for "{searchQuery}"</p>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Collapsible Sidebar */}
      <AnimatePresence mode="wait">
        {isSidebarOpen && (
          <motion.div
            initial={{ x: -320, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -320, opacity: 0 }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="fixed inset-y-0 left-0 z-50 w-[19rem]"
          >
            <Sidebar onClose={() => setIsSidebarOpen(false)} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content Area */}
      <main 
        className={`flex-1 transition-all duration-500 ease-in-out ${isSidebarOpen ? 'ml-[19rem]' : 'ml-0'} mr-8 my-8 relative`}
      >
        <header className="mb-12 flex justify-between items-center px-4">
          <div className="flex items-center gap-8">
            {/* Hamburger Button & Persistent Logo */}
            <div className="flex items-center gap-4">
              {!isSidebarOpen && (
                <>
                  <motion.button
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    onClick={() => setIsSidebarOpen(true)}
                    className="w-12 h-12 bg-white shadow-lg rounded-2xl flex items-center justify-center text-slate-600 hover:text-emerald-600 border border-slate-100 transition-all group"
                  >
                    <Menu size={20} className="group-hover:scale-110 transition-transform" />
                  </motion.button>
                  
                  {/* Branding Logo when sidebar is closed */}
                  <motion.div 
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="flex items-center gap-2"
                  >
                    <div className="w-8 h-8 bg-emerald-500 rounded-lg flex items-center justify-center text-white shadow-lg shadow-emerald-200">
                      <Activity size={16} />
                    </div>
                    <span className="logo-cursive text-xl">Trans<span className="text-emerald-500">Graph</span></span>
                  </motion.div>
                </>
              )}
            </div>
            
            <div className={!isSidebarOpen ? 'border-l border-slate-200 pl-8' : ''}>
              <h2 className="text-[10px] font-black text-emerald-600 uppercase tracking-[0.4em] mb-1">
                {meta.title}
              </h2>
              {meta.subtitle && (
                <p className="text-xl text-slate-800 font-black tracking-tight">{meta.subtitle}</p>
              )}
            </div>
          </div>

          <div className="flex items-center gap-6">
            {/* Global Search Trigger */}
            <button 
              onClick={() => setIsSearchOpen(true)}
              className="flex items-center gap-4 bg-white px-4 py-2.5 rounded-2xl border border-slate-100 shadow-sm text-slate-400 hover:border-emerald-200 transition-all group"
            >
              <Search size={16} className="group-hover:text-emerald-500 transition-colors" />
              <span className="text-xs font-bold pr-12">Search anything...</span>
              <div className="flex items-center gap-1 opacity-50">
                <Command size={12} />
                <span className="text-[10px] font-black">K</span>
              </div>
            </button>

            <div className="glass-card bg-white px-6 py-3 flex items-center gap-3 text-[10px] font-black text-slate-400 tracking-widest border border-slate-100 shadow-sm">
              <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)] animate-pulse"></div>
              SYSTEM ACTIVE
            </div>
          </div>
        </header>

        <div className="px-4 animate-in fade-in slide-in-from-bottom-4 duration-700">
          {children}
        </div>
      </main>
    </div>
  );
};

export default Layout;
