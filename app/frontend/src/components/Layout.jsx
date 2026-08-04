import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import { 
  Search, 
  Command, 
  Activity, 
  Zap, 
  Dna, 
  Boxes, 
  Share2, 
  Pill, 
  Bot, 
  Home as HomeIcon,
  Globe,
  PanelLeft,
  Rows
} from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import FloatingGuide from './FloatingGuide';

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

const NAV_LINKS = [
  { path: '/', label: 'Home', icon: HomeIcon },
  { path: '/predict', label: 'Predict', icon: Zap },
  { path: '/mutation', label: 'Mutation', icon: Dna },
  { path: '/structure', label: 'Structure', icon: Boxes },
  { path: '/network-3d', label: '3D Graph', icon: Share2 },
  { path: '/drug-targets', label: 'Drugs', icon: Pill },
  { path: '/zero-shot', label: 'Cross-Species', icon: Globe },
  { path: '/assistant', label: 'AI Assistant', icon: Bot },
];

const Layout = ({ children }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [navOrientation, setNavOrientation] = useState(() => {
    return localStorage.getItem('transgraph_nav_orientation') || 'horizontal';
  });

  const toggleOrientation = () => {
    const nextMode = navOrientation === 'horizontal' ? 'vertical' : 'horizontal';
    setNavOrientation(nextMode);
    localStorage.setItem('transgraph_nav_orientation', nextMode);
  };

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

  const isVertical = navOrientation === 'vertical';

  return (
    <div className="min-h-screen bg-slate-50 relative overflow-x-hidden font-inter text-slate-800 pb-16">
      
      {/* ========================================== */}
      {/* NAVIGATION BAR (HORIZONTAL OR VERTICAL) */}
      {/* ========================================== */}
      {isVertical ? (
        /* VERTICAL SIDEBAR LAYOUT */
        <motion.aside
          initial={{ x: -50, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          className="fixed top-0 left-0 bottom-0 w-64 bg-white/95 backdrop-blur-xl z-50 border-r border-slate-200/80 p-5 flex flex-col justify-between shadow-2xl shadow-slate-900/10"
        >
          <div className="space-y-6">
            {/* Logo */}
            <Link to="/" className="flex items-center gap-2.5 pb-4 border-b border-slate-100 group">
              <div className="w-9 h-9 bg-gradient-to-tr from-emerald-600 to-teal-500 rounded-full flex items-center justify-center text-white shadow-md shadow-emerald-500/20 group-hover:scale-105 transition-transform">
                <Activity size={18} className="animate-pulse" />
              </div>
              <span className="font-cursive text-xl tracking-wide font-bold" style={{ fontFamily: "'Dancing Script', cursive" }}>
                Trans<span className="text-emerald-600">Graph</span>
              </span>
            </Link>

            {/* Quick Search */}
            <button 
              onClick={() => setIsSearchOpen(true)}
              className="w-full flex items-center justify-between bg-slate-100/80 hover:bg-slate-200/60 px-3.5 py-2 rounded-2xl text-slate-500 transition-colors text-xs font-medium"
              title="Search Platform (Ctrl + K)"
            >
              <div className="flex items-center gap-2">
                <Search size={14} className="text-slate-400" />
                <span className="text-[11px] font-semibold">Search...</span>
              </div>
              <kbd className="flex items-center gap-0.5 px-1.5 py-0.5 bg-white rounded text-[9px] font-black text-slate-400 border border-slate-200">
                <Command size={10} />K
              </kbd>
            </button>

            {/* Nav Links */}
            <nav className="space-y-1">
              <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest px-2 mb-2">Navigation</p>
              {NAV_LINKS.map((link) => {
                const isActive = location.pathname === link.path;
                const Icon = link.icon;
                return (
                  <Link
                    key={link.path}
                    to={link.path}
                    className={`px-3.5 py-2.5 rounded-2xl text-xs font-semibold transition-all flex items-center gap-3 ${
                      isActive 
                        ? 'text-emerald-700 bg-emerald-50 shadow-sm border border-emerald-200/60 font-bold' 
                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/70'
                    }`}
                  >
                    <Icon size={16} className={isActive ? 'text-emerald-600' : 'text-slate-400'} />
                    <span>{link.label}</span>
                  </Link>
                );
              })}
            </nav>
          </div>

          {/* Footer Controls (Layout Switcher) */}
          <div className="pt-4 border-t border-slate-100">
            <button
              onClick={toggleOrientation}
              className="w-full flex items-center justify-between px-3.5 py-2.5 rounded-2xl bg-slate-100 hover:bg-slate-200/80 text-slate-700 text-xs font-bold transition-all"
              title="Switch Navbar Layout"
            >
              <div className="flex items-center gap-2">
                <Rows size={15} className="text-emerald-600" />
                <span>Switch to Horizontal</span>
              </div>
              <span className="text-[10px] bg-white px-2 py-0.5 rounded-full text-slate-500 border border-slate-200 font-mono">Top</span>
            </button>
          </div>
        </motion.aside>
      ) : (
        /* HORIZONTAL FLOATING NAVBAR LAYOUT */
        <header className="fixed top-4 left-0 right-0 z-50 px-4 md:px-8 max-w-7xl mx-auto flex items-center justify-between gap-4 pointer-events-none">
          
          {/* PART 1: LEFT FLOATING PILL (Logo + Nav Links) */}
          <motion.nav 
            initial={{ y: -20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="pointer-events-auto flex items-center gap-1.5 md:gap-2.5 bg-white/90 backdrop-blur-xl px-3.5 py-2 rounded-full shadow-lg shadow-slate-900/5 border border-slate-200/80"
          >
            {/* Logo */}
            <Link to="/" className="flex items-center gap-2 pr-3 border-r border-slate-200/80 group">
              <div className="w-8 h-8 bg-gradient-to-tr from-emerald-600 to-teal-500 rounded-full flex items-center justify-center text-white shadow-md shadow-emerald-500/20 group-hover:scale-105 transition-transform">
                <Activity size={16} className="animate-pulse" />
              </div>
              <span className="font-cursive text-lg tracking-wide hidden sm:inline-block font-bold" style={{ fontFamily: "'Dancing Script', cursive" }}>
                Trans<span className="text-emerald-600">Graph</span>
              </span>
            </Link>

            {/* Links */}
            <div className="flex items-center gap-1">
              {NAV_LINKS.map((link) => {
                const isActive = location.pathname === link.path;
                const Icon = link.icon;
                return (
                  <Link
                    key={link.path}
                    to={link.path}
                    className={`relative px-3 py-1.5 rounded-full text-xs font-semibold transition-all flex items-center gap-1.5 ${
                      isActive 
                        ? 'text-emerald-700 bg-emerald-50 shadow-sm border border-emerald-200/60 font-bold' 
                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/70'
                    }`}
                  >
                    <Icon size={14} className={isActive ? 'text-emerald-600' : 'text-slate-400'} />
                    <span className="hidden md:inline-block">{link.label}</span>
                  </Link>
                );
              })}
            </div>
          </motion.nav>

          {/* PART 2: RIGHT FLOATING PILL (Search + Layout Toggle) */}
          <motion.div 
            initial={{ y: -20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.1 }}
            className="pointer-events-auto flex items-center gap-2 bg-white/90 backdrop-blur-xl px-3 py-1.5 rounded-full shadow-lg shadow-slate-900/5 border border-slate-200/80"
          >
            {/* Quick Search Trigger */}
            <button 
              onClick={() => setIsSearchOpen(true)}
              className="flex items-center gap-2 bg-slate-100/80 hover:bg-slate-200/60 px-3 py-1.5 rounded-full text-slate-500 transition-colors text-xs font-medium"
              title="Search Platform (Ctrl + K)"
            >
              <Search size={14} className="text-slate-400" />
              <span className="hidden sm:inline-block text-[11px] font-semibold">Search...</span>
              <kbd className="hidden sm:flex items-center gap-0.5 px-1.5 py-0.5 bg-white rounded text-[9px] font-black text-slate-400 border border-slate-200">
                <Command size={10} />K
              </kbd>
            </button>

            {/* Layout Toggle Button (Horizontal / Vertical Switcher) */}
            <button
              onClick={toggleOrientation}
              className="flex items-center gap-1.5 bg-slate-100 hover:bg-slate-200/80 px-2.5 py-1.5 rounded-full text-slate-700 text-xs font-bold transition-all"
              title="Switch Navbar to Vertical Sidebar"
            >
              <PanelLeft size={14} className="text-emerald-600" />
              <span className="hidden md:inline-block text-[11px]">Vertical Nav</span>
            </button>
          </motion.div>

        </header>
      )}

      {/* Global Search Modal Overlay */}
      <AnimatePresence>
        {isSearchOpen && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] bg-slate-900/40 backdrop-blur-md flex items-start justify-center pt-28 px-4"
            onClick={() => setIsSearchOpen(false)}
          >
            <motion.div 
              initial={{ scale: 0.95, opacity: 0, y: -20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.95, opacity: 0, y: -20 }}
              className="bg-white w-full max-w-2xl rounded-3xl shadow-2xl overflow-hidden border border-slate-100"
              onClick={e => e.stopPropagation()}
            >
              <div className="p-5 border-b border-slate-100 flex items-center gap-3">
                <Search className="text-slate-400" size={20} />
                <input 
                  autoFocus
                  placeholder="Search interaction models, proteins, disease explorer..."
                  className="flex-1 bg-transparent border-none outline-none text-lg font-medium text-slate-800 placeholder:text-slate-300"
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                />
                <div className="px-2 py-1 bg-slate-100 rounded-md text-[10px] font-black text-slate-400">
                  ESC
                </div>
              </div>
              <div className="max-h-[360px] overflow-y-auto p-3">
                <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-3 px-3">Available Modules</p>
                <div className="space-y-1">
                  {filteredPages.map(([path, data]) => (
                    <button
                      key={path}
                      onClick={() => {
                        navigate(path);
                        setIsSearchOpen(false);
                      }}
                      className="w-full text-left p-3 rounded-2xl hover:bg-emerald-50 transition-all group flex items-center justify-between"
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-9 h-9 bg-slate-50 rounded-xl flex items-center justify-center text-slate-400 group-hover:bg-white group-hover:text-emerald-600 transition-all shadow-sm">
                          <Activity size={16} />
                        </div>
                        <div>
                          <p className="text-sm font-bold text-slate-700 group-hover:text-emerald-700">{data.title}</p>
                          <p className="text-[11px] text-slate-400 group-hover:text-emerald-600/70">{data.subtitle}</p>
                        </div>
                      </div>
                      <Zap size={14} className="text-slate-200 group-hover:text-emerald-500 opacity-0 group-hover:opacity-100 transition-all" />
                    </button>
                  ))}
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content Area */}
      <main className={`px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto transition-all ${isVertical ? 'md:pl-72 pt-6' : 'pt-24'}`}>
        <div className="animate-in fade-in slide-in-from-bottom-3 duration-500">
          {children}
        </div>
      </main>

      {/* Global Floating Guide Dot */}
      <FloatingGuide />
    </div>
  );
};

export default Layout;

