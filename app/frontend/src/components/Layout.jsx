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
  Rows,
  BarChart3,
  GitCompare,
  Network
} from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import FloatingGuide from './FloatingGuide';

const PAGE_META = {
  '/':            { title: 'Dashboard',              subtitle: 'System Overview & Quick Actions' },
  '/predict':     { title: 'Interaction Prediction',  subtitle: 'Predict Protein-Protein Interactions' },
  '/structure':   { title: '3D Structure Studio',     subtitle: 'AlphaFold Mol* Visualization' },
  '/mutation':    { title: 'Mutation Analysis',        subtitle: 'In-Silico Mutation Impact Scanner' },
  '/compare':     { title: 'WT vs Mutant Comparator',  subtitle: 'Sensitivity & Probability Delta' },
  '/network':     { title: 'Network Explorer',         subtitle: '2D Interactome Graph Analysis' },
  '/network-3d':  { title: 'Interactome 3D',           subtitle: 'Global Interaction Topography' },
  '/drug-targets':{ title: 'Drug Insights',            subtitle: 'Drug Target Discovery & ChEMBL Data' },
  '/benchmark':   { title: 'Empirical Benchmarks',     subtitle: 'Statistical Rigor, 95% CIs & External Sets' },
  '/zero-shot':   { title: 'Cross-Species Exploration',subtitle: 'Exploratory, Non-Validated Inference' },
  '/assistant':   { title: 'Protein Assistant',        subtitle: 'AI-Powered Biological Query Engine' },
  '/about':       { title: 'About & Technical Specs',  subtitle: 'Architecture, Ablations & Pipeline' },
};

const NAV_LINKS = [
  { path: '/', label: 'Home', icon: HomeIcon },
  { path: '/predict', label: 'Predict', icon: Zap },
  { path: '/structure', label: '3D Studio', icon: Boxes },
  { path: '/mutation', label: 'Mutation', icon: Dna },
  { path: '/network', label: '2D Graph', icon: Network },
  { path: '/network-3d', label: '3D Graph', icon: Share2 },
  { path: '/drug-targets', label: 'Drugs', icon: Pill },
  { path: '/benchmark', label: 'Benchmark', icon: BarChart3, badge: 'Validation' },
  { path: '/compare', label: 'Compare', icon: GitCompare },
  { path: '/assistant', label: 'AI Copilot', icon: Bot },
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
            <nav className="space-y-1 overflow-y-auto max-h-[calc(100vh-220px)] no-scrollbar pr-1">
              <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest px-2 mb-2">Research Modules</p>
              {NAV_LINKS.map((link) => {
                const isActive = location.pathname === link.path;
                const Icon = link.icon;
                return (
                  <Link
                    key={link.path}
                    to={link.path}
                    className={`px-3.5 py-2 rounded-2xl text-xs font-semibold transition-all flex items-center justify-between ${
                      isActive 
                        ? 'text-emerald-700 bg-emerald-50 shadow-sm border border-emerald-200/60 font-bold' 
                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/70'
                    }`}
                  >
                    <div className="flex items-center gap-2.5">
                      <Icon size={15} className={isActive ? 'text-emerald-600' : 'text-slate-400'} />
                      <span>{link.label}</span>
                    </div>
                    {link.badge && (
                      <span className="text-[9px] bg-emerald-100 text-emerald-800 px-1.5 py-0.5 rounded-md font-bold border border-emerald-200">
                        {link.badge}
                      </span>
                    )}
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
        <header className="fixed top-3 left-0 right-0 z-50 px-3 md:px-6 max-w-[1440px] mx-auto flex items-center justify-between gap-3 pointer-events-none">
          
          {/* PART 1: LEFT BRAND LOGO */}
          <motion.div 
            initial={{ y: -15, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="pointer-events-auto shrink-0"
          >
            <Link to="/" className="flex items-center gap-2.5 group">
              <div className="w-9 h-9 bg-gradient-to-tr from-emerald-600 to-teal-700 rounded-full flex items-center justify-center text-white shadow-md shadow-emerald-700/20 group-hover:scale-105 transition-transform">
                <Activity size={18} className="stroke-[2.5]" />
              </div>
              <div className="flex flex-col">
                <div className="flex items-baseline">
                  <span className="text-lg font-extrabold text-slate-800 tracking-tight">TransGraph-</span>
                  <span className="text-lg font-black text-emerald-600 italic">PPI</span>
                </div>
                <span className="text-[11px] text-slate-500 tracking-wide -mt-1 font-semibold" style={{ fontFamily: "'Dancing Script', cursive" }}>
                  From Proteins to Possibilities
                </span>
              </div>
            </Link>
          </motion.div>

          {/* PART 2: CENTER FLOATING PILL */}
          <motion.nav 
            initial={{ y: -20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="pointer-events-auto flex items-center gap-1 bg-white/95 backdrop-blur-xl px-2.5 py-1.5 rounded-full shadow-lg shadow-slate-900/5 border border-slate-200/80 overflow-x-auto no-scrollbar max-w-[65vw]"
          >
            {NAV_LINKS.map((link) => {
              const isActive = location.pathname === link.path;
              const Icon = link.icon;
              return (
                <Link
                  key={link.path}
                  to={link.path}
                  className={`px-3 py-1.5 rounded-full text-xs font-semibold transition-all flex items-center gap-1.5 shrink-0 ${
                    isActive 
                      ? 'text-emerald-700 bg-emerald-50 shadow-sm border border-emerald-300/80 font-bold' 
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/70'
                  }`}
                >
                  <Icon size={14} className={isActive ? 'text-emerald-600 stroke-[2.2]' : 'text-slate-400'} />
                  <span className="hidden xl:inline-block">{link.label}</span>
                </Link>
              );
            })}
          </motion.nav>

          {/* PART 3: RIGHT SEARCH & PROFILE */}
          <motion.div 
            initial={{ y: -15, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.1 }}
            className="pointer-events-auto flex items-center gap-2 shrink-0"
          >
            {/* Quick Search Trigger */}
            <button 
              onClick={() => setIsSearchOpen(true)}
              className="flex items-center gap-2.5 bg-white/90 hover:bg-slate-50 px-3.5 py-1.5 rounded-full text-slate-400 transition-colors text-xs font-medium border border-slate-200/80 shadow-sm"
              title="Search Platform (Ctrl + K)"
            >
              <Search size={14} className="text-slate-400" />
              <span className="hidden md:inline-block text-[11px] text-slate-500 font-medium">Search proteins...</span>
              <kbd className="hidden sm:flex items-center gap-0.5 px-1.5 py-0.5 bg-slate-100 rounded text-[9px] font-bold text-slate-500 border border-slate-200">
                ⌘ K
              </kbd>
            </button>

            {/* Profile Avatar with Online Dot */}
            <div className="relative group cursor-pointer" title="Computational Biology Session Active">
              <div className="w-8 h-8 rounded-full bg-slate-200 border border-slate-300 flex items-center justify-center text-slate-600 group-hover:bg-slate-300 transition-colors">
                <svg className="w-4 h-4 fill-slate-500" viewBox="0 0 24 24">
                  <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
                </svg>
              </div>
              <span className="w-2.5 h-2.5 bg-emerald-500 rounded-full border-2 border-white absolute -top-0.5 -right-0.5 shadow-sm" />
            </div>

            {/* Layout Switcher (Compact icon button) */}
            <button
              onClick={toggleOrientation}
              className="p-2 rounded-full bg-white/90 hover:bg-slate-100 text-slate-500 border border-slate-200/80 shadow-sm transition-all"
              title="Toggle Vertical / Horizontal Navigation"
            >
              <PanelLeft size={14} className="text-slate-600" />
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

