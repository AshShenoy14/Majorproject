import { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import {
  Search,
  Activity,
  Zap,
  Dna,
  Boxes,
  Pill,
  Bot,
  Home as HomeIcon,
  BarChart3,
  Network,
  Orbit,
  GitCompare,
  Lightbulb,
  Info,
  ChevronDown,
  Menu,
  X,
} from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import FloatingGuide from './FloatingGuide';
import { TechDetailsSwitch } from './TechDetails';

// Main menu: the pages that answer the project's questions.
const MAIN_LINKS = [
  { path: '/', label: 'Home', icon: HomeIcon },
  { path: '/predict', label: 'Predict', icon: Zap },
  { path: '/drug-targets', label: 'Drug Targets', icon: Pill },
  { path: '/benchmark', label: 'Model Results', icon: BarChart3 },
  { path: '/mutation', label: 'Mutations', icon: Dna },
];

// Extra tools, under "More".
const MORE_LINKS = [
  { path: '/how-it-works', label: 'How it works', description: 'The project explained in 5 simple steps', icon: Lightbulb },
  { path: '/assistant', label: 'AI Assistant', description: 'Ask questions about proteins in plain English', icon: Bot },
  { path: '/structure', label: '3D Protein Viewer', description: 'See the 3D shape of a protein', icon: Boxes },
  { path: '/network', label: 'Protein Network', description: 'Explore which proteins are connected', icon: Network },
  { path: '/network-3d', label: '3D Network', description: 'The same network in 3D', icon: Orbit },
  { path: '/compare', label: 'Normal vs Mutant', description: 'Compare a protein before and after a change', icon: GitCompare },
  { path: '/about', label: 'About the project', description: 'Team, architecture and technical specs', icon: Info },
];

// Titles used by the search box (plain language first).
const PAGE_META = {
  '/': { title: 'Home', subtitle: 'What this app does and where to start' },
  '/predict': { title: 'Predict', subtitle: 'Check whether two proteins are likely to interact' },
  '/drug-targets': { title: 'Drug Targets', subtitle: 'Proteins that could be good targets for medicines' },
  '/benchmark': { title: 'Model Results', subtitle: 'How accurate the model is, and where it falls short' },
  '/mutation': { title: 'Mutations', subtitle: 'See how changing one amino acid affects an interaction' },
  '/how-it-works': { title: 'How it works', subtitle: 'The project explained in 5 simple steps' },
  '/assistant': { title: 'AI Assistant', subtitle: 'Ask questions about proteins' },
  '/structure': { title: '3D Protein Viewer', subtitle: 'See the 3D shape of a protein' },
  '/network': { title: 'Protein Network', subtitle: 'Explore which proteins are connected' },
  '/network-3d': { title: '3D Network', subtitle: 'The protein network in 3D' },
  '/compare': { title: 'Normal vs Mutant', subtitle: 'Compare a protein before and after a change' },
  '/about': { title: 'About the project', subtitle: 'Team, architecture and technical specs' },
};

const Layout = ({ children }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [moreOpen, setMoreOpen] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const moreRef = useRef(null);

  // Close menus when the page changes
  useEffect(() => {
    setMoreOpen(false);
    setMobileOpen(false);
  }, [location.pathname]);

  // Ctrl+K / Cmd+K opens search, Escape closes any open menu
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsSearchOpen(true);
      }
      if (e.key === 'Escape') {
        setIsSearchOpen(false);
        setMoreOpen(false);
        setMobileOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Close "More" when clicking elsewhere
  useEffect(() => {
    if (!moreOpen) return;
    const onClick = (e) => {
      if (moreRef.current && !moreRef.current.contains(e.target)) setMoreOpen(false);
    };
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, [moreOpen]);

  const filteredPages = Object.entries(PAGE_META).filter(([, data]) =>
    data.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    data.subtitle.toLowerCase().includes(searchQuery.toLowerCase())
  );
  const moreActive = MORE_LINKS.some(l => l.path === location.pathname);

  const linkClass = (active) =>
    `px-3 py-1.5 rounded-full text-sm font-semibold transition-all flex items-center gap-1.5 whitespace-nowrap focus-visible:outline focus-visible:outline-2 focus-visible:outline-emerald-600 ${
      active
        ? 'text-emerald-800 bg-emerald-50 border border-emerald-200'
        : 'text-slate-700 hover:text-slate-900 hover:bg-slate-100'
    }`;

  return (
    <div className="min-h-screen bg-slate-50 relative overflow-x-hidden font-inter text-slate-800 pb-16">
      <a href="#main-content" className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-[200] focus:bg-white focus:px-4 focus:py-2 focus:rounded-lg focus:shadow">
        Skip to content
      </a>

      <header className="fixed top-4 left-0 right-0 z-50 px-4 md:px-8 max-w-7xl mx-auto flex items-center justify-between gap-3 pointer-events-none">
        <motion.nav
          aria-label="Main"
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="pointer-events-auto flex items-center gap-1 bg-white/95 backdrop-blur-xl px-3 py-1.5 rounded-full shadow-lg shadow-slate-900/5 border border-slate-200"
        >
          <Link to="/" className="flex items-center gap-2 pr-3 border-r border-slate-200 group shrink-0" aria-label="TransGraph home">
            <div className="w-8 h-8 bg-gradient-to-tr from-emerald-600 to-teal-500 rounded-full flex items-center justify-center text-white shadow-md shadow-emerald-500/20">
              <Activity size={16} aria-hidden="true" />
            </div>
            <span className="text-lg tracking-wide hidden sm:inline-block font-bold" style={{ fontFamily: "'Dancing Script', cursive" }}>
              Trans<span className="text-emerald-700">Graph</span>
            </span>
          </Link>

          {/* Desktop links */}
          <div className="hidden md:flex items-center gap-0.5">
            {MAIN_LINKS.map((link) => {
              const active = location.pathname === link.path;
              const Icon = link.icon;
              return (
                <Link key={link.path} to={link.path} className={linkClass(active)} aria-current={active ? 'page' : undefined}>
                  <Icon size={15} aria-hidden="true" className={active ? 'text-emerald-700' : 'text-slate-500'} />
                  <span>{link.label}</span>
                </Link>
              );
            })}

            <div className="relative" ref={moreRef}>
              <button
                type="button"
                onClick={() => setMoreOpen(v => !v)}
                aria-expanded={moreOpen}
                aria-haspopup="true"
                className={linkClass(moreActive)}
              >
                <span>More</span>
                <ChevronDown size={15} aria-hidden="true" className={`transition-transform ${moreOpen ? 'rotate-180' : ''}`} />
              </button>
              <AnimatePresence>
                {moreOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: -6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -6 }}
                    className="absolute left-0 mt-2 w-80 bg-white rounded-2xl shadow-xl border border-slate-200 p-2"
                  >
                    {MORE_LINKS.map((link) => {
                      const Icon = link.icon;
                      const active = location.pathname === link.path;
                      return (
                        <Link
                          key={link.path}
                          to={link.path}
                          aria-current={active ? 'page' : undefined}
                          className={`flex items-start gap-3 p-2.5 rounded-xl transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-emerald-600 ${active ? 'bg-emerald-50' : 'hover:bg-slate-100'}`}
                        >
                          <Icon size={18} aria-hidden="true" className="text-emerald-700 mt-0.5 shrink-0" />
                          <span>
                            <span className="block text-sm font-bold text-slate-800">{link.label}</span>
                            <span className="block text-xs text-slate-600">{link.description}</span>
                          </span>
                        </Link>
                      );
                    })}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Mobile menu button */}
          <button
            type="button"
            className="md:hidden p-2 rounded-full text-slate-700 hover:bg-slate-100"
            onClick={() => setMobileOpen(v => !v)}
            aria-expanded={mobileOpen}
            aria-label={mobileOpen ? 'Close menu' : 'Open menu'}
          >
            {mobileOpen ? <X size={20} aria-hidden="true" /> : <Menu size={20} aria-hidden="true" />}
          </button>
        </motion.nav>

        <motion.div
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="pointer-events-auto flex items-center gap-1 bg-white/95 backdrop-blur-xl px-2 py-1.5 rounded-full shadow-lg shadow-slate-900/5 border border-slate-200"
        >
          <TechDetailsSwitch />
          <button
            type="button"
            onClick={() => setIsSearchOpen(true)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-full text-slate-700 hover:bg-slate-100 transition-colors text-sm font-semibold focus-visible:outline focus-visible:outline-2 focus-visible:outline-emerald-600"
            aria-label="Search pages (Ctrl + K)"
            title="Search pages (Ctrl + K)"
          >
            <Search size={15} aria-hidden="true" />
            <span className="hidden lg:inline-block">Search</span>
          </button>
        </motion.div>
      </header>

      {/* Mobile menu panel */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.nav
            aria-label="Mobile"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="md:hidden fixed top-20 left-4 right-4 z-50 bg-white rounded-2xl shadow-xl border border-slate-200 p-3 space-y-1 max-h-[75vh] overflow-y-auto"
          >
            {[...MAIN_LINKS, ...MORE_LINKS].map((link) => {
              const Icon = link.icon;
              const active = location.pathname === link.path;
              return (
                <Link key={link.path} to={link.path} aria-current={active ? 'page' : undefined}
                  className={`flex items-center gap-3 p-3 rounded-xl text-sm font-semibold ${active ? 'bg-emerald-50 text-emerald-800' : 'text-slate-700 hover:bg-slate-100'}`}>
                  <Icon size={18} aria-hidden="true" className="text-emerald-700" />
                  {link.label}
                </Link>
              );
            })}
          </motion.nav>
        )}
      </AnimatePresence>

      {/* Search dialog */}
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
              role="dialog"
              aria-modal="true"
              aria-label="Search pages"
              initial={{ scale: 0.95, opacity: 0, y: -20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.95, opacity: 0, y: -20 }}
              className="bg-white w-full max-w-2xl rounded-3xl shadow-2xl overflow-hidden border border-slate-100"
              onClick={e => e.stopPropagation()}
            >
              <div className="p-5 border-b border-slate-100 flex items-center gap-3">
                <Search className="text-slate-500" size={20} aria-hidden="true" />
                <input
                  autoFocus
                  aria-label="Search pages"
                  placeholder="Search pages, e.g. 'drug' or 'mutation'"
                  className="flex-1 bg-transparent border-none outline-none text-lg font-medium text-slate-800 placeholder:text-slate-500"
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                />
                <kbd className="px-2 py-1 bg-slate-100 rounded-md text-xs font-bold text-slate-600">Esc</kbd>
              </div>
              <div className="max-h-[360px] overflow-y-auto p-3 space-y-1">
                {filteredPages.map(([path, data]) => (
                  <button
                    type="button"
                    key={path}
                    onClick={() => { navigate(path); setIsSearchOpen(false); }}
                    className="w-full text-left p-3 rounded-2xl hover:bg-emerald-50 transition-all"
                  >
                    <p className="text-sm font-bold text-slate-800">{data.title}</p>
                    <p className="text-xs text-slate-600">{data.subtitle}</p>
                  </button>
                ))}
                {filteredPages.length === 0 && <p className="p-3 text-sm text-slate-600">No page matches "{searchQuery}".</p>}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <main id="main-content" className="px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto pt-24">
        <div className="animate-in fade-in slide-in-from-bottom-3 duration-500">
          {children}
        </div>
      </main>

      <FloatingGuide />
    </div>
  );
};

export default Layout;
