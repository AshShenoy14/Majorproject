import React from 'react';
import { useLocation } from 'react-router-dom';
import Sidebar from './Sidebar';

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
  const meta = PAGE_META[location.pathname] ?? { title: 'Bioinformatics Analysis', subtitle: '' };

  return (
    <div className="flex min-h-screen bg-slate-50">
      <Sidebar />
      <main className="flex-1 ml-[19rem] mr-8 my-8 transition-all duration-300">
        <header className="mb-12 flex justify-between items-center">
          <div>
            <h2 className="text-[10px] font-black text-emerald-600 uppercase tracking-[0.4em] mb-1">
              {meta.title}
            </h2>
            {meta.subtitle && (
              <p className="text-xl text-slate-800 font-black tracking-tight">{meta.subtitle}</p>
            )}
          </div>
          <div className="flex items-center gap-4">
            <div className="glass-card bg-white px-6 py-3 flex items-center gap-3 text-[10px] font-black text-slate-400 tracking-widest border-none">
              <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)] animate-pulse"></div>
              SYSTEM ACTIVE
            </div>
          </div>
        </header>

        <div className="animate-in fade-in slide-in-from-bottom-4 duration-700">
          {children}
        </div>
      </main>
    </div>
  );
};

export default Layout;
