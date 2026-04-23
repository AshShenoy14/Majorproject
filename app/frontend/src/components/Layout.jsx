import React from 'react';
import { useLocation } from 'react-router-dom';
import Sidebar from './Sidebar';

const PAGE_META = {
  '/':            { title: 'Dashboard',              subtitle: 'System Overview & Quick Actions' },
  '/predict':     { title: 'Interaction Prediction',  subtitle: 'Predict Protein-Protein Interactions' },
  '/mutation':    { title: 'Mutation Analysis',        subtitle: 'In-Silico Mutation Impact Scanner' },
  '/structure':   { title: 'Structure Viewer',         subtitle: '3D Protein Structure Visualization' },
  '/network':     { title: 'Network Explorer',         subtitle: 'Interactome Graph Analysis' },
  '/drug-targets':{ title: 'Drug Insights',            subtitle: 'Drug Target Discovery & ChEMBL Data' },
  '/assistant':   { title: 'Protein Assistant',        subtitle: 'AI-Powered Biological Query Engine' },
  '/about':       { title: 'About',                    subtitle: 'Project Overview & Model Details' },
  '/zero-shot':   { title: 'Cross-Species Testing',    subtitle: 'Zero-Shot Generalization Evaluation' },
};

const Layout = ({ children }) => {
  const location = useLocation();
  const meta = PAGE_META[location.pathname] ?? { title: 'Bioinformatics Analysis', subtitle: '' };

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 ml-[19rem] mr-6 my-6 p-8 transition-all duration-300">
        <header className="mb-10 flex justify-between items-center">
          <div>
            <h2 className="text-xs font-black text-scientific-primary uppercase tracking-[0.3em] mb-1">
              {meta.title}
            </h2>
            {meta.subtitle && (
              <p className="text-sm text-slate-500 font-medium tracking-tight">{meta.subtitle}</p>
            )}
          </div>
          <div className="flex items-center gap-4">
            <div className="glass-card px-4 py-2 flex items-center gap-2 text-xs font-bold text-slate-600">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
              SYSTEM ONLINE
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
