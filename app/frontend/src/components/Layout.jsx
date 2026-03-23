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
      <main className="flex-1 ml-64 p-8 transition-all duration-300">
        <header className="mb-8 flex justify-between items-center">
          <div>
            <h2 className="text-sm font-bold text-scientific-primary uppercase tracking-[0.2em]">
              {meta.title}
            </h2>
            {meta.subtitle && (
              <p className="text-xs text-gray-400 mt-0.5 tracking-wide">{meta.subtitle}</p>
            )}
          </div>
          <div className="flex items-center gap-4">
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
