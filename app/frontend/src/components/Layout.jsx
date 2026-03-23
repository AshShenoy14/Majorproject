import React from 'react';
import Sidebar from './Sidebar';

const Layout = ({ children }) => {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 ml-64 p-8 transition-all duration-300">
        <header className="mb-8 flex justify-between items-center">
          <div>
            <h2 className="text-sm font-bold text-scientific-primary uppercase tracking-[0.2em]">
              Bioinformatics Analysis
            </h2>
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
