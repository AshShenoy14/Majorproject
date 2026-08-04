import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, X, Zap, Dna, Share2, Pill, Globe, ArrowRight, HelpCircle, BookOpen } from 'lucide-react';
import { Link } from 'react-router-dom';

const FloatingGuide = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      {/* FLOATING DOT WIDGET BUTTON */}
      <motion.button
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 z-40 group flex items-center justify-center pointer-events-auto"
        title="Open TransGraph PPI Guide"
      >
        {/* Outer glowing pulse ring */}
        <span className="absolute inset-0 rounded-full bg-emerald-400 opacity-75 animate-ping group-hover:opacity-100" />
        
        {/* Main Floating Dot Button with Logo */}
        <div className="relative w-14 h-14 rounded-full bg-gradient-to-tr from-slate-900 via-emerald-950 to-teal-800 p-0.5 shadow-2xl shadow-emerald-500/30 border border-emerald-400/50 flex items-center justify-center">
          <div className="w-full h-full rounded-full bg-slate-900 flex items-center justify-center group-hover:bg-emerald-600 transition-colors duration-300">
            <Activity className="w-6 h-6 text-emerald-400 group-hover:text-white transition-colors duration-300 animate-pulse" />
          </div>
        </div>

        {/* Floating Tag */}
        <div className="absolute right-16 top-1/2 -translate-y-1/2 bg-slate-900 text-white px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest shadow-xl border border-slate-700 opacity-0 group-hover:opacity-100 transition-opacity duration-300 whitespace-nowrap pointer-events-none">
          TransGraph Guide
        </div>
      </motion.button>

      {/* INTERACTIVE GUIDE MODAL OVERLAY */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-slate-950/60 backdrop-blur-md flex items-end sm:items-center justify-center p-4 sm:p-6 overflow-y-auto"
            onClick={() => setIsOpen(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-white w-full max-w-2xl rounded-[2.5rem] shadow-2xl overflow-hidden border border-slate-100 relative"
            >
              {/* Header */}
              <div className="bg-slate-900 text-white p-6 sm:p-8 relative">
                <button
                  onClick={() => setIsOpen(false)}
                  className="absolute top-6 right-6 p-2 rounded-full bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700 transition-colors"
                >
                  <X size={18} />
                </button>

                <div className="flex items-center gap-3 mb-2">
                  <div className="w-10 h-10 rounded-2xl bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center text-white shadow-lg shadow-emerald-500/30">
                    <Activity size={22} className="animate-pulse" />
                  </div>
                  <div>
                    <h3 className="text-2xl font-black tracking-tight text-white">TransGraph-PPI User Guide</h3>
                    <p className="text-xs text-emerald-400 font-mono">Platform Capabilities & Step-by-Step Instructions</p>
                  </div>
                </div>
              </div>

              {/* Body */}
              <div className="p-6 sm:p-8 space-y-6 max-h-[70vh] overflow-y-auto">
                <p className="text-slate-600 text-xs sm:text-sm font-medium leading-relaxed">
                  Welcome to <strong>TransGraph-PPI</strong>. Our platform harnesses ESM-2 protein language representations and Graph Attention Networks to predict and analyze biological interactions.
                </p>

                {/* Modules Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {/* Module 1 */}
                  <div className="p-4 rounded-2xl bg-slate-50 border border-slate-100 space-y-2">
                    <div className="flex items-center gap-2 text-emerald-600 font-bold text-xs uppercase tracking-wider">
                      <Zap size={16} /> 1. Interaction Predictor
                    </div>
                    <p className="text-xs text-slate-500 leading-normal">
                      Input protein IDs (UniProt/Ensembl), raw amino acid FASTA sequences, or choose demo case studies to calculate binding probability.
                    </p>
                    <Link
                      to="/predict"
                      onClick={() => setIsOpen(false)}
                      className="inline-flex items-center gap-1 text-[11px] font-bold text-emerald-600 hover:text-emerald-700 pt-1"
                    >
                      Open Predictor <ArrowRight size={12} />
                    </Link>
                  </div>

                  {/* Module 2 */}
                  <div className="p-4 rounded-2xl bg-slate-50 border border-slate-100 space-y-2">
                    <div className="flex items-center gap-2 text-teal-600 font-bold text-xs uppercase tracking-wider">
                      <Dna size={16} /> 2. Mutation Analysis
                    </div>
                    <p className="text-xs text-slate-500 leading-normal">
                      Simulate single-point amino acid mutations in silico to evaluate binding affinity gains or disruptions.
                    </p>
                    <Link
                      to="/mutation"
                      onClick={() => setIsOpen(false)}
                      className="inline-flex items-center gap-1 text-[11px] font-bold text-teal-600 hover:text-teal-700 pt-1"
                    >
                      Open Mutation Scanner <ArrowRight size={12} />
                    </Link>
                  </div>

                  {/* Module 3 */}
                  <div className="p-4 rounded-2xl bg-slate-50 border border-slate-100 space-y-2">
                    <div className="flex items-center gap-2 text-indigo-600 font-bold text-xs uppercase tracking-wider">
                      <Share2 size={16} /> 3. 3D Interactome
                    </div>
                    <p className="text-xs text-slate-500 leading-normal">
                      Explore interactive 3D graph networks. Zoom, drag, and inspect topological centrality hubs in real time.
                    </p>
                    <Link
                      to="/network-3d"
                      onClick={() => setIsOpen(false)}
                      className="inline-flex items-center gap-1 text-[11px] font-bold text-indigo-600 hover:text-indigo-700 pt-1"
                    >
                      Open 3D Network <ArrowRight size={12} />
                    </Link>
                  </div>

                  {/* Module 4 */}
                  <div className="p-4 rounded-2xl bg-slate-50 border border-slate-100 space-y-2">
                    <div className="flex items-center gap-2 text-purple-600 font-bold text-xs uppercase tracking-wider">
                      <Globe size={16} /> 4. Cross-Species & Drugs
                    </div>
                    <p className="text-xs text-slate-500 leading-normal">
                      Test zero-shot generalization across model organisms and screen ChEMBL drug target candidates.
                    </p>
                    <Link
                      to="/zero-shot"
                      onClick={() => setIsOpen(false)}
                      className="inline-flex items-center gap-1 text-[11px] font-bold text-purple-600 hover:text-purple-700 pt-1"
                    >
                      Open Cross-Species <ArrowRight size={12} />
                    </Link>
                  </div>
                </div>

                {/* Helpful Tip */}
                <div className="p-4 bg-emerald-50/70 border border-emerald-200/60 rounded-2xl flex items-start gap-3">
                  <BookOpen size={18} className="text-emerald-600 shrink-0 mt-0.5" />
                  <p className="text-xs text-emerald-900 leading-relaxed font-medium">
                    <strong>Quick Tip:</strong> Press <kbd className="px-1.5 py-0.5 bg-white rounded text-[10px] font-bold border border-emerald-300">Ctrl + K</kbd> anywhere on the site to quickly search and switch between features!
                  </p>
                </div>
              </div>

              {/* Footer */}
              <div className="p-4 bg-slate-50 border-t border-slate-100 flex justify-end">
                <button
                  onClick={() => setIsOpen(false)}
                  className="px-6 py-2.5 bg-slate-900 hover:bg-slate-800 text-white rounded-xl text-xs font-bold transition-colors"
                >
                  Got It
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default FloatingGuide;
