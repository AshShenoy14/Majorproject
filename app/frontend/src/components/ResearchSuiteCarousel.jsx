import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  Zap,
  Boxes,
  Dna,
  GitCompare,
  Share2,
  Globe,
  Pill,
  BarChart3,
  ArrowRight,
  ChevronLeft,
  ChevronRight,
  Sparkles
} from 'lucide-react';

const RESEARCH_ENGINES = [
  {
    id: 'predict',
    name: 'Interaction Predictor',
    badge: 'Prediction',
    isCore: true,
    desc: 'ESM-2 + GraphSAGE stacking ensemble with Platt calibration and SHAP feature attribution.',
    path: '/predict',
    icon: Zap,
    gradient: 'from-emerald-600 to-teal-500',
    accentColor: 'text-emerald-600',
    badgeColor: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    shadowColor: 'shadow-emerald-500/15',
    borderActive: 'border-emerald-300',
  },
  {
    id: 'structure',
    name: '3D Molecular Studio',
    badge: 'Structure',
    desc: 'Full atomic protein structure visualization rendered in real time with PDBe Mol*.',
    path: '/structure',
    icon: Boxes,
    gradient: 'from-cyan-600 to-blue-500',
    accentColor: 'text-cyan-600',
    badgeColor: 'bg-cyan-50 text-cyan-700 border-cyan-200',
    shadowColor: 'shadow-cyan-500/15',
    borderActive: 'border-cyan-300',
  },
  {
    id: 'mutation',
    name: 'Mutation Scanner',
    badge: 'Mutation',
    desc: 'In-silico amino-acid substitution analysis for identifying mutation-sensitive regions.',
    path: '/mutation',
    icon: Dna,
    gradient: 'from-teal-600 to-emerald-500',
    accentColor: 'text-teal-600',
    badgeColor: 'bg-teal-50 text-teal-700 border-teal-200',
    shadowColor: 'shadow-teal-500/15',
    borderActive: 'border-teal-300',
  },
  {
    id: 'compare',
    name: 'WT vs Mutant Comparator',
    badge: 'Comparator',
    desc: 'Side-by-side comparison of native wildtype and mutant protein analysis.',
    path: '/compare',
    icon: GitCompare,
    gradient: 'from-amber-500 to-orange-500',
    accentColor: 'text-amber-600',
    badgeColor: 'bg-amber-50 text-amber-700 border-amber-200',
    shadowColor: 'shadow-amber-500/15',
    borderActive: 'border-amber-300',
  },
  {
    id: 'network',
    name: '2D Interactome Explorer',
    badge: 'Network',
    desc: 'Explore shortest paths, hub centralities, shared neighbors, and functional subnetworks.',
    path: '/network',
    icon: Share2,
    gradient: 'from-violet-600 to-indigo-500',
    accentColor: 'text-violet-600',
    badgeColor: 'bg-violet-50 text-violet-700 border-violet-200',
    shadowColor: 'shadow-violet-500/15',
    borderActive: 'border-violet-300',
  },
  {
    id: 'network-3d',
    name: '3D Force Topography',
    badge: 'Topography',
    desc: 'Interactive 3D force-directed visualization of protein interaction networks and hubs.',
    path: '/network-3d',
    icon: Globe,
    gradient: 'from-purple-600 to-pink-500',
    accentColor: 'text-purple-600',
    badgeColor: 'bg-purple-50 text-purple-700 border-purple-200',
    shadowColor: 'shadow-purple-500/15',
    borderActive: 'border-purple-300',
  },
  {
    id: 'drug-targets',
    name: 'Drug Target Insights',
    badge: 'Drug Discovery',
    desc: 'Explore ChEMBL drug-target information and Therapeutic Target Priority Scores.',
    path: '/drug-targets',
    icon: Pill,
    gradient: 'from-blue-600 to-cyan-500',
    accentColor: 'text-blue-600',
    badgeColor: 'bg-blue-50 text-blue-700 border-blue-200',
    shadowColor: 'shadow-blue-500/15',
    borderActive: 'border-blue-300',
  },
  {
    id: 'benchmark',
    name: 'Empirical Benchmarks',
    badge: 'Validation',
    desc: 'View bootstrap confidence intervals, cold-start evaluation, SHS27k, HuRI, and other validation results.',
    path: '/benchmark',
    icon: BarChart3,
    gradient: 'from-rose-500 to-amber-500',
    accentColor: 'text-rose-600',
    badgeColor: 'bg-rose-50 text-rose-700 border-rose-200',
    shadowColor: 'shadow-rose-500/15',
    borderActive: 'border-rose-300',
  },
];

const ResearchSuiteCarousel = () => {
  const [activeIndex, setActiveIndex] = useState(0);
  const total = RESEARCH_ENGINES.length;

  const handleNext = useCallback(() => {
    setActiveIndex((prev) => (prev + 1) % total);
  }, [total]);

  const handlePrev = useCallback(() => {
    setActiveIndex((prev) => (prev - 1 + total) % total);
  }, [total]);

  // Keyboard arrow navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'ArrowRight') handleNext();
      if (e.key === 'ArrowLeft') handlePrev();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleNext, handlePrev]);

  // Circular offset calculation
  const getOffset = (index) => {
    let diff = index - activeIndex;
    if (diff > total / 2) diff -= total;
    if (diff < -total / 2) diff += total;
    return diff;
  };

  return (
    <div className="w-full relative flex flex-col items-center select-none py-4">
      {/* ── STACKED CARD CAROUSEL VIEWPORT ── */}
      <div className="relative w-full max-w-4xl h-[360px] sm:h-[370px] md:h-[390px] flex items-center justify-center overflow-visible">
        {RESEARCH_ENGINES.map((card, index) => {
          const diff = getOffset(index);
          const isCenter = diff === 0;
          const isVisible = Math.abs(diff) <= 2;

          // Compute transform styles based on circular distance
          let x = 0;
          let y = 0;
          let scale = 1;
          let rotateZ = 0;
          let zIndex = 10;
          let opacity = 0;
          let pointerEvents = 'none';

          if (isCenter) {
            x = 0;
            y = 0;
            scale = 1;
            rotateZ = 0;
            zIndex = 40;
            opacity = 1;
            pointerEvents = 'auto';
          } else if (Math.abs(diff) === 1) {
            // Immediate adjacent cards
            x = diff * 185;
            y = 12;
            scale = 0.91;
            rotateZ = diff * 4.5;
            zIndex = 30;
            opacity = 0.82;
            pointerEvents = 'auto';
          } else if (Math.abs(diff) === 2) {
            // Secondary adjacent cards
            x = diff * 325;
            y = 24;
            scale = 0.82;
            rotateZ = diff * 8;
            zIndex = 20;
            opacity = 0.45;
            pointerEvents = 'auto';
          } else {
            // Distant cards off-canvas
            x = diff > 0 ? 460 : -460;
            y = 35;
            scale = 0.72;
            rotateZ = diff > 0 ? 12 : -12;
            zIndex = 10;
            opacity = 0;
            pointerEvents = 'none';
          }

          const Icon = card.icon;

          return (
            <motion.div
              key={card.id}
              className="absolute"
              style={{ zIndex }}
              animate={{
                x,
                y,
                scale,
                rotateZ,
                opacity,
              }}
              transition={{
                type: 'spring',
                stiffness: 280,
                damping: 28,
                mass: 0.85,
              }}
              // Drag gesture on active card
              drag={isCenter ? 'x' : false}
              dragConstraints={{ left: 0, right: 0 }}
              dragElastic={0.25}
              onDragEnd={(_, info) => {
                const swipeThreshold = 40;
                if (info.offset.x < -swipeThreshold || info.velocity.x < -250) {
                  handleNext();
                } else if (info.offset.x > swipeThreshold || info.velocity.x > 250) {
                  handlePrev();
                }
              }}
              onClick={() => {
                if (!isCenter) {
                  setActiveIndex(index);
                }
              }}
            >
              <div
                className={`w-[320px] sm:w-[370px] md:w-[410px] p-6 sm:p-7 rounded-[2rem] bg-white/95 backdrop-blur-xl border transition-all duration-300 flex flex-col justify-between cursor-pointer ${
                  isCenter
                    ? `${card.borderActive} shadow-2xl ${card.shadowColor} ring-1 ring-emerald-500/10`
                    : 'border-slate-200/80 shadow-md hover:border-slate-300 hover:shadow-lg'
                }`}
                style={{
                  minHeight: '290px',
                }}
              >
                {/* Ambient glow behind core card */}
                {card.isCore && isCenter && (
                  <div className="absolute top-0 right-0 w-48 h-48 bg-emerald-100/30 rounded-full blur-3xl pointer-events-none" />
                )}

                <div>
                  {/* Card Header: Icon + Badge */}
                  <div className="flex items-center justify-between gap-3 mb-4">
                    <div
                      className={`w-12 h-12 rounded-2xl bg-gradient-to-tr ${card.gradient} text-white flex items-center justify-center font-bold shadow-md ${card.shadowColor}`}
                    >
                      <Icon size={22} />
                    </div>

                    {card.isCore ? (
                      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 text-[10px] font-black uppercase tracking-wider">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" /> Core Framework
                      </span>
                    ) : (
                      <span
                        className={`px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-wider border ${card.badgeColor}`}
                      >
                        {card.badge}
                      </span>
                    )}
                  </div>

                  {/* Feature Title */}
                  <h3 className="text-xl font-black text-slate-800 tracking-tight mb-2 group-hover:text-emerald-700 transition-colors">
                    {card.name}
                  </h3>

                  {/* Feature Description */}
                  <p className="text-xs text-slate-500 leading-relaxed mb-4">
                    {card.desc}
                  </p>
                </div>

                {/* Card Action Link */}
                <div className="pt-3 border-t border-slate-100 flex items-center justify-between">
                  {isCenter ? (
                    <Link
                      to={card.path}
                      className="inline-flex items-center justify-between w-full px-4 py-2.5 rounded-xl bg-slate-900 text-white text-xs font-bold hover:bg-emerald-600 transition-all shadow-md group/btn"
                    >
                      <span>Explore Module</span>
                      <ArrowRight size={14} className="group-hover/btn:translate-x-1 transition-transform" />
                    </Link>
                  ) : (
                    <div className="flex items-center justify-between w-full text-slate-400 text-xs font-semibold">
                      <span>Click to focus</span>
                      <ArrowRight size={13} className="text-slate-300" />
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* ── NAVIGATION CONTROLS: ← Previous • 1 / 8 • Next → ── */}
      <div className="flex items-center justify-center gap-5 sm:gap-6 mt-6 z-20">
        <button
          onClick={handlePrev}
          className="px-4 py-2 rounded-xl bg-white border border-slate-200/80 text-slate-700 text-xs font-bold hover:bg-slate-50 hover:border-emerald-300 hover:text-emerald-700 transition-all shadow-sm flex items-center gap-1.5 active:scale-95 cursor-pointer"
          aria-label="Previous Module"
        >
          <ChevronLeft size={16} />
          <span className="hidden sm:inline">Previous</span>
        </button>

        {/* Indicator: 1 / 8 & Subtle Dots */}
        <div className="flex items-center gap-3">
          <span className="font-mono text-xs font-black text-slate-700 tracking-wider">
            {activeIndex + 1} <span className="text-slate-300 font-normal">/</span> {total}
          </span>

          <div className="flex items-center gap-1.5">
            {RESEARCH_ENGINES.map((item, idx) => (
              <button
                key={item.id}
                onClick={() => setActiveIndex(idx)}
                className={`transition-all rounded-full cursor-pointer ${
                  idx === activeIndex
                    ? 'w-6 h-1.5 bg-emerald-600'
                    : 'w-1.5 h-1.5 bg-slate-200 hover:bg-slate-300'
                }`}
                aria-label={`Go to ${item.name}`}
              />
            ))}
          </div>
        </div>

        <button
          onClick={handleNext}
          className="px-4 py-2 rounded-xl bg-white border border-slate-200/80 text-slate-700 text-xs font-bold hover:bg-slate-50 hover:border-emerald-300 hover:text-emerald-700 transition-all shadow-sm flex items-center gap-1.5 active:scale-95 cursor-pointer"
          aria-label="Next Module"
        >
          <span className="hidden sm:inline">Next</span>
          <ChevronRight size={16} />
        </button>
      </div>
    </div>
  );
};

export default ResearchSuiteCarousel;
