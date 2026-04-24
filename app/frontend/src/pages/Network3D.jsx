import React, { useState, useEffect, useRef } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import { motion } from 'framer-motion';
import { Globe, ZoomIn, ZoomOut, Maximize, Loader2 } from 'lucide-react';

const NetworkExplorer3D = () => {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const fgRef = useRef();

  useEffect(() => {
    // Generate a beautiful sample interactome for demonstration
    // In a real scenario, this would come from the ppi_graph.pt exported as JSON
    const nodes = [...Array(100).keys()].map(i => ({ 
      id: i, 
      name: `Protein-${i}`,
      val: Math.random() * 20 + 5,
      color: i % 5 === 0 ? '#22D3EE' : '#6366F1'
    }));
    
    const links = [...Array(100).keys()].map(() => ({
      source: Math.floor(Math.random() * 100),
      target: Math.floor(Math.random() * 100)
    }));

    setGraphData({ nodes, links });
    setLoading(false);
  }, []);

  return (
    <div className="h-[calc(100vh-120px)] w-full relative bg-slate-950 rounded-3xl overflow-hidden border border-white/5 shadow-2xl">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-950/80 z-50 backdrop-blur-sm">
          <Loader2 className="animate-spin text-scientific-primary" size={48} />
        </div>
      )}

      {/* 3D Graph */}
      {!loading && (
        <ForceGraph3D
          ref={fgRef}
          graphData={graphData}
          backgroundColor="#020617"
          nodeLabel="name"
          nodeColor={node => node.color}
          nodeVal={node => node.val}
          nodeOpacity={0.9}
          linkWidth={0.5}
          linkColor={() => '#ffffff22'}
          showNavInfo={false}
        />
      )}

      {/* Overlay UI */}
      <div className="absolute top-8 left-8 z-10 space-y-4">
        <motion.div 
          initial={{ x: -20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          className="p-6 bg-slate-900/60 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl"
        >
          <div className="flex items-center gap-3 mb-2">
            <Globe className="text-scientific-primary" size={20} />
            <h2 className="text-xl font-bold text-white tracking-tight">Interactome 3D</h2>
          </div>
          <p className="text-xs text-slate-400 font-medium uppercase tracking-widest">Global Interaction Topography</p>
          
          <div className="mt-6 space-y-3">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-cyan-400" />
              <span className="text-[10px] text-slate-300 font-bold uppercase">Hub Proteins (High Degree)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-indigo-500" />
              <span className="text-[10px] text-slate-300 font-bold uppercase">Signal Transducers</span>
            </div>
          </div>
        </motion.div>

        <div className="flex flex-col gap-2">
          <button className="p-3 bg-white/5 hover:bg-white/10 backdrop-blur-md rounded-xl border border-white/10 text-white transition-all shadow-lg">
            <ZoomIn size={18} />
          </button>
          <button className="p-3 bg-white/5 hover:bg-white/10 backdrop-blur-md rounded-xl border border-white/10 text-white transition-all shadow-lg">
            <ZoomOut size={18} />
          </button>
          <button className="p-3 bg-white/5 hover:bg-white/10 backdrop-blur-md rounded-xl border border-white/10 text-white transition-all shadow-lg">
            <Maximize size={18} />
          </button>
        </div>
      </div>

      <div className="absolute bottom-8 right-8 z-10">
        <div className="px-4 py-2 bg-scientific-primary/20 backdrop-blur-xl border border-scientific-primary/30 rounded-full">
          <span className="text-[10px] font-black text-scientific-primary uppercase tracking-widest">Real-Time WebGL Rendering</span>
        </div>
      </div>
    </div>
  );
};

export default NetworkExplorer3D;
