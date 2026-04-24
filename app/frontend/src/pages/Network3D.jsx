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
    <div className="h-[calc(100vh-120px)] w-full relative bg-slate-950 rounded-xl overflow-hidden border border-slate-800 shadow-2xl">
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
      <div className="absolute top-8 left-8 z-10 flex flex-col gap-4 pointer-events-none">
        <motion.div 
          initial={{ x: -20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          className="p-6 bg-white/95 backdrop-blur-xl border border-white rounded-[2rem] shadow-2xl pointer-events-auto max-w-[280px]"
        >
          <div className="flex items-center gap-3 mb-1">
            <div className="p-2 bg-emerald-500/10 rounded-xl text-emerald-600">
              <Globe size={20} />
            </div>
            <h2 className="text-xl font-black text-slate-800 tracking-tight">Interactome 3D</h2>
          </div>
          <p className="text-[9px] text-slate-400 font-black uppercase tracking-[0.2em] mb-6">Structural Topography</p>
          
          <div className="space-y-3">
            <div className="flex items-center gap-3 p-2 hover:bg-slate-50 rounded-xl transition-colors">
              <div className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.5)]" />
              <span className="text-[10px] text-slate-600 font-black uppercase tracking-wider">Hub Proteins</span>
            </div>
            <div className="flex items-center gap-3 p-2 hover:bg-slate-50 rounded-xl transition-colors">
              <div className="w-2.5 h-2.5 rounded-full bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.5)]" />
              <span className="text-[10px] text-slate-600 font-black uppercase tracking-wider">Signal Transducers</span>
            </div>
          </div>
        </motion.div>

        <div className="flex flex-row gap-3 pointer-events-auto mt-2">
          {[
            { icon: ZoomIn, action: () => fgRef.current?.zoomIn?.() || fgRef.current?.cameraPosition({ z: fgRef.current.cameraPosition().z - 20 }), label: 'Zoom In' },
            { icon: ZoomOut, action: () => fgRef.current?.zoomOut?.() || fgRef.current?.cameraPosition({ z: fgRef.current.cameraPosition().z + 20 }), label: 'Zoom Out' },
            { icon: Maximize, action: () => {
              if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen();
              } else if (document.exitFullscreen) {
                document.exitFullscreen();
              }
            }, label: 'Toggle Fullscreen' }
          ].map((item, i) => (
            <button 
              key={i}
              onClick={item.action}
              title={item.label}
              className="w-11 h-11 flex items-center justify-center bg-white/95 hover:bg-white backdrop-blur-md rounded-full border border-white text-slate-700 transition-all shadow-[0_8px_20px_rgba(0,0,0,0.1)] hover:shadow-[0_12px_25px_rgba(0,0,0,0.15)] hover:-translate-y-0.5 active:translate-y-0 group"
            >
              <item.icon size={18} className="group-hover:text-emerald-500 transition-colors" />
            </button>
          ))}
        </div>
      </div>

      <div className="absolute bottom-8 right-8 z-10 pointer-events-none">
        <div className="px-5 py-2.5 bg-emerald-500/90 backdrop-blur-xl border border-emerald-400 rounded-full shadow-2xl flex items-center gap-3">
          <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
          <span className="text-[10px] font-black text-white uppercase tracking-[0.2em]">Real-Time WebGL Core</span>
        </div>
      </div>
    </div>
  );
};

export default NetworkExplorer3D;
