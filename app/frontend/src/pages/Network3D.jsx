import React, { useState, useEffect, useRef, useMemo } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import { motion } from 'framer-motion';
import { Globe, ZoomIn, ZoomOut, Maximize, Loader2, AlertTriangle } from 'lucide-react';
import { ppiService } from '../services/api';

// Number of real training-set positive interactions requested from GET /network
const EDGE_LIMIT = 300;

const NetworkExplorer3D = () => {
  const [network, setNetwork] = useState({ nodes: [], edges: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const fgRef = useRef();

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await ppiService.getNetwork(EDGE_LIMIT);
        if (!cancelled) setNetwork({ nodes: res.data?.nodes || [], edges: res.data?.edges || [] });
      } catch (err) {
        console.error('Failed to load interaction network:', err);
        if (!cancelled) setError(err?.response?.data?.detail || 'Could not load the interaction network from the backend.');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, []);

  // Node size/colour derive only from the real edges returned by the backend (degree within the displayed subgraph).
  const graphData = useMemo(() => {
    const degree = {};
    network.edges.forEach(e => {
      degree[e.source] = (degree[e.source] || 0) + 1;
      degree[e.target] = (degree[e.target] || 0) + 1;
    });
    const sorted = Object.values(degree).sort((a, b) => a - b);
    const cutoff = sorted.length ? sorted[Math.floor(sorted.length * 0.9)] : Infinity;
    return {
      nodes: network.nodes.map(n => ({
        id: n.id,
        name: `${n.label} (degree ${degree[n.id] || 0})`,
        val: 2 + (degree[n.id] || 0),
        color: (degree[n.id] || 0) >= cutoff ? '#22D3EE' : '#6366F1'
      })),
      links: network.edges.map(e => ({ source: e.source, target: e.target }))
    };
  }, [network]);

  const isEmpty = !loading && !error && graphData.nodes.length === 0;

  return (
    <div className="h-[calc(100vh-120px)] w-full relative bg-slate-950 rounded-xl overflow-hidden border border-slate-800 shadow-2xl">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-950/80 z-50 backdrop-blur-sm">
          <Loader2 className="animate-spin text-scientific-primary" size={48} />
        </div>
      )}

      {error && (
        <div className="absolute inset-0 flex items-center justify-center z-40">
          <div className="max-w-md p-6 bg-rose-50 border border-rose-200 rounded-2xl flex items-start gap-3">
            <AlertTriangle className="text-rose-500 flex-shrink-0" size={20} />
            <div>
              <p className="text-sm font-black text-rose-700">Interaction network unavailable</p>
              <p className="text-sm text-rose-600 mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      {isEmpty && (
        <div className="absolute inset-0 flex items-center justify-center z-40">
          <p className="text-sm font-semibold text-slate-300">The backend returned no interactions to display.</p>
        </div>
      )}

      {/* 3D Graph */}
      {!loading && !error && !isEmpty && (
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
          <p className="text-[9px] text-slate-400 font-black uppercase tracking-[0.2em] mb-6">Training-set positive interactions</p>
          
          <div className="space-y-3">
            <div className="flex items-center gap-3 p-2 hover:bg-slate-50 rounded-xl transition-colors">
              <div className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.5)]" />
              <span className="text-[10px] text-slate-600 font-black uppercase tracking-wider">Top 10% by degree (in this subgraph)</span>
            </div>
            <div className="flex items-center gap-3 p-2 hover:bg-slate-50 rounded-xl transition-colors">
              <div className="w-2.5 h-2.5 rounded-full bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.5)]" />
              <span className="text-[10px] text-slate-600 font-black uppercase tracking-wider">Other proteins</span>
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
          <span className="text-[10px] font-black text-white uppercase tracking-[0.2em]">WebGL Renderer</span>
        </div>
      </div>
    </div>
  );
};

export default NetworkExplorer3D;
