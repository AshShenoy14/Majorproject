import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import cytoscape from 'cytoscape';
import { 
  Share2, 
  Maximize2, 
  RotateCcw, 
  Info, 
  Search, 
  ChevronRight,
  Activity,
  Database,
  Star
} from 'lucide-react';
import { ppiService } from '../services/api';

const NetworkExplorer = () => {
  const containerRef = useRef(null);
  const cyRef = useRef(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [metrics, setMetrics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const initNetwork = async () => {
      setLoading(true);
      try {
        const [netRes, centralityRes] = await Promise.all([
          ppiService.getNetwork(50),
          ppiService.getCentrality(10)
        ]);

        const elements = [
          ...netRes.data.nodes.map(n => ({ data: { id: n.id, label: n.label } })),
          ...netRes.data.edges.map(e => ({ data: { source: e.source, target: e.target } }))
        ];

        setMetrics(centralityRes.data);

        if (containerRef.current) {
          cyRef.current = cytoscape({
            container: containerRef.current,
            elements: elements,
            style: [
              {
                selector: 'node',
                style: {
                  'background-color': '#0D9488',
                  'label': 'data(label)',
                  'color': '#475569',
                  'font-size': '10px',
                  'text-valign': 'bottom',
                  'text-margin-y': '5px',
                  'width': '30px',
                  'height': '30px',
                  'border-width': '2px',
                  'border-color': '#ffffff',
                  'transition-property': 'background-color, line-color, target-arrow-color, width, height',
                  'transition-duration': '0.3s'
                }
              },
              {
                selector: 'edge',
                style: {
                  'width': 2,
                  'line-color': '#CBD5E1',
                  'curve-style': 'bezier',
                  'opacity': 0.6
                }
              },
              {
                selector: 'node:selected',
                style: {
                  'background-color': '#7C3AED',
                  'width': '40px',
                  'height': '40px',
                  'border-color': '#7C3AED',
                  'border-opacity': 0.2,
                  'border-width': '10px'
                }
              }
            ],
            layout: {
              name: 'cose',
              animate: true,
              componentSpacing: 100,
              nodeRepulsion: 400000,
              idealEdgeLength: 100
            }
          });

          cyRef.current.on('tap', 'node', (evt) => {
            setSelectedNode(evt.target.data());
          });

          cyRef.current.on('tap', (evt) => {
            if (evt.target === cyRef.current) setSelectedNode(null);
          });
        }
      } catch (err) {
        console.error("Network init error:", err);
        setError("Failed to load network data. Check backend connectivity.");
      } finally {
        setLoading(false);
      }
    };

    initNetwork();

    return () => {
      if (cyRef.current) cyRef.current.destroy();
    };
  }, []);

  const resetLayout = () => {
    if (cyRef.current) {
      cyRef.current.layout({ name: 'cose', animate: true }).run();
      cyRef.current.fit();
    }
  };

  return (
    <div className="h-[calc(100vh-160px)] flex gap-6 overflow-hidden">
      {/* Network Canvas */}
      <div className="flex-1 glass-card p-4 relative overflow-hidden bg-slate-50/50">
        <div className="absolute top-6 left-6 z-20 flex gap-4">
           <div className="px-4 py-2 bg-white/90 backdrop-blur shadow-sm border border-slate-200 rounded-xl flex items-center gap-3">
              <div className="p-1.5 bg-scientific-primary/10 rounded-lg text-scientific-primary">
                 <Share2 size={18} />
              </div>
              <div>
                 <h2 className="text-sm font-bold text-slate-800">Topological Explorer</h2>
                 <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Interactive PPI Schema</p>
              </div>
           </div>
        </div>

        <div className="absolute bottom-6 left-6 z-20 flex flex-col gap-2">
           <button onClick={resetLayout} className="p-2.5 bg-white shadow-lg border border-slate-200 rounded-xl text-slate-600 hover:text-scientific-primary transition-all">
              <RotateCcw size={20} />
           </button>
           <button className="p-2.5 bg-white shadow-lg border border-slate-200 rounded-xl text-slate-600 hover:text-scientific-primary transition-all">
              <Maximize2 size={20} />
           </button>
        </div>

        {loading && (
          <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-white/50 backdrop-blur-sm">
             <div className="w-12 h-12 border-4 border-scientific-primary/20 border-t-scientific-primary rounded-full animate-spin mb-4" />
             <p className="text-sm font-bold text-scientific-primary uppercase tracking-widest">Generating Network Graph...</p>
          </div>
        )}

        <div ref={containerRef} className="w-full h-full rounded-2xl cursor-grab active:cursor-grabbing" />
      </div>

      {/* Control Panel / Metrics */}
      <div className="w-80 space-y-6 flex flex-col overflow-y-auto pr-2 custom-scrollbar">
        {/* Detail Panel */}
        <AnimatePresence mode="wait">
          {selectedNode ? (
            <motion.div 
              key="details"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className="glass-card p-6 border-l-4 border-scientific-accent"
            >
              <div className="flex items-center gap-2 mb-4 text-scientific-accent">
                 <Database size={16} />
                 <span className="text-[10px] font-bold uppercase tracking-widest">Protein Metadata</span>
              </div>
              <h3 className="text-xl font-bold text-slate-800 mb-1">{selectedNode.label}</h3>
              <p className="text-xs text-slate-500 mb-6 font-medium leading-relaxed">
                 Central biological node involved in complex metabolic signaling pathways.
              </p>
              
              <div className="grid grid-cols-2 gap-3 mb-6">
                 <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                    <p className="text-[10px] font-bold text-slate-400 uppercase mb-1">Degree</p>
                    <p className="text-sm font-bold text-slate-700">12 Neighbors</p>
                 </div>
                 <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                    <p className="text-[10px] font-bold text-slate-400 uppercase mb-1">Closeness</p>
                    <p className="text-sm font-bold text-slate-700">0.842</p>
                 </div>
              </div>

              <button className="w-full btn-primary py-2.5 text-sm flex items-center justify-center gap-2">
                 <Search size={16} /> View Predictions
              </button>
            </motion.div>
          ) : (
            <motion.div 
              key="placeholder"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="glass-card p-8 flex flex-col items-center justify-center text-center bg-slate-50/50"
            >
               <Info size={32} className="text-slate-200 mb-4" />
               <p className="text-sm font-bold text-slate-400 uppercase tracking-widest">Select a node to view properties</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Centrality Metrics */}
        <div className="glass-card p-6 flex-1 min-h-0 flex flex-col">
          <div className="flex items-center justify-between mb-6">
             <h3 className="text-sm font-bold text-slate-800">Network Hubs</h3>
             <Activity size={18} className="text-scientific-primary" />
          </div>
          
          <div className="space-y-4 overflow-y-auto pr-2">
             {metrics.map((m, i) => (
                <div key={i} className="group p-3 hover:bg-slate-50 rounded-xl border border-transparent hover:border-slate-100 transition-all cursor-pointer">
                   <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-bold text-slate-700">{m.protein}</span>
                      {i < 3 && <Star size={14} className="fill-warning text-warning" />}
                   </div>
                   <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                      <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: `${(m.degree / (metrics[0]?.degree || 1)) * 100}%` }}
                        transition={{ delay: i * 0.1, duration: 1 }}
                        className="h-full bg-scientific-gradient" 
                      />
                   </div>
                   <div className="flex justify-between mt-1 text-[10px] font-bold text-slate-400 uppercase tracking-tighter">
                      <span>Degree: {m.degree}</span>
                      <span>BC: {m.betweenness?.toFixed(3) || '0.000'}</span>
                   </div>
                </div>
             ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default NetworkExplorer;
