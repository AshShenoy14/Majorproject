import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Box, 
  Search, 
  Database, 
  Info, 
  Download, 
  Maximize2, 
  ChevronRight,
  Activity,
  MapPin,
  Tag
} from 'lucide-react';
import { ppiService } from '../services/api';

const StructureViewer = () => {
  const [proteinId, setProteinId] = useState('ENSP00000327694');
  const [viewerLoading, setViewerLoading] = useState(true);
  const [metadata, setMetadata] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const viewerContainerRef = useRef(null);
  const pluginRef = useRef(null);

  useEffect(() => {
    const scriptId = 'pdbe-molstar-script';
    const linkId = 'pdbe-molstar-link';

    const initViewer = () => {
      if (window.PDBeMolstarPlugin && viewerContainerRef.current && !pluginRef.current) {
        try {
          pluginRef.current = new window.PDBeMolstarPlugin();
          pluginRef.current.render(viewerContainerRef.current, {
            moleculeId: 'P01112',
            alphafoldView: true,
            expanded: false,
            hideCanvasControls: ['selection', 'animation', 'geometry'],
            theme: {
              background: { color: { r: 248, g: 250, b: 252 } }
            }
          });
          setViewerLoading(false);
        } catch (err) {
          console.error("Molstar render error:", err);
          setError("Failed to initialize 3D viewer.");
          setViewerLoading(false);
        }
      }
    };

    if (!document.getElementById(scriptId)) {
      const script = document.createElement('script');
      script.id = scriptId;
      script.src = 'https://www.ebi.ac.uk/pdbe/pdb-component-library/js/pdbe-molstar-plugin-3.1.2.js';
      script.async = true;
      
      const link = document.createElement('link');
      link.id = linkId;
      link.rel = 'stylesheet';
      link.href = 'https://www.ebi.ac.uk/pdbe/pdb-component-library/css/pdbe-molstar-3.1.2.css';
      
      document.head.appendChild(link);
      document.head.appendChild(script);

      script.onload = initViewer;
    } else {
      // Script already exists, wait a bit for window object if it's still loading
      const interval = setInterval(() => {
        if (window.PDBeMolstarPlugin) {
          initViewer();
          clearInterval(interval);
        }
      }, 100);
      return () => clearInterval(interval);
    }
  }, []);

  const handleSearch = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      // 1. Fetch metadata from local backend
      const response = await ppiService.getBioMetadata(proteinId);
      const data = response.data[0];
      setMetadata(data);

      // 2. Try to render structure
      // We need UniProt ID for AlphaFold. our backend might return it as protein1_uniprot_id if we used /predict
      // or we can try to use the ENSP ID directly if AlphaFold supports it, but usually it's UniProt.
      // For now let's assume 'data.protein_id' might be UniProt if mapped, or we use a fallback.
      let uniProtId = data?.uniprot_id || (proteinId.length === 6 || proteinId.length === 10 ? proteinId : null);
      
      if (!uniProtId) {
        setError("Could not map to a valid UniProt ID for AlphaFold visualization.");
        setLoading(false);
        return;
      }
      
      if (pluginRef.current) {
        pluginRef.current.visual.update({
          moleculeId: uniProtId,
          alphafoldView: true
        });
      }
    } catch (err) {
      setError("Failed to fetch structure or metadata.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Left Side: Controls & Meta */}
        <div className="lg:col-span-1 space-y-6">
          <div className="glass-card p-6">
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">Structure Query</h3>
            <form onSubmit={handleSearch} className="space-y-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
                <input 
                  type="text" 
                  value={proteinId}
                  onChange={(e) => setProteinId(e.target.value)}
                  placeholder="UniProt/ENSP ID..."
                  className="w-full pl-10 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:ring-2 focus:ring-scientific-primary outline-none"
                />
              </div>
              <button 
                type="submit" 
                disabled={loading}
                className="w-full btn-primary py-2 text-sm flex items-center justify-center gap-2"
              >
                {loading ? <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <Box size={16} />}
                Load Structure
              </button>
            </form>
          </div>

          <AnimatePresence>
            {metadata && (
              <motion.div 
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="glass-card p-6 space-y-6"
              >
                <div>
                  <h4 className="text-sm font-bold text-slate-800 flex items-center gap-2 mb-2">
                    <Tag size={16} className="text-scientific-primary" />
                    Annotation
                  </h4>
                  <p className="text-xs text-slate-600 leading-relaxed font-medium">
                    {metadata.function || "No functional annotation available for this sequence cluster."}
                  </p>
                </div>

                <div className="grid grid-cols-1 gap-4">
                  <div className="p-3 bg-teal-50 rounded-xl border border-teal-100">
                    <div className="flex items-center gap-2 text-scientific-primary mb-1">
                      <MapPin size={14} />
                      <span className="text-[10px] font-bold uppercase">Localization</span>
                    </div>
                    <p className="text-sm font-bold text-slate-700">{metadata.subcellular_location || "Unknown"}</p>
                  </div>
                  
                  <div className="p-3 bg-purple-50 rounded-xl border border-purple-100">
                    <div className="flex items-center gap-2 text-scientific-accent mb-1">
                      <Activity size={14} />
                      <span className="text-[10px] font-bold uppercase">Relevance</span>
                    </div>
                    <p className="text-sm font-bold text-slate-700">{metadata.biological_process || "Metabolic Process"}</p>
                  </div>
                </div>

                <div className="pt-4 border-t border-slate-100 flex justify-between items-center text-[10px] font-bold text-slate-400">
                   <span>SOURCE: ALPHAFOLD DB</span>
                   <button className="text-scientific-primary hover:underline flex items-center gap-1">
                      FULL UNIPROT <ChevronRight size={10} />
                   </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Right Side: Viewer */}
        <div className="lg:col-span-3 space-y-6">
          <div className="glass-card p-1 h-[600px] relative overflow-hidden group">
            <div className="absolute top-6 left-6 z-20 flex gap-2">
               <div className="px-3 py-1.5 bg-white shadow-sm border border-slate-200 rounded-lg text-xs font-bold text-slate-700 flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                  3D INTERACTIVE RENDER
               </div>
            </div>
            
            <div className="absolute top-6 right-6 z-20 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
               <button className="p-2 bg-white shadow-sm border border-slate-200 rounded-lg text-slate-600 hover:text-scientific-primary">
                  <Maximize2 size={18} />
               </button>
               <button className="p-2 bg-white shadow-sm border border-slate-200 rounded-lg text-slate-600 hover:text-scientific-primary">
                  <Download size={18} />
               </button>
            </div>

            <div 
              ref={viewerContainerRef} 
              className="w-full h-full rounded-2xl bg-slate-50"
              style={{ position: 'relative' }}
            />
            
            {!metadata && !loading && (
              <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                 <Database size={48} className="text-slate-100 mb-4" />
                 <p className="text-slate-300 font-bold uppercase tracking-widest text-sm">Waiting for selection</p>
              </div>
            )}
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
             {['Rotate', 'Zoom', 'Pan', 'Highlight'].map(action => (
                <div key={action} className="glass-card p-4 flex items-center justify-between group cursor-default">
                   <span className="text-xs font-bold text-slate-500 uppercase tracking-tighter">{action}</span>
                   <div className="w-8 h-8 rounded-full bg-slate-50 border border-slate-100 flex items-center justify-center text-slate-400 group-hover:text-scientific-primary transition-colors">
                      <Info size={14} />
                   </div>
                </div>
             ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default StructureViewer;
