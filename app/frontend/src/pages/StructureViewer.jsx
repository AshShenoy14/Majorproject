import React, { useState, useEffect, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
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
  const [searchParams] = useSearchParams();
  const queryProtein = searchParams.get('protein') || searchParams.get('p') || '';
  const [proteinId, setProteinId] = useState(queryProtein || 'P04637');
  const [viewerLoading, setViewerLoading] = useState(true);
  const [metadata, setMetadata] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const viewerContainerRef = useRef(null);
  const pluginRef = useRef(null);
  const autoLoadedRef = useRef(false);

  const isLikelyUniProt = (id) => {
    if (!id) return false;
    const clean = id.trim().toUpperCase();
    return /^[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$/i.test(clean);
  };

  const resolveTargetToUniProt = async (targetId) => {
    if (!targetId) return { uniProtId: 'P04637', meta: null };
    const raw = targetId.trim();
    const cleanId = raw.replace(/^9606\./, '');

    // 1. Try querying backend metadata service to map ENSP -> UniProt & get bio context
    try {
      const response = await ppiService.getBioMetadata(cleanId);
      const data = response.data?.[0];
      if (data) {
        setMetadata(data);
        if (data.uniprot_id) {
          return { uniProtId: data.uniprot_id, meta: data };
        }
      }
    } catch (err) {
      console.warn("Bio metadata lookup failed for ID resolution:", err);
    }

    // 2. If already formatted like a UniProt ID, use it directly
    if (isLikelyUniProt(cleanId)) {
      return { uniProtId: cleanId, meta: null };
    }

    return { uniProtId: cleanId, meta: null };
  };

  const renderMolstar = async (targetUniProtId) => {
    if (!pluginRef.current || !viewerContainerRef.current) return;

    // Destroy existing plugin instance if present
    try {
      if (pluginRef.current.plugin) {
        pluginRef.current.plugin.dispose();
      }
    } catch (e) { /* ignore */ }

    // Fetch the current CIF URL from AlphaFold API
    let cifUrl = `https://alphafold.ebi.ac.uk/files/AF-${targetUniProtId}-F1-model_v6.cif`;
    try {
      const res = await fetch(`https://alphafold.ebi.ac.uk/api/prediction/${targetUniProtId}`);
      if (!res.ok) {
        throw new Error(`AlphaFold API returned HTTP ${res.status}`);
      }
      const data = await res.json();
      if (data && data.length > 0 && data[0].cifUrl) {
        cifUrl = data[0].cifUrl;
      }
    } catch (fetchErr) {
      console.warn(`AlphaFold API lookup warning for ${targetUniProtId}, attempting direct CIF:`, fetchErr);
    }

    pluginRef.current = new window.PDBeMolstarPlugin();
    pluginRef.current.render(viewerContainerRef.current, {
      customData: {
        url: cifUrl,
        format: 'cif'
      },
      alphafoldView: true,
      expanded: false,
      hideCanvasControls: ['selection', 'animation', 'geometry'],
      bgColor: { r: 248, g: 250, b: 252 }
    });
  };

  const executeSearchForId = async (targetId) => {
    if (!targetId) return;
    setLoading(true);
    setError(null);
    try {
      const { uniProtId, meta } = await resolveTargetToUniProt(targetId);
      if (meta) setMetadata(meta);
      await renderMolstar(uniProtId);
    } catch (err) {
      console.error("Structure search error:", err);
      setError(`Could not load AlphaFold 3D structure for ${targetId}. Verify that this protein has an AlphaFold prediction.`);
    } finally {
      setLoading(false);
    }
  };

  const initViewer = async () => {
    if (window.PDBeMolstarPlugin && viewerContainerRef.current && !pluginRef.current) {
      try {
        setViewerLoading(true);
        const initialTarget = queryProtein || proteinId || 'P04637';
        pluginRef.current = new window.PDBeMolstarPlugin();
        
        // Resolve ID first so ENSP and 9606.ENSP load correctly on first mount
        const { uniProtId, meta } = await resolveTargetToUniProt(initialTarget);
        if (meta) setMetadata(meta);

        await renderMolstar(uniProtId);
        autoLoadedRef.current = true;
      } catch (err) {
        console.error("Molstar init error:", err);
        setError("Failed to initialize 3D viewer.");
      } finally {
        setViewerLoading(false);
      }
    }
  };

  useEffect(() => {
    const scriptId = 'pdbe-molstar-script';
    const linkId = 'pdbe-molstar-link';

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
    if (e) e.preventDefault();
    executeSearchForId(proteinId);
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

            {error && (
              <div className="p-4 bg-amber-50 border border-amber-200 rounded-xl text-xs text-amber-800 flex items-start gap-2.5">
                <Info size={16} className="text-amber-600 shrink-0 mt-0.5" />
                <div>
                  <p className="font-bold mb-0.5">Structure Notice</p>
                  <p>{error}</p>
                </div>
              </div>
            )}

            <AnimatePresence>
              {metadata && (
                <motion.div 
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="glass-card p-6 space-y-6"
                >
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="text-sm font-bold text-slate-800 flex items-center gap-2">
                        <Tag size={16} className="text-scientific-primary" />
                        Identified Target
                      </h4>
                      {metadata.uniprot_id && (
                        <span className="text-[10px] font-mono px-2 py-0.5 bg-blue-50 text-blue-700 border border-blue-200 rounded font-semibold">
                          UniProt: {metadata.uniprot_id}
                        </span>
                      )}
                    </div>
                    {metadata.protein_id && (
                      <p className="text-[11px] font-mono text-slate-500 mb-2">
                        Query ID: {metadata.protein_id}
                      </p>
                    )}
                    <p className="text-xs text-slate-600 leading-relaxed font-medium">
                      {metadata.families || metadata.domains || "Protein entry resolved from Ensembl/UniProt cross-reference."}
                    </p>
                  </div>

                  <div className="grid grid-cols-1 gap-4">
                    <div className="p-3 bg-teal-50 rounded-xl border border-teal-100">
                      <div className="flex items-center gap-2 text-scientific-primary mb-1">
                        <MapPin size={14} />
                        <span className="text-[10px] font-bold uppercase">Localization</span>
                      </div>
                      <p className="text-xs font-semibold text-slate-700 leading-snug">
                        {metadata.localization || metadata.subcellular_location || "Unknown"}
                      </p>
                    </div>
                    
                    <div className="p-3 bg-purple-50 rounded-xl border border-purple-100">
                      <div className="flex items-center gap-2 text-scientific-accent mb-1">
                        <Activity size={14} />
                        <span className="text-[10px] font-bold uppercase">Pathways & Processes</span>
                      </div>
                      <p className="text-xs font-semibold text-slate-700 leading-snug">
                        {metadata.pathways || metadata.biological_process || "Cellular Signaling / Unclassified"}
                      </p>
                    </div>
                  </div>

                  <div className="pt-4 border-t border-slate-100 flex justify-between items-center text-[10px] font-bold text-slate-400">
                     <span>SOURCE: ALPHAFOLD DB</span>
                     {metadata.uniprot_id && (
                       <a 
                         href={`https://www.uniprot.org/uniprotkb/${metadata.uniprot_id}/entry`}
                         target="_blank"
                         rel="noreferrer"
                         className="text-scientific-primary hover:underline flex items-center gap-1"
                       >
                          FULL UNIPROT <ChevronRight size={10} />
                       </a>
                     )}
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
