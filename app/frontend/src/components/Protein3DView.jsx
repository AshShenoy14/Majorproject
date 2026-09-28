import { useEffect, useRef, useState } from 'react';

// Industry standard PDBe-Molstar 3D protein structure viewer
// Maps Ensembl / UniProt IDs to representative PDB 3D structures and highlights interaction interfaces.

const PDB_MAP = {
  // Case study pairs & common ENSP/UniProt IDs
  'ENSP00000327694': '1tnr',
  'ENSP00000373627': '1a2y',
  'ENSP00000269305': '1tup', // TP53 (Tumor Suppressor P53)
  'ENSP00000258149': '1ycr', // MDM2 (E3 ubiquitin-protein ligase)
  'ENSP00000300161': '1gw5', // AP2A2 (AP-2 complex subunit alpha-2)
  'ENSP00000267029': '1b89', // CLTC (Clathrin heavy chain)
  'ENSP00000293879': '1f16', // BAX (Apoptosis regulator BAX)
  'ENSP00000307677': '1lxl', // BCL2L1 (Bcl-2-like protein 1)
  'ENSP00000385802': '2kxa', // Cold-start Protein A
  'ENSP00000361000': '3h84', // Cold-start Protein B
  'P04637': '1tup',
  'Q00987': '1ycr',
  'P05412': '1jun',
  'P01106': '1fos',
};

const resolvePdbId = (id, fallback = null) => {
  if (!id) return fallback;
  const clean = id.trim().toUpperCase();
  if (PDB_MAP[clean]) return PDB_MAP[clean];
  if (PDB_MAP[id.trim()]) return PDB_MAP[id.trim()];
  if (clean.length === 4 && /^[0-9][A-Za-z0-9]{3}$/.test(clean)) return clean.toLowerCase();
  return fallback;
};

const Protein3DView = ({ pdbId, label, selectedResidue, interactionRegion, fallbackPdbId = null }) => {
  const viewerContainerRef = useRef(null);
  const pluginInstanceRef = useRef(null);
  const [loadError, setLoadError] = useState(false);

  const activePdbId = resolvePdbId(pdbId, fallbackPdbId);

  useEffect(() => {
    setLoadError(false);
    if (!activePdbId) {
      setLoadError(true);
      return;
    }

    let isMounted = true;

    const loadViewer = async () => {
      try {
        const { PDBeMolstarPlugin } = await import('pdbe-molstar');
        if (!isMounted) return;

        const pluginInstance = new PDBeMolstarPlugin();
        pluginInstanceRef.current = pluginInstance;
        
        if (viewerContainerRef.current) {
          viewerContainerRef.current.innerHTML = '';

          await pluginInstance.render(viewerContainerRef.current, {
            moleculeId: activePdbId,
            expanded: false,
            loadContext: { auth_asym_id: 'A' },
            bgColor: { r: 2, g: 6, b: 23 }, // Matches slate-950 theme
            hideCanvasControls: ['selection', 'animation', 'controlToggle', 'controlInfo'],
          });
        }
      } catch (error) {
        console.warn("3D Structure loading note:", error);
        if (isMounted) setLoadError(true);
      }
    };

    const container = viewerContainerRef.current;
    loadViewer();

    return () => {
      isMounted = false;
      if (container) {
        container.innerHTML = '';
      }
      pluginInstanceRef.current = null;
    };
  }, [activePdbId]);

  // Synchronize interaction interface region and residue selection on 3D viewer
  useEffect(() => {
    if (!activePdbId || loadError) return;

    const timer = setTimeout(() => {
      if (pluginInstanceRef.current && pluginInstanceRef.current.visual?.select) {
        try {
          const selections = [];

          // 1. Highlight binding interface region if present
          if (interactionRegion && Array.isArray(interactionRegion) && interactionRegion.length === 2) {
            const [start, end] = interactionRegion;
            for (let r = start; r <= Math.min(end, start + 25); r++) {
              selections.push({
                residue_number: r,
                struct_asym_id: 'A',
                color: { r: 16, g: 185, b: 129 }, // Emerald green highlight for interaction region
                focus: false
              });
            }
          }

          // 2. Focused specific residue if clicked by user
          if (selectedResidue?.residue_number) {
            selections.push({
              residue_number: selectedResidue.residue_number,
              struct_asym_id: 'A',
              color: { r: 239, g: 68, b: 68 }, // Rose red highlight for specific focused residue
              focus: true
            });
          }

          if (selections.length > 0) {
            pluginInstanceRef.current.visual.select({ data: selections });
          }
        } catch (err) {
          console.warn("3D selection sync notice:", err);
        }
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [selectedResidue, interactionRegion, activePdbId, loadError]);

  const handleResetCamera = () => {
    try {
      if (pluginInstanceRef.current && pluginInstanceRef.current.visual?.reset) {
        pluginInstanceRef.current.visual.reset({ camera: true, theme: true });
      }
    } catch (e) {
      console.warn("Camera reset notice:", e);
    }
  };

  return (
    <div className="flex flex-col items-center group w-full max-w-full min-w-0 box-border">
      <div className="flex items-center justify-between gap-2 w-full mb-2.5 min-w-0 max-w-full flex-wrap">
        <span className="flex items-center gap-2 font-black text-xs text-slate-800 uppercase tracking-wider truncate min-w-0">
          <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${activePdbId && !loadError ? 'bg-cyan-500 animate-pulse' : 'bg-slate-400'}`} />
          <span className="truncate">
            {label || 'Structure'} {activePdbId ? `(PDB: ${activePdbId.toUpperCase()})` : ''}
          </span>
        </span>

        <div className="flex items-center gap-2 shrink-0">
          {activePdbId && !loadError && (
            <button
              onClick={handleResetCamera}
              className="text-[10px] font-semibold text-slate-500 hover:text-slate-800 bg-slate-100 hover:bg-slate-200 border border-slate-300 rounded px-2 py-0.5 transition"
              title="Reset 3D camera rotation and zoom"
            >
              Reset View
            </button>
          )}

          {selectedResidue?.residue_number ? (
            <span className="text-rose-600 font-mono text-[10px] bg-rose-50 px-2.5 py-1 rounded-lg border border-rose-200 font-bold shrink-0">
              focused: res #{selectedResidue.residue_number}
            </span>
          ) : interactionRegion ? (
            <span className="text-emerald-700 font-mono text-[10px] bg-emerald-50 px-2.5 py-1 rounded-lg border border-emerald-200 font-bold shrink-0">
              interface: #{interactionRegion[0]}-#{interactionRegion[1]}
            </span>
          ) : null}
        </div>
      </div>

      <div 
        className="w-full max-w-full h-[360px] md:h-[400px] rounded-2xl overflow-hidden border border-slate-800 bg-slate-950 shadow-md relative flex items-center justify-center"
        style={{ position: 'relative', width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}
      >
        {(!activePdbId || loadError) ? (
          <div className="flex flex-col items-center justify-center p-6 text-center max-w-sm text-slate-400">
            <svg className="w-12 h-12 text-slate-600 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
            </svg>
            <h4 className="text-sm font-semibold text-slate-300 mb-1">Structure Unavailable</h4>
            <p className="text-xs text-slate-500 leading-relaxed">
              No experimental PDB coordinates resolved for <span className="font-mono text-slate-400">{pdbId || 'this protein'}</span>.
              Prediction relies on ESM-2 sequence and topological graph embeddings.
            </p>
          </div>
        ) : (
          <>
            <div ref={viewerContainerRef} className="w-full h-full" />

            {/* Interface Region Indicator Badge */}
            {interactionRegion && Array.isArray(interactionRegion) && interactionRegion.length === 2 && (
              <div className="absolute top-3 left-3 z-10 bg-emerald-950/85 text-emerald-300 text-[10px] font-bold px-3 py-1.5 rounded-lg shadow-lg border border-emerald-500/40 backdrop-blur-md flex items-center gap-2 pointer-events-none max-w-[calc(100%-1.5rem)] truncate">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping shrink-0" />
                <span className="truncate">Binding Interface (#{interactionRegion[0]}-#{interactionRegion[1]})</span>
              </div>
            )}

            {/* Selected Residue Badge */}
            {selectedResidue?.residue_number && (
              <div className="absolute top-3 right-3 z-10 bg-rose-950/85 text-rose-300 text-[10px] font-bold px-3 py-1.5 rounded-lg shadow-lg border border-rose-500/40 backdrop-blur-md flex items-center gap-2 animate-pulse pointer-events-none max-w-[calc(100%-1.5rem)] truncate">
                <span className="w-2 h-2 rounded-full bg-rose-400 shrink-0" />
                <span className="truncate">Residue #{selectedResidue.residue_number} ({selectedResidue.residue_name || 'AA'})</span>
              </div>
            )}

            {/* Interactive hint */}
            <div className="absolute bottom-2 right-3 z-10 text-[9px] text-slate-500 bg-slate-900/80 px-2 py-0.5 rounded pointer-events-none border border-slate-800">
              Drag to rotate • Scroll to zoom
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default Protein3DView;
