import React, { useEffect, useRef } from 'react';

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

const resolvePdbId = (id, fallback = '1tnr') => {
  if (!id) return fallback;
  const clean = id.trim().toUpperCase();
  if (PDB_MAP[clean]) return PDB_MAP[clean];
  if (PDB_MAP[id.trim()]) return PDB_MAP[id.trim()];
  if (clean.length === 4) return clean.toLowerCase();
  return fallback;
};

const Protein3DView = ({ pdbId, label, selectedResidue, interactionRegion, fallbackPdbId = '1tnr' }) => {
  const viewerContainerRef = useRef(null);
  const pluginInstanceRef = useRef(null);

  const activePdbId = resolvePdbId(pdbId, fallbackPdbId);

  useEffect(() => {
    const loadViewer = async () => {
      try {
        const { PDBeMolstarPlugin } = await import('pdbe-molstar');
        const pluginInstance = new PDBeMolstarPlugin();
        pluginInstanceRef.current = pluginInstance;
        
        if (viewerContainerRef.current) {
          // Clear previous canvas if any
          viewerContainerRef.current.innerHTML = '';

          pluginInstance.render(viewerContainerRef.current, {
            moleculeId: activePdbId,
            expanded: false,
            loadContext: { auth_asym_id: 'A' },
            bgColor: { r: 2, g: 6, b: 23 }, // Matches slate-950 theme
            hideCanvasControls: ['selection', 'animation', 'controlToggle', 'controlInfo'],
          });
        }
      } catch (error) {
        console.error("3D Viewer Error:", error);
      }
    };

    loadViewer();

    return () => {
      if (viewerContainerRef.current) {
        viewerContainerRef.current.innerHTML = '';
      }
      pluginInstanceRef.current = null;
    };
  }, [activePdbId]);

  // Synchronize interaction interface region and residue selection on 3D viewer
  useEffect(() => {
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
  }, [selectedResidue, interactionRegion, activePdbId]);

  return (
    <div className="flex flex-col items-center group w-full max-w-full min-w-0 box-border">
      <div className="flex items-center justify-between gap-2 w-full mb-2.5 min-w-0 max-w-full flex-wrap">
        <span className="flex items-center gap-2 font-black text-xs text-slate-800 uppercase tracking-wider truncate min-w-0">
          <span className="w-2.5 h-2.5 rounded-full bg-cyan-500 animate-pulse shrink-0" />
          <span className="truncate">{label || 'Structure'} (PDB: {activePdbId.toUpperCase()})</span>
        </span>
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

      <div 
        ref={viewerContainerRef} 
        className="w-full max-w-full h-[360px] md:h-[400px] rounded-2xl overflow-hidden border border-slate-800 bg-slate-950 shadow-md relative"
        style={{ position: 'relative', width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}
      >
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
      </div>
    </div>
  );
};

export default Protein3DView;
