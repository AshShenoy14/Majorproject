import React, { useEffect, useRef } from 'react';

// This component uses the PDBe-Molstar plugin which is already in your package.json
// It's the industry standard for showing 3D protein structures in web apps.

const Protein3DView = ({ pdbId, label, selectedResidue }) => {
  const viewerContainerRef = useRef(null);
  const pluginInstanceRef = useRef(null);

  useEffect(() => {
    // Dynamic import to ensure this only runs client-side after mount
    const loadViewer = async () => {
      try {
        const { PDBeMolstarPlugin } = await import('pdbe-molstar');
        const pluginInstance = new PDBeMolstarPlugin();
        pluginInstanceRef.current = pluginInstance;
        
        const id = (pdbId && pdbId.length === 4) ? pdbId.toLowerCase() : '1tnr';
        
        if (viewerContainerRef.current) {
          pluginInstance.render(viewerContainerRef.current, {
            moleculeId: id,
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
  }, [pdbId]);

  // Synchronize residue focus/selection on 3D viewer when selectedResidue prop updates
  useEffect(() => {
    if (pluginInstanceRef.current && selectedResidue?.residue_number) {
      try {
        if (pluginInstanceRef.current.visual?.select) {
          pluginInstanceRef.current.visual.select({
            data: [{
              residue_number: selectedResidue.residue_number,
              struct_asym_id: 'A',
              color: { r: 239, g: 68, b: 68 },
              focus: true
            }]
          });
        }
      } catch (err) {
        console.warn("3D selection sync notice:", err);
      }
    }
  }, [selectedResidue]);

  return (
    <div className="flex flex-col items-center group w-full">
      <div className="text-[10px] font-black text-cyan-400/60 mb-2 uppercase tracking-[0.2em] group-hover:text-cyan-400 transition-colors flex items-center justify-between w-full">
        <span>{label || 'Structure'}</span>
        {selectedResidue?.residue_number && (
          <span className="text-rose-400 font-mono text-[9px] lowercase bg-rose-500/10 px-1.5 py-0.5 rounded border border-rose-500/20">
            focused: res #{selectedResidue.residue_number}
          </span>
        )}
      </div>
      <div 
        ref={viewerContainerRef} 
        className="w-full h-[280px] rounded-2xl overflow-hidden border border-white/10 bg-slate-950 shadow-2xl relative"
        style={{ position: 'relative' }}
      >
        {selectedResidue?.residue_number && (
          <div className="absolute top-2 right-2 z-10 bg-rose-500/90 text-white text-[10px] font-black px-2.5 py-1 rounded-md shadow-lg border border-rose-400 backdrop-blur-md flex items-center gap-1.5 animate-pulse pointer-events-none">
            <span className="w-1.5 h-1.5 rounded-full bg-white" />
            Focused Residue #{selectedResidue.residue_number} ({selectedResidue.residue_name || 'AA'})
          </div>
        )}
        {/* The PDBe-Molstar plugin will inject the canvas here */}
      </div>
    </div>
  );
};

export default Protein3DView;
