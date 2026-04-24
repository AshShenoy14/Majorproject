import React, { useEffect, useRef } from 'react';

// This component uses the PDBe-Molstar plugin which is already in your package.json
// It's the industry standard for showing 3D protein structures in web apps.

const Protein3DView = ({ pdbId, label }) => {
  const viewerContainerRef = useRef(null);

  useEffect(() => {
    // We use a dynamic import to ensure this only runs on the client
    // and after the component has mounted.
    const loadViewer = async () => {
      try {
        const { PDBeMolstarPlugin } = await import('pdbe-molstar');
        const pluginInstance = new PDBeMolstarPlugin();
        
        const id = (pdbId && pdbId.length === 4) ? pdbId.toLowerCase() : '1tnr';
        
        if (viewerContainerRef.current) {
          pluginInstance.render(viewerContainerRef.current, {
            moleculeId: id,
            expanded: false,
            loadContext: { auth_asym_id: 'A' },
            bgColor: { r: 2, g: 6, b: 23 }, // Matches your slate-950 theme
            hideCanvasControls: ['selection', 'animation', 'controlToggle', 'controlInfo'],
          });
        }
      } catch (error) {
        console.error("3D Viewer Error:", error);
      }
    };

    loadViewer();
  }, [pdbId]);

  return (
    <div className="flex flex-col items-center group w-full">
      <div className="text-[10px] font-black text-cyan-400/60 mb-2 uppercase tracking-[0.2em] group-hover:text-cyan-400 transition-colors">
        {label || 'Structure'}
      </div>
      <div 
        ref={viewerContainerRef} 
        className="w-full h-[280px] rounded-2xl overflow-hidden border border-white/10 bg-slate-950 shadow-2xl relative"
        style={{ position: 'relative' }}
      >
        {/* The PDBe-Molstar plugin will inject the canvas here */}
      </div>
    </div>
  );
};

export default Protein3DView;
