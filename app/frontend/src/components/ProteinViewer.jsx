import React, { useEffect, useRef } from 'react';
import { Box, Typography, Skeleton, useTheme } from '@mui/material';

const ProteinViewer = ({ proteinId, hotspotData, residueImpact }) => {
    const viewerRef = useRef(null);
    const viewerInstanceRef = useRef(null);
    const isMounted = useRef(true);
    const theme = useTheme();
    const isDark = theme.palette.mode === 'dark';

    useEffect(() => {
        isMounted.current = true;
        if (!proteinId || proteinId === "Protein A" || proteinId === "Protein B") return;

        // Dynamically load CSS and JS if not present (logic already exists in previous version)
        const linkId = 'pdbe-molstar-css';
        let link = document.getElementById(linkId);
        if (!link) {
            link = document.createElement('link');
            link.id = linkId;
            link.rel = 'stylesheet';
            link.href = 'https://cdn.jsdelivr.net/npm/pdbe-molstar@3.2.0/build/pdbe-molstar-light.css';
            document.head.appendChild(link);
        }

        const scriptId = 'pdbe-molstar-js';
        let script = document.getElementById(scriptId);

        const applyColors = () => {
            if (!viewerInstanceRef.current || !residueImpact) return;
            
            // Map residueImpact (0-1) to colors (Blue to Red)
            const colorData = residueImpact.map((impact, i) => {
                if (impact < 0.05) return null; // Ignore low impact
                
                // Red for high impact, Yellow for medium
                const color = impact > 0.2 ? { r: 255, g: 0, b: 0 } : { r: 255, g: 165, b: 0 };
                return {
                    residue_number: i + 1,
                    color: color
                };
            }).filter(x => x !== null);

            if (colorData.length > 0) {
                viewerInstanceRef.current.visual.select({
                    data: colorData,
                    nonSelectedColor: { r: 200, g: 200, b: 200 }
                });
            }
        };

        const renderViewer = async () => {
            if (viewerRef.current && window.PDBeMolstarPlugin && isMounted.current) {
                viewerRef.current.innerHTML = ''; 

                let realPdbUrl = `https://alphafold.ebi.ac.uk/files/AF-${proteinId.toUpperCase()}-F1-model_v6.pdb`;
                try {
                    const res = await fetch(`https://alphafold.ebi.ac.uk/api/prediction/${proteinId.toUpperCase()}`);
                    const data = await res.json();
                    if (data && data.length > 0 && data[0].pdbUrl) {
                        realPdbUrl = data[0].pdbUrl;
                    }
                } catch (e) { } 

                const viewerInstance = new window.PDBeMolstarPlugin();
                viewerInstanceRef.current = viewerInstance;

                const bgColor = isDark ? { r: 15, g: 23, b: 42 } : { r: 248, g: 250, b: 252 };

                const options = {
                    customData: {
                        url: realPdbUrl,
                        format: 'pdb'
                    },
                    alphafoldView: true,
                    bgColor: bgColor,
                    hideControls: true,
                    hideLog: true,
                    hideSequencePanels: true,
                    visualStyle: 'cartoon',
                    lighting: 'glossy'
                };

                try {
                    viewerInstance.render(viewerRef.current, options);
                    
                    // Apply hotspot colors after a short delay to ensure model is loaded
                    setTimeout(applyColors, 2000);
                } catch (e) {
                    console.error("Molstar rendering error:", e);
                }
            }
        };

        const checkPluginAndRender = () => {
            if (!isMounted.current) return;
            if (window.PDBeMolstarPlugin) {
                renderViewer();
            } else {
                setTimeout(checkPluginAndRender, 100);
            }
        };

        if (!script) {
            script = document.createElement('script');
            script.id = scriptId;
            script.src = 'https://cdn.jsdelivr.net/npm/pdbe-molstar@3.2.0/build/pdbe-molstar-plugin.js';
            script.async = true;
            document.body.appendChild(script);
            checkPluginAndRender();
        } else {
            checkPluginAndRender();
        }

        return () => {
            isMounted.current = false;
        };
    }, [proteinId, isDark, residueImpact]);

    return (
        <Box sx={{
            width: '100%',
            height: 400,
            position: 'relative',
            borderRadius: 3,
            overflow: 'hidden',
            border: `1px solid ${isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0,0,0,0.1)'}`,
            bgcolor: isDark ? '#0a192f' : '#f4f6f8'
        }}>
            {!proteinId || proteinId === "Protein A" || proteinId === "Protein B" ? (
                <Box display="flex" alignItems="center" justifyContent="center" height="100%" textAlign="center" p={3}>
                    <Typography color="text.secondary">
                        <strong>No 3D Structure Available</strong><br /><br />
                        You entered a custom sequence without a known UniProt/ENSP ID. While our model can predict interactions purely from amino acid sequences, rendering a visual 3D structure requires a pre-computed PDB file associated with a known ID.
                    </Typography>
                </Box>
            ) : (
                <div ref={viewerRef} style={{ width: '100%', height: '100%', position: 'absolute', top: 0, left: 0 }} />
            )}
        </Box>
    );
};

export default ProteinViewer;
