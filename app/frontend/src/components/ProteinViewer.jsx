import React, { useEffect, useRef } from 'react';
import { Box, Typography, Skeleton, useTheme } from '@mui/material';

const ProteinViewer = ({ proteinId }) => {
    const viewerRef = useRef(null);
    const isMounted = useRef(true);
    const theme = useTheme();
    const isDark = theme.palette.mode === 'dark';

    useEffect(() => {
        isMounted.current = true;
        if (!proteinId || proteinId === "Protein A" || proteinId === "Protein B") return;

        // Dynamically load the correct CSS based on theme
        const linkId = 'pdbe-molstar-css';
        let link = document.getElementById(linkId);
        if (!link) {
            link = document.createElement('link');
            link.id = linkId;
            link.rel = 'stylesheet';
            document.head.appendChild(link);
        }
        link.href = isDark
            ? 'https://cdn.jsdelivr.net/npm/pdbe-molstar@3.2.0/build/pdbe-molstar-light.css' // We might use custom background anyway, but load standard css
            : 'https://cdn.jsdelivr.net/npm/pdbe-molstar@3.2.0/build/pdbe-molstar-light.css';

        // Dynamically load JS
        const scriptId = 'pdbe-molstar-js';
        let script = document.getElementById(scriptId);

        const renderViewer = async () => {
            if (viewerRef.current && window.PDBeMolstarPlugin && isMounted.current) {
                viewerRef.current.innerHTML = ''; // Clear previous

                let realPdbUrl = `https://alphafold.ebi.ac.uk/files/AF-${proteinId.toUpperCase()}-F1-model_v4.pdb`;
                try {
                    // Try to fetch the accurate URL from AlphaFold's API directly
                    const res = await fetch(`https://alphafold.ebi.ac.uk/api/prediction/${proteinId.toUpperCase()}`);
                    const data = await res.json();
                    if (data && data.length > 0 && data[0].pdbUrl) {
                        realPdbUrl = data[0].pdbUrl;
                    }
                } catch (e) { } // Fallback to v4 string

                const viewerInstance = new window.PDBeMolstarPlugin();

                const bgColor = isDark ? { r: 10, g: 25, b: 47 } : { r: 244, g: 246, b: 248 };

                const options = {
                    customData: {
                        url: realPdbUrl,
                        format: 'pdb'
                    },
                    alphafoldView: true,
                    bgColor: bgColor,
                    hideControls: false,
                    hideLog: true,
                    hideSequencePanels: true,
                    visualStyle: 'cartoon',
                    lighting: 'plastic'
                };

                try {
                    viewerInstance.render(viewerRef.current, options);
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
                // Poll every 100ms until the script is fully parsed
                setTimeout(checkPluginAndRender, 100);
            }
        };

        if (!script) {
            script = document.createElement('script');
            script.id = scriptId;
            script.src = 'https://cdn.jsdelivr.net/npm/pdbe-molstar@3.2.0/build/pdbe-molstar-plugin.js';
            script.async = true;
            document.body.appendChild(script);
            checkPluginAndRender(); // Start polling
        } else {
            // Already added to DOM, start polling in case it's still downloading
            checkPluginAndRender();
        }

        return () => {
            isMounted.current = false;
            if (viewerRef.current) {
                viewerRef.current.innerHTML = '';
            }
        };
    }, [proteinId, isDark]);

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
