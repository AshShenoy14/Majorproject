import React, { useEffect, useRef } from 'react';
import { Box, Typography, Skeleton, useTheme } from '@mui/material';

const ProteinViewer = ({ proteinId }) => {
    const viewerRef = useRef(null);
    const theme = useTheme();
    const isDark = theme.palette.mode === 'dark';

    useEffect(() => {
        if (!proteinId) return;

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

        const renderViewer = () => {
            if (viewerRef.current && window.PDBeMolstarPlugin) {
                viewerRef.current.innerHTML = ''; // Clear previous
                const viewerInstance = new window.PDBeMolstarPlugin();

                const bgColor = isDark ? { r: 10, g: 25, b: 47 } : { r: 244, g: 246, b: 248 };

                const options = {
                    customData: {
                        url: `https://alphafold.ebi.ac.uk/files/AF-${proteinId.toUpperCase()}-F1-model_v4.pdb`,
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

        if (!script) {
            script = document.createElement('script');
            script.id = scriptId;
            script.src = 'https://cdn.jsdelivr.net/npm/pdbe-molstar@3.2.0/build/pdbe-molstar-plugin.js';
            script.async = true;
            script.onload = renderViewer;
            document.body.appendChild(script);
        } else {
            // Already loaded
            renderViewer();
        }

        return () => {
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
            {!proteinId ? (
                <Box display="flex" alignItems="center" justifyContent="center" height="100%">
                    <Typography color="text.secondary">Select a protein to view 3D structure</Typography>
                </Box>
            ) : (
                <div ref={viewerRef} style={{ width: '100%', height: '100%', position: 'absolute', top: 0, left: 0 }} />
            )}
        </Box>
    );
};

export default ProteinViewer;
