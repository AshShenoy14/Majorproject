import React, { useRef, useEffect, useState } from 'react';
import { Box, Typography, Paper, CircularProgress, Chip, Stack } from '@mui/material';
import ForceGraph2D from 'react-force-graph-2d';
import { ppiService } from '../services/api';

const ResidueInteractionGraph = ({ proteinId, sequence, isDark }) => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const graphRef = useRef();

    useEffect(() => {
        const fetchData = async () => {
            if (!proteinId) return;
            setLoading(true);
            try {
                const resp = await ppiService.getResidueGraph(proteinId, sequence);
                setData(resp.data);
            } catch (err) {
                console.error("Failed to fetch RIG:", err);
                setError("Could not generate sequence graph.");
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [proteinId, sequence]);

    if (loading) return <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}><CircularProgress /></Box>;
    if (error) return <Typography color="error">{error}</Typography>;
    if (!data) return null;

    return (
        <Paper 
            sx={{ 
                p: 2, 
                mb: 4, 
                borderRadius: 4, 
                overflow: 'hidden',
                bgcolor: isDark ? 'rgba(0,0,0,0.2)' : 'rgba(255,255,255,0.4)',
                border: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`
            }}
        >
            <Stack direction="row" justifyContent="space-between" alignItems="center" mb={2}>
                <Box>
                    <Typography variant="h6" sx={{ fontWeight: 800 }}>Residue Interaction Network</Typography>
                    <Typography variant="caption" color="text.secondary">
                        Topological graph of {proteinId}. Nodes=Residues, Edges=Predicted Contacts.
                    </Typography>
                </Box>
                <Stack direction="row" spacing={1}>
                    <Chip label={`${data.nodes.length} Residues`} size="small" />
                    <Chip label={`${data.metadata.num_contacts} Predicted Contacts`} size="small" color="primary" variant="outlined" />
                </Stack>
            </Stack>

            <Box sx={{ height: 400, width: '100%', cursor: 'crosshair', borderRadius: 2, overflow: 'hidden' }}>
                <ForceGraph2D
                    ref={graphRef}
                    graphData={{
                        nodes: data?.nodes || [],
                        links: data?.links || []
                    }}
                    nodeLabel={(node) => `Residue: ${node.label}`}
                    nodeColor={() => isDark ? '#00e5ff' : '#00695c'}
                    nodeRelSize={4}
                    linkColor={(link) => link.type === 'backbone' ? (isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.1)') : (isDark ? 'rgba(0,229,255,0.4)' : 'rgba(0,105,92,0.4)')}
                    linkWidth={(link) => link.type === 'backbone' ? 1.5 : 0.5}
                    backgroundColor={isDark ? 'transparent' : '#f5f5f5'}
                    d3AlphaDecay={0.01}
                    d3VelocityDecay={0.3}
                />
            </Box>
            <Typography variant="body2" sx={{ mt: 1, fontStyle: 'italic', fontSize: '0.75rem', textAlign: 'center' }} color="text.secondary">
                Graph uses ESM-2 embedding similarity to estimate residue-residue proximity beyond sequential distance.
            </Typography>
        </Paper>
    );
};

export default ResidueInteractionGraph;
