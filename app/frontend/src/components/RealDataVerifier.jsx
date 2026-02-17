import { useState, useEffect } from 'react';
import axios from 'axios';
import {
    Box,
    Button,
    Typography,
    Paper,
    CircularProgress,
    Card,
    CardContent,
    Grid,
    Chip,
    Divider,
    List,
    ListItem,
    ListItemText
} from '@mui/material';
import { motion } from 'framer-motion';

const RealDataVerifier = () => {
    const [networkNodes, setNetworkNodes] = useState([]);
    const [selectedProtein, setSelectedProtein] = useState(null);
    const [targets, setTargets] = useState([]);
    const [loadingNetwork, setLoadingNetwork] = useState(false);
    const [loadingTargets, setLoadingTargets] = useState(false);

    // Fetch Network on Load
    useEffect(() => {
        fetchNetwork();
    }, []);

    const fetchNetwork = async () => {
        setLoadingNetwork(true);
        try {
            const res = await axios.get('http://localhost:8000/network?limit=20');
            setNetworkNodes(res.data.nodes || []);
        } catch (e) {
            console.error("Failed to fetch network", e);
        } finally {
            setLoadingNetwork(false);
        }
    };

    const handleNodeClick = async (nodeId) => {
        setSelectedProtein(nodeId);
        setLoadingTargets(true);
        try {
            const res = await axios.get(`http://localhost:8000/drug_targets?proteins=${nodeId}`);
            setTargets(res.data);
        } catch (e) {
            console.error("Failed to fetch targets", e);
            setTargets([]);
        } finally {
            setLoadingTargets(false);
        }
    };

    return (
        <Box sx={{ mt: 4 }}>
            <Typography variant="h5" gutterBottom>Real Data Verification</Typography>
            <Divider sx={{ mb: 2 }} />

            <Grid container spacing={3}>
                {/* Network List */}
                <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 2, height: '400px', overflow: 'auto' }}>
                        <Typography variant="h6">Real Network Nodes (from train.csv)</Typography>
                        <Button size="small" onClick={fetchNetwork} disabled={loadingNetwork}>Refresh</Button>

                        {loadingNetwork ? <CircularProgress /> : (
                            <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                                {networkNodes.length === 0 ? <Typography>No nodes found. Did you run the pipeline?</Typography> :
                                    networkNodes.map(n => (
                                        <Chip
                                            key={n.id}
                                            label={n.id}
                                            onClick={() => handleNodeClick(n.id)}
                                            color={selectedProtein === n.id ? "primary" : "default"}
                                            clickable
                                        />
                                    ))
                                }
                            </Box>
                        )}
                    </Paper>
                </Grid>

                {/* Drug Targets */}
                <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 2, height: '400px', overflow: 'auto' }}>
                        <Typography variant="h6">
                            Drug Targets (ChEMBL) {selectedProtein && `for ${selectedProtein}`}
                        </Typography>

                        {loadingTargets ? <CircularProgress /> : (
                            <List>
                                {targets.length === 0 ? <Typography sx={{ mt: 2 }}>Select a protein to view targets.</Typography> :
                                    targets.map((t, i) => (
                                        <ListItem key={i} divider>
                                            <ListItemText
                                                primary={t.target_name}
                                                secondary={`ChEMBL ID: ${t.chembl_id} | Type: ${t.target_type}`}
                                            />
                                        </ListItem>
                                    ))
                                }
                            </List>
                        )}
                    </Paper>
                </Grid>
            </Grid>
        </Box>
    );
};

export default RealDataVerifier;
