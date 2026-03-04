import { useState, useEffect } from 'react';
import axios from 'axios';
import {
    Box,
    Button,
    Typography,
    Paper,
    CircularProgress,
    Grid,
    Chip,
    Divider,
    List,
    ListItem,
    ListItemText,
    Card,
    CardActionArea,
    CardContent,
    TextField,
    InputAdornment,
    Collapse,
    IconButton
} from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
// Icons would ideally be imported from @mui/icons-material, but avoiding new deps for now if not installed.
// Assuming we can use text or basic shapes if icons are missing, or basic SVG.

const RealDataVerifier = () => {
    const [networkNodes, setNetworkNodes] = useState([]);
    const [filteredNodes, setFilteredNodes] = useState([]);
    const [searchTerm, setSearchTerm] = useState('');
    const [selectedProtein, setSelectedProtein] = useState(null);
    const [targets, setTargets] = useState([]);
    const [loadingNetwork, setLoadingNetwork] = useState(false);
    const [loadingTargets, setLoadingTargets] = useState(false);

    // Fetch Network on Load
    useEffect(() => {
        fetchNetwork();
    }, []);

    useEffect(() => {
        if (!searchTerm) {
            setFilteredNodes(networkNodes);
        } else {
            setFilteredNodes(networkNodes.filter(n => n.id.toLowerCase().includes(searchTerm.toLowerCase())));
        }
    }, [searchTerm, networkNodes]);

    const fetchNetwork = async () => {
        setLoadingNetwork(true);
        try {
            const res = await axios.get('http://localhost:8000/network?limit=50'); // Increased limit
            setNetworkNodes(res.data.nodes || []);
        } catch (e) {
            console.error("Failed to fetch network", e);
        } finally {
            setLoadingNetwork(false);
        }
    };

    const handleNodeClick = async (nodeId) => {
        if (selectedProtein === nodeId) return; // Don't refetch if already selected
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
        <Paper
            elevation={0}
            component={motion.div}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            sx={{
                p: 4,
                borderRadius: 4,
                overflow: 'hidden',
                background: 'rgba(16, 33, 65, 0.6)',
                backdropFilter: 'blur(20px)',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                boxShadow: '0 20px 40px rgba(0,0,0,0.2)'
            }}
        >
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Box>
                    <Typography variant="h5" color="primary" sx={{ fontWeight: 600 }}>
                        Real Data Verification
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Explore protein interaction networks and drug targets from the graph.
                    </Typography>
                </Box>
                <Button variant="outlined" onClick={fetchNetwork} disabled={loadingNetwork}>
                    Refresh Data
                </Button>
            </Box>

            <Divider sx={{ mb: 4 }} />

            <Grid container spacing={4}>
                {/* Network List */}
                <Grid item xs={12} md={5}>
                    <Box sx={{ mb: 2 }}>
                        <TextField
                            fullWidth
                            placeholder="Search Protein ID..."
                            variant="outlined"
                            size="small"
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            sx={{
                                '& .MuiOutlinedInput-root': {
                                    backgroundColor: 'rgba(255, 255, 255, 0.03)'
                                }
                            }}
                        />
                    </Box>

                    <Paper
                        variant="outlined"
                        sx={{
                            height: '500px',
                            overflow: 'auto',
                            p: 2,
                            bgcolor: 'rgba(0, 0, 0, 0.2)',
                            borderColor: 'rgba(255, 255, 255, 0.05)',
                            borderRadius: 2
                        }}
                    >
                        {loadingNetwork ? (
                            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
                                <CircularProgress />
                            </Box>
                        ) : (
                            <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(100px, 1fr))', gap: 1 }}>
                                {filteredNodes.length === 0 ? (
                                    <Typography color="text.secondary" align="center" sx={{ gridColumn: '1/-1', mt: 4 }}>
                                        No proteins found.
                                    </Typography>
                                ) : (
                                    filteredNodes.map(n => (
                                        <Chip
                                            key={n.id}
                                            label={n.id}
                                            onClick={() => handleNodeClick(n.id)}
                                            color={selectedProtein === n.id ? "primary" : "default"}
                                            variant={selectedProtein === n.id ? "filled" : "outlined"}
                                            clickable
                                            sx={{
                                                fontWeight: selectedProtein === n.id ? 'bold' : 'normal',
                                                transition: 'all 0.2s'
                                            }}
                                        />
                                    ))
                                )}
                            </Box>
                        )}
                    </Paper>
                </Grid>

                {/* Drug Targets Details */}
                <Grid item xs={12} md={7}>
                    <Paper
                        variant="outlined"
                        sx={{
                            height: '548px', // Match Search + List height approx
                            overflow: 'hidden',
                            display: 'flex',
                            flexDirection: 'column',
                            bgcolor: 'rgba(0, 0, 0, 0.2)',
                            borderColor: 'rgba(255, 255, 255, 0.05)',
                            borderRadius: 2
                        }}
                    >
                        <Box sx={{ p: 2, bgcolor: 'rgba(0, 229, 255, 0.05)', borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                            <Typography variant="h6" color="primary" sx={{ fontWeight: 600 }}>
                                {selectedProtein ? `Targets for ${selectedProtein}` : "Select a Protein"}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                                {selectedProtein ? "Source: ChEMBL Database" : "Click on a protein node to view known drug targets."}
                            </Typography>
                        </Box>

                        <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
                            {loadingTargets ? (
                                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mt: 8 }}>
                                    <CircularProgress size={30} sx={{ mb: 2 }} />
                                    <Typography color="text.secondary">Fetching targets...</Typography>
                                </Box>
                            ) : (
                                <AnimatePresence mode="wait">
                                    {selectedProtein ? (
                                        targets.length === 0 ? (
                                            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                                                <Box sx={{ textAlign: 'center', mt: 8, px: 4 }}>
                                                    <Typography variant="body1" color="text.secondary" paragraph>
                                                        No known drug targets found for this protein in the current dataset.
                                                    </Typography>
                                                    <Chip label="Try another node" size="small" />
                                                </Box>
                                            </motion.div>
                                        ) : (
                                            <List>
                                                {targets.map((t, i) => (
                                                    <Paper
                                                        key={i}
                                                        elevation={0}
                                                        component={motion.div}
                                                        initial={{ opacity: 0, x: 10 }}
                                                        animate={{ opacity: 1, x: 0 }}
                                                        transition={{ delay: i * 0.05 }}
                                                        sx={{
                                                            mb: 1.5,
                                                            border: '1px solid rgba(255, 255, 255, 0.05)',
                                                            bgcolor: 'rgba(255, 255, 255, 0.02)',
                                                            p: 2,
                                                            borderRadius: 2,
                                                            transition: 'all 0.2s',
                                                            '&:hover': {
                                                                bgcolor: 'rgba(255, 255, 255, 0.05)',
                                                                borderColor: 'rgba(0, 229, 255, 0.3)'
                                                            }
                                                        }}
                                                    >
                                                        <ListItem disablePadding>
                                                            <ListItemText
                                                                primary={
                                                                    <Typography variant="subtitle1" color="primary" sx={{ fontWeight: 500 }}>
                                                                        {t.target_name}
                                                                    </Typography>
                                                                }
                                                                secondary={
                                                                    <Box component="span" sx={{ display: 'block', mt: 0.5 }}>
                                                                        <Chip label={`ChEMBL: ${t.chembl_id}`} size="small" sx={{ mr: 1, fontSize: '0.7em', height: 20 }} />
                                                                        <Chip label={t.target_type} size="small" color="secondary" variant="outlined" sx={{ fontSize: '0.7em', height: 20 }} />
                                                                    </Box>
                                                                }
                                                            />
                                                        </ListItem>
                                                    </Paper>
                                                ))}
                                            </List>
                                        )
                                    ) : (
                                        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                                            <Box sx={{ textAlign: 'center', mt: 10, opacity: 0.5 }}>
                                                <Typography variant="h3" color="text.secondary" gutterBottom>
                                                    🔬
                                                </Typography>
                                                <Typography>Select a protein from the network list to begin analysis.</Typography>
                                            </Box>
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            )}
                        </Box>
                    </Paper>
                </Grid>
            </Grid>
        </Paper>
    );
};

export default RealDataVerifier;
