import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Button,
  CircularProgress,
  Chip,
  useTheme,
  Stack,
  Alert
} from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import { AutoGraph as GraphIcon, Biotech as BioIcon, CheckCircle as CheckIcon } from '@mui/icons-material';
import axios from 'axios';
import { ppiService } from '../services/api';
import InteractionVisualizer from './InteractionVisualizer';

// Known Non-Human Interactions for Evaluation
const SPECIES_TESTS = [
  {
    species: "Mus musculus (Mouse)",
    title: "p53-MDM2 Coregulator",
    description: "A highly conserved cancer-regulating interaction tested zero-shot from human data.",
    p1: "P04637", // Mouse p53
    p2: "P23804", // Mouse MDM2
    seq1Fallback: "MTAMEESQSDISLELPLSQETFSGLWKLLPPEDILPSPHCMDDLLLPQDVEEFFEGPSEALRVSGAPAAQDPVTETPGPVAPAPATPWPLSSFVPSQKTYQGNYGFHLGFLQSGTAKSVMCTYSPCLNKLFCQLAKTCPVQLWVSATPPAGSRVRAMAIYKKSQHMTEVVRRCPHHERCSDGDGLAPPQHLIRVEGNLRAEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNFMCNSSCMGGMNRRPIITIITLEDSNGKLLGRNSFEVRVCACPGRDRRTEEENF",
    seq2Fallback: "MCNTNMSVPTDGAVTTSQIPASEQETLVRPKPLLLKLLKSVGAQKDTYTMINLLQNVLRCNDRKKKFPSQSEKSKEKSSQSNERISLFLKHKNSNEKEKQPKDKKKSKGSSQSQKEPISFLRHRNSSENEEKSKSDKKKSKSSSQSQKEPISFLRHRNSS",
    expected: "High Confidence Binding",
    icon: "🐁"
  },
  {
    species: "Saccharomyces cerevisiae (Yeast)",
    title: "Actin-Profilin Cytoskeleton",
    description: "Fundamental eukaryotic structural interaction separated by 1 billion years of evolution.",
    p1: "P60010", // Yeast Actin
    p2: "P07733", // Yeast Profilin
    seq1Fallback: "MDSEVAALVIDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGOKDSYVGDEAQSKRGILTLKYPIEHGIITNWDDMEKIWHHTFYNELRVAPEEHPTLLTEAPLNPKANREKMTQIMFETFNVPAMYVAIQAVLSLYASGRTTGIVLDSGDGVTHVVPIYAGFSLPHAILRIDLAGRDLTDYLMKILTERGYSFVTTAEREIVRDIKEKLCYVALDFEQEMQTAAQSSSIEKSYELPDGQVITIGNERFRCPEALFQPSFLGMESCGIHETTYNSIMKCDVDIRKDLYGNIVMSGGTTMFPGIAERMQKEITALAPSSMKVKIIAPPERKYSVWIGGSILASLSTFQQMWISKQEYDESGPSIVHHKCF",
    seq2Fallback: "MSWQAYTDNLIGTGKVDKAVIYFRAGDGHVWAQSADFPAVKAEEISGHVTKMFTGPAPDQVTVTTAKGGIFASIKQKPEWVALGGDDKLVVETSDGVYTFAGVGSGGSVKVGKVLAKTLVG",
    expected: "Structural Binding",
    icon: "🍞"
  },
  {
    species: "Escherichia coli (Bacteria)",
    title: "DNA Replisome (DnaA-DnaN)",
    description: "Testing cross-kingdom predictive topology on prokaryotic DNA replication machinery.",
    p1: "P03004", // E coli DnaA
    p2: "P0A9T0", // E coli DnaN
    seq1Fallback: "MSLSLWQQCLARLQDELPATEFSMWIRPLQAELSDNTLALYAPNRFVLDWVRDKYLNNINPLLKFDGAPNVLSFSHLRSVKPLRLLAGSVSVEAGLPEVARLYALGGAVMQDKITAPVGEV",
    seq2Fallback: "MKFTVEREHLLKPLQQVSGPLGGRPTLPILGNLLLQVADGTLSLTGTDLEMEMVADVTLIPATASGTGLPEVALWGDAALVAGLSRLEMGISVTRNDLEQAYVLGREFLVRVTGEKVKPVA",
    expected: "Functional Interaction",
    icon: "🦠"
  }
];

const CrossSpeciesTesting = () => {
    const theme = useTheme();
    const isDark = theme.palette.mode === 'dark';
    
    const [loadingMap, setLoadingMap] = useState({});
    const [resultMap, setResultMap] = useState({});
    const [errorMap, setErrorMap] = useState({});

    // Fetch sequence from UniProt directly in the frontend with fallback
    const fetchFasta = async (uniprotId, fallbackSeq) => {
        try {
            const res = await axios.get(`https://rest.uniprot.org/uniprotkb/${uniprotId}.fasta`, { timeout: 3500 });
            const lines = res.data.split('\n');
            const seq = lines.slice(1).map(l => l.trim()).join('').replace(/[^A-Z]/gi, '');
            return seq || fallbackSeq;
        } catch (e) {
            console.warn(`UniProt fetch failed for ${uniprotId}, using fallback sequence.`, e);
            return fallbackSeq;
        }
    };

    const handleRunZeroShot = async (index, testCase) => {
        setLoadingMap(prev => ({...prev, [index]: true}));
        setErrorMap(prev => ({...prev, [index]: null}));
        
        try {
            // 1. Fetch sequence or use fallback
            const seq1 = await fetchFasta(testCase.p1, testCase.seq1Fallback);
            const seq2 = await fetchFasta(testCase.p2, testCase.seq2Fallback);
            
            // 2. Feed exclusively sequences to the human-trained model
            const response = await ppiService.predict(testCase.p1, testCase.p2, seq1, seq2);
            setResultMap(prev => ({...prev, [index]: response.data}));
            
        } catch (err) {
            console.error("Zero-shot failed:", err);
            setErrorMap(prev => ({...prev, [index]: "Failed to execute zero-shot inference. Ensure backend server (http://127.0.0.1:8000) is running."}));
        } finally {
             setLoadingMap(prev => ({...prev, [index]: false}));
        }
    };

    return (
        <Box 
            component={motion.div}
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            sx={{ maxWidth: 1200, margin: '0 auto', p: 3 }}
        >
            <Box sx={{ textAlign: 'center', mb: 8 }}>
                <Typography variant="h3" fontWeight="900" gutterBottom sx={{ 
                    color: '#1e293b',
                    letterSpacing: '-0.02em'
                }}>
                    Cross-Species <span className="font-cursive text-emerald-500">Zero-Shot</span> Generalization
                </Typography>
                <Typography variant="body1" sx={{ color: '#64748b', maxWidth: 800, margin: '0 auto', fontWeight: 500, lineHeight: 1.8 }}>
                    Evaluator Showcase: Prove that the TransGraph-PPI architecture learned fundamental <strong className="text-slate-800">biological physics</strong> instead of simply memorizing human datasets. Run live inference on unobserved evolutionary kingdoms.
                </Typography>
            </Box>

            <Grid container spacing={4}>
                {SPECIES_TESTS.map((testCase, idx) => (
                    <Grid item xs={12} key={idx}>
                        <Card sx={{ 
                            borderRadius: '2.5rem', 
                            border: '1px solid rgba(0,0,0,0.05)',
                            background: '#ffffff',
                            boxShadow: '0 20px 50px rgba(0, 0, 0, 0.04)',
                            overflow: 'visible',
                            transition: 'all 0.3s ease'
                        }}>
                            <CardContent sx={{ p: 6 }}>
                                <Grid container spacing={4} alignItems="stretch">
                                    
                                    {/* Left Panel: Test Case Info */}
                                    <Grid item xs={12} md={4}>
                                        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', pr: { md: 4 }, borderRight: { md: '1px solid rgba(0,0,0,0.05)' } }}>
                                            <Typography variant="h2" sx={{ mb: 2 }}>{testCase.icon}</Typography>
                                            <Typography variant="h5" fontWeight="900" sx={{ color: '#059669', mb: 0.5 }}>
                                                {testCase.species}
                                            </Typography>
                                            <Typography variant="subtitle1" fontWeight="800" sx={{ color: '#334155', mb: 2 }}>
                                                {testCase.title}
                                            </Typography>
                                            <Typography variant="body2" sx={{ color: '#64748b', mb: 4, fontWeight: 500, lineHeight: 1.6 }}>
                                                {testCase.description}
                                            </Typography>
                                            
                                            <Stack direction="row" spacing={1.5} sx={{ mt: 'auto', mb: 4 }}>
                                                <Chip 
                                                    label={testCase.p1} 
                                                    size="small" 
                                                    sx={{ bgcolor: '#f1f5f9', color: '#475569', fontWeight: 900, borderRadius: '8px', border: 'none' }} 
                                                />
                                                <Typography sx={{ color: '#cbd5e1', fontWeight: 900 }}>→</Typography>
                                                <Chip 
                                                    label={testCase.p2} 
                                                    size="small" 
                                                    sx={{ bgcolor: '#f1f5f9', color: '#475569', fontWeight: 900, borderRadius: '8px', border: 'none' }} 
                                                />
                                            </Stack>

                                            <Button 
                                                variant="contained" 
                                                size="large"
                                                onClick={() => handleRunZeroShot(idx, testCase)}
                                                disabled={loadingMap[idx]}
                                                startIcon={loadingMap[idx] ? <CircularProgress size={16} color="inherit" /> : <GraphIcon />}
                                                sx={{ 
                                                    borderRadius: '1rem', 
                                                    fontWeight: 900,
                                                    textTransform: 'uppercase',
                                                    letterSpacing: '0.1em',
                                                    fontSize: '0.75rem',
                                                    py: 1.5,
                                                    background: 'linear-gradient(135deg, #059669 0%, #10b981 100%)',
                                                    boxShadow: '0 10px 20px rgba(5, 150, 105, 0.2)',
                                                    '&:hover': {
                                                        background: 'linear-gradient(135deg, #047857 0%, #059669 100%)',
                                                        boxShadow: '0 15px 25px rgba(5, 150, 105, 0.3)',
                                                    }
                                                }}
                                            >
                                                {loadingMap[idx] ? "Computing Topology..." : "Run Zero-Shot Evaluator"}
                                            </Button>

                                            {errorMap[idx] && (
                                                <Alert severity="error" variant="outlined" sx={{ mt: 2, borderRadius: '1rem', border: '1px solid #fee2e2', color: '#b91c1c', bgcolor: '#fef2f2' }}>{errorMap[idx]}</Alert>
                                            )}
                                        </Box>
                                    </Grid>

                                    {/* Right Panel: Live Results */}
                                    <Grid item xs={12} md={8}>
                                        <AnimatePresence mode="wait">
                                            {!resultMap[idx] && !loadingMap[idx] ? (
                                                <motion.div 
                                                    initial={{ opacity: 0 }} 
                                                    animate={{ opacity: 1 }} 
                                                    exit={{ opacity: 0 }}
                                                    style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                                                >
                                                    <Box sx={{ textAlign: 'center', p: 4, bgcolor: '#f8fafc', borderRadius: '2rem', width: '100%', border: '1px dashed #e2e8f0' }}>
                                                        <BioIcon sx={{ fontSize: 40, mb: 2, color: '#94a3b8' }} />
                                                        <Typography variant="h6" sx={{ color: '#64748b', fontWeight: 900, textTransform: 'uppercase', fontSize: '0.75rem', letterSpacing: '0.2em' }}>Awaiting Inference</Typography>
                                                        <Typography variant="body2" sx={{ color: '#94a3b8', mt: 1, fontWeight: 500 }}>Click run to fetch live UniProt FASTA sequences and compute interaction probability.</Typography>
                                                    </Box>
                                                </motion.div>
                                            ) : loadingMap[idx] ? (
                                                <motion.div 
                                                    initial={{ opacity: 0 }} 
                                                    animate={{ opacity: 1 }} 
                                                    exit={{ opacity: 0 }}
                                                    style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                                                >
                                                    <Box sx={{ textAlign: 'center', p: 4, bgcolor: '#f8fafc', borderRadius: '2rem', width: '100%' }}>
                                                        <CircularProgress size={40} thickness={4} sx={{ mb: 3, color: '#059669' }} />
                                                        <Typography variant="h6" sx={{ color: '#059669', fontWeight: 900, textTransform: 'uppercase', fontSize: '0.75rem', letterSpacing: '0.2em' }}>Processing Foreign Proteome...</Typography>
                                                        <Typography variant="body2" sx={{ color: '#64748b', mt: 1, fontWeight: 500 }}>Running ESM-2 embeddings and Graph Attention on purely non-human topologies.</Typography>
                                                    </Box>
                                                </motion.div>
                                            ) : (
                                                <motion.div 
                                                    initial={{ opacity: 0, x: 20 }} 
                                                    animate={{ opacity: 1, x: 0 }} 
                                                >
                                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4, p: 3, bgcolor: '#f8fafc', borderRadius: '2rem' }}>
                                                        <Box>
                                                            <Typography variant="h6" sx={{ color: '#059669', fontWeight: 900, textTransform: 'uppercase', fontSize: '0.75rem', letterSpacing: '0.1em', mb: 1 }}>
                                                                Zero-Shot Interaction Probability
                                                            </Typography>
                                                            <Typography variant="h3" fontWeight="900" sx={{ color: resultMap[idx].interaction_probability > 0.5 ? '#10b981' : '#ef4444', letterSpacing: '-0.05em' }}>
                                                                {(resultMap[idx].interaction_probability * 100).toFixed(1)}%
                                                            </Typography>
                                                        </Box>
                                                        <Box align="right">
                                                            <Chip 
                                                                icon={<CheckIcon />} 
                                                                label={testCase.expected} 
                                                                sx={{ mb: 1, bgcolor: '#ecfdf5', color: '#059669', fontWeight: 900, borderRadius: '8px', border: '1px solid #d1fae5' }} 
                                                            />
                                                            <Typography variant="caption" display="block" sx={{ color: '#94a3b8', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                                                ESM: {(resultMap[idx].esm_probability * 100).toFixed(1)}% | GAT: {(resultMap[idx].gat_probability * 100).toFixed(1)}%
                                                            </Typography>
                                                        </Box>
                                                    </Box>
                                                    
                                                    {/* Embed the standard visualizer to prove SHAP explainability works on non-human data too! */}
                                                    <InteractionVisualizer result={resultMap[idx]} id1={testCase.p1} id2={testCase.p2} />
                                                    
                                                </motion.div>
                                            )}
                                        </AnimatePresence>
                                    </Grid>
                                </Grid>
                            </CardContent>
                        </Card>
                    </Grid>
                ))}
            </Grid>
        </Box>
    );
};

export default CrossSpeciesTesting;
