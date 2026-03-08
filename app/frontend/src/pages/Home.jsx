import React, { useContext } from 'react';
import { Box, Container, Typography, Button, Stack, Chip } from '@mui/material';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { ThemeContext } from '../ThemeContext';
import ScienceIcon from '@mui/icons-material/Science';
import BiotechIcon from '@mui/icons-material/Biotech';
import HubIcon from '@mui/icons-material/Hub';
import SpeedIcon from '@mui/icons-material/Speed';

const stagger = {
    animate: { transition: { staggerChildren: 0.12 } },
};

const fadeUp = {
    initial: { opacity: 0, y: 30 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.7, ease: [0.4, 0, 0.2, 1] } },
};

const features = [
    {
        icon: <ScienceIcon sx={{ fontSize: 36 }} />,
        title: 'ESM-2 Embeddings',
        desc: 'State-of-the-art protein language model for encoding amino acid sequences into rich numerical representations.',
    },
    {
        icon: <HubIcon sx={{ fontSize: 36 }} />,
        title: 'Graph Attention Network',
        desc: 'Learns interaction topology through multi-head attention on protein interaction graphs.',
    },
    {
        icon: <BiotechIcon sx={{ fontSize: 36 }} />,
        title: 'Ensemble Meta-Learner',
        desc: 'XGBoost meta-learner fuses predictions from multiple models for superior accuracy.',
    },
    {
        icon: <SpeedIcon sx={{ fontSize: 36 }} />,
        title: 'Real-Time Prediction',
        desc: 'Instant predictions with SHAP explainability and 3D molecular structure visualization.',
    },
];

const Home = () => {
    const { mode } = useContext(ThemeContext);
    const isDark = mode === 'dark';

    return (
        <Box sx={{ position: 'relative', overflow: 'hidden' }}>
            {/* ── Hero Section ───────────────────────── */}
            <Box sx={{
                pt: { xs: 10, md: 16 },
                pb: { xs: 8, md: 14 },
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: '85vh',
                position: 'relative',
            }}>
                {/* Top glow */}
                <Box sx={{
                    position: 'absolute',
                    top: '-20%',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    width: '140%',
                    height: '60%',
                    background: isDark
                        ? 'radial-gradient(ellipse, rgba(0, 229, 255, 0.12) 0%, transparent 70%)'
                        : 'radial-gradient(ellipse, rgba(0, 105, 92, 0.06) 0%, transparent 70%)',
                    pointerEvents: 'none',
                    zIndex: 0,
                }} />

                <Container maxWidth="md" sx={{ position: 'relative', zIndex: 1 }}>
                    <motion.div variants={stagger} initial="initial" animate="animate">
                        {/* Badge */}
                        <motion.div variants={fadeUp} style={{ display: 'flex', justifyContent: 'center', marginBottom: 24 }}>
                            <Chip
                                label="AI-Powered Bioinformatics"
                                variant="outlined"
                                sx={{
                                    borderColor: isDark ? 'rgba(0, 229, 255, 0.3)' : 'rgba(0, 105, 92, 0.3)',
                                    color: isDark ? '#00e5ff' : '#00695c',
                                    fontFamily: '"Outfit", sans-serif',
                                    fontWeight: 600,
                                    fontSize: '0.85rem',
                                    px: 2,
                                    py: 2.5,
                                    backdropFilter: 'blur(8px)',
                                    background: isDark ? 'rgba(0, 229, 255, 0.06)' : 'rgba(0, 105, 92, 0.04)',
                                    letterSpacing: '0.08em',
                                    '&:hover': {
                                        borderColor: isDark ? '#00e5ff' : '#00695c',
                                        background: isDark ? 'rgba(0, 229, 255, 0.12)' : 'rgba(0, 105, 92, 0.08)',
                                    },
                                    transition: 'all 0.3s ease',
                                }}
                            />
                        </motion.div>

                        {/* Main Heading */}
                        <motion.div variants={fadeUp}>
                            <Typography
                                variant="h1"
                                align="center"
                                gutterBottom
                                sx={{
                                    fontSize: { xs: '2.5rem', sm: '3.5rem', md: '4.8rem' },
                                    fontWeight: 900,
                                    lineHeight: 1.1,
                                    background: isDark
                                        ? 'linear-gradient(135deg, #ffffff 0%, #ccd6f6 50%, #8892b0 100%)'
                                        : 'linear-gradient(135deg, #1a202c 0%, #2d3748 100%)',
                                    WebkitBackgroundClip: 'text',
                                    WebkitTextFillColor: 'transparent',
                                    mb: 1,
                                }}
                            >
                                Predict Protein{' '}
                                <Box component="span" sx={{
                                    background: isDark
                                        ? 'linear-gradient(135deg, #00e5ff 0%, #7c4dff 50%, #d500f9 100%)'
                                        : 'linear-gradient(135deg, #00695c 0%, #1565c0 100%)',
                                    WebkitBackgroundClip: 'text',
                                    WebkitTextFillColor: 'transparent',
                                    position: 'relative',
                                    '&::after': isDark ? {
                                        content: '"Interactions"',
                                        position: 'absolute',
                                        left: 0,
                                        top: 0,
                                        background: 'linear-gradient(135deg, #00e5ff 0%, #7c4dff 50%, #d500f9 100%)',
                                        WebkitBackgroundClip: 'text',
                                        WebkitTextFillColor: 'transparent',
                                        filter: 'blur(20px)',
                                        opacity: 0.5,
                                        zIndex: -1,
                                    } : {},
                                }}>
                                    Interactions
                                </Box>
                            </Typography>
                        </motion.div>

                        {/* Subtitle */}
                        <motion.div variants={fadeUp}>
                            <Typography
                                variant="h5"
                                align="center"
                                sx={{
                                    maxWidth: '75%',
                                    mx: 'auto',
                                    lineHeight: 1.7,
                                    color: isDark ? '#8892b0' : '#4a5568',
                                    fontWeight: 400,
                                    fontSize: { xs: '1rem', md: '1.25rem' },
                                    mb: 5,
                                }}
                            >
                                Advanced PPI prediction using a Hybrid Ensemble of{' '}
                                <Box component="span" sx={{ color: isDark ? '#00e5ff' : '#00695c', fontWeight: 600 }}>
                                    ESM-2
                                </Box>{' '}
                                transformers and{' '}
                                <Box component="span" sx={{ color: isDark ? '#d500f9' : '#1565c0', fontWeight: 600 }}>
                                    Graph Attention Networks
                                </Box>
                            </Typography>
                        </motion.div>

                        {/* CTA Buttons */}
                        <motion.div variants={fadeUp}>
                            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} justifyContent="center">
                                <Button
                                    component={Link}
                                    to="/predict"
                                    variant="contained"
                                    color="primary"
                                    size="large"
                                    sx={{
                                        px: 5,
                                        py: 1.5,
                                        fontSize: '1.1rem',
                                        fontWeight: 700,
                                        borderRadius: '50px',
                                        boxShadow: isDark
                                            ? '0 0 30px rgba(0, 229, 255, 0.25), 0 8px 32px rgba(0, 0, 0, 0.3)'
                                            : '0 8px 32px rgba(0, 105, 92, 0.2)',
                                        '&:hover': {
                                            boxShadow: isDark
                                                ? '0 0 50px rgba(0, 229, 255, 0.4), 0 12px 40px rgba(0, 0, 0, 0.4)'
                                                : '0 12px 40px rgba(0, 105, 92, 0.35)',
                                            transform: 'translateY(-3px) scale(1.02)',
                                        },
                                        transition: 'all 0.35s cubic-bezier(0.4, 0, 0.2, 1)',
                                    }}
                                >
                                    Start Predicting →
                                </Button>
                                <Button
                                    component={Link}
                                    to="/about"
                                    variant="outlined"
                                    color="primary"
                                    size="large"
                                    sx={{
                                        px: 5,
                                        py: 1.5,
                                        fontSize: '1.1rem',
                                        borderRadius: '50px',
                                        borderWidth: 2,
                                        '&:hover': {
                                            borderWidth: 2,
                                            transform: 'translateY(-3px)',
                                            backgroundColor: isDark ? 'rgba(0, 229, 255, 0.08)' : 'rgba(0, 105, 92, 0.08)',
                                        },
                                        transition: 'all 0.35s cubic-bezier(0.4, 0, 0.2, 1)',
                                    }}
                                >
                                    Learn More
                                </Button>
                            </Stack>
                        </motion.div>
                    </motion.div>
                </Container>
            </Box>

            {/* ── Feature Cards ──────────────────────── */}
            <Container maxWidth="lg" sx={{ pb: 12 }}>
                <motion.div
                    initial="initial"
                    whileInView="animate"
                    viewport={{ once: true, amount: 0.2 }}
                    variants={stagger}
                >
                    <Box sx={{
                        display: 'grid',
                        gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', md: 'repeat(4, 1fr)' },
                        gap: 3,
                    }}>
                        {features.map((feat, i) => (
                            <motion.div key={i} variants={fadeUp}>
                                <Box
                                    className="glass-card"
                                    sx={{
                                        p: 3.5,
                                        borderRadius: 4,
                                        height: '100%',
                                        background: isDark
                                            ? 'rgba(16, 33, 65, 0.5)'
                                            : 'rgba(255, 255, 255, 0.6)',
                                        cursor: 'default',
                                        '&:hover .feat-icon': {
                                            transform: 'scale(1.15) rotate(-5deg)',
                                            color: isDark ? '#00e5ff' : '#00695c',
                                        },
                                    }}
                                >
                                    <Box
                                        className="feat-icon"
                                        sx={{
                                            color: isDark ? '#8892b0' : '#4a5568',
                                            mb: 2,
                                            transition: 'all 0.35s ease',
                                        }}
                                    >
                                        {feat.icon}
                                    </Box>
                                    <Typography
                                        variant="h6"
                                        sx={{
                                            fontFamily: '"Outfit", sans-serif',
                                            fontWeight: 700,
                                            mb: 1,
                                            color: isDark ? '#e6f1ff' : '#1a202c',
                                        }}
                                    >
                                        {feat.title}
                                    </Typography>
                                    <Typography
                                        variant="body2"
                                        sx={{
                                            color: isDark ? '#8892b0' : '#4a5568',
                                            lineHeight: 1.7,
                                        }}
                                    >
                                        {feat.desc}
                                    </Typography>
                                </Box>
                            </motion.div>
                        ))}
                    </Box>
                </motion.div>
            </Container>
        </Box>
    );
};

export default Home;
