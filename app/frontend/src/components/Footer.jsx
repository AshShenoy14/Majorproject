import { useContext } from 'react';
import { Box, Container, Typography } from '@mui/material';
import { motion } from 'framer-motion';
import { ThemeContext } from '../ThemeContext';

const Footer = () => {
    const { mode } = useContext(ThemeContext);
    const isDark = mode === 'dark';

    return (
        <Box
            component={motion.footer}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            sx={{
                py: 4,
                px: 2,
                mt: 'auto',
                position: 'relative',
                zIndex: 1,
                backgroundColor: isDark ? 'rgba(10, 25, 47, 0.7)' : 'rgba(255, 255, 255, 0.7)',
                backdropFilter: 'blur(16px) saturate(180%)',
                borderTop: `1px solid ${isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)'}`,
            }}
        >
            <Container maxWidth="lg">
                <Typography
                    variant="body2"
                    color="text.secondary"
                    align="center"
                    sx={{ fontFamily: '"Outfit", sans-serif' }}
                >
                    {'© '}
                    {new Date().getFullYear()}
                    {' TransGraph-PPI Project. All rights reserved.'}
                </Typography>
                <Typography
                    variant="caption"
                    display="block"
                    align="center"
                    sx={{
                        mt: 1,
                        fontFamily: '"Outfit", sans-serif',
                        fontWeight: 600,
                        background: isDark
                            ? 'linear-gradient(90deg, #00e5ff, #d500f9)'
                            : 'linear-gradient(90deg, #00695c, #1565c0)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                        letterSpacing: '0.04em',
                    }}
                >
                    Powered by ESM-2, GAT & XGBoost
                </Typography>
            </Container>
        </Box>
    );
};

export default Footer;
