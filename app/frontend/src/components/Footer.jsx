import { Box, Container, Typography, Link, useTheme } from '@mui/material';

const Footer = () => {
    const theme = useTheme();

    return (
        <Box
            component="footer"
            sx={{
                py: 4,
                px: 2,
                mt: 'auto',
                backgroundColor: 'rgba(10, 25, 47, 0.8)',
                backdropFilter: 'blur(10px)',
                borderTop: '1px solid rgba(255, 255, 255, 0.05)',
                position: 'relative',
                zIndex: 1
            }}
        >
            <Container maxWidth="lg">
                <Typography variant="body2" color="text.secondary" align="center" sx={{ fontFamily: '"Outfit", sans-serif' }}>
                    {'© '}
                    {new Date().getFullYear()}
                    {' TransGraph-PPI Project. All rights reserved.'}
                </Typography>
                <Typography variant="caption" display="block" color="primary" align="center" sx={{ mt: 1, textShadow: '0 0 10px rgba(0, 229, 255, 0.3)' }}>
                    Powered by ESM-2, GAT, and XGBoost.
                </Typography>
            </Container>
        </Box>
    );
};

export default Footer;
