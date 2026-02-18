import { Box, Container, Typography, Link, useTheme } from '@mui/material';

const Footer = () => {
    const theme = useTheme();

    return (
        <Box
            component="footer"
            sx={{
                py: 3,
                px: 2,
                mt: 'auto',
                backgroundColor: theme.palette.background.default,
                borderTop: `1px solid ${theme.palette.divider}`
            }}
        >
            <Container maxWidth="lg">
                <Typography variant="body2" color="text.secondary" align="center">
                    {'© '}
                    {new Date().getFullYear()}
                    {' TransGraph-PPI Project. All rights reserved.'}
                </Typography>
                <Typography variant="caption" display="block" color="text.secondary" align="center" sx={{ mt: 1 }}>
                    Powered by ESM-2, GAT, and XGBoost.
                </Typography>
            </Container>
        </Box>
    );
};

export default Footer;
