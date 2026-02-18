import { AppBar, Toolbar, Typography, Button, Box, Container, useTheme } from '@mui/material';
import { motion } from 'framer-motion';

const Navbar = () => {
    const theme = useTheme();

    return (
        <AppBar position="sticky"
            sx={{
                background: 'rgba(255, 255, 255, 0.8)',
                backdropFilter: 'blur(10px)',
                boxShadow: '0 4px 30px rgba(0, 0, 0, 0.05)',
                borderBottom: '1px solid rgba(255, 255, 255, 0.3)'
            }}
        >
            <Container maxWidth="lg">
                <Toolbar disableGutters>
                    {/* Logo / Title */}
                    <Typography
                        variant="h5"
                        noWrap
                        component={motion.div}
                        initial={{ x: -20, opacity: 0 }}
                        animate={{ x: 0, opacity: 1 }}
                        sx={{
                            mr: 2,
                            display: { xs: 'none', md: 'flex' },
                            fontFamily: 'monospace',
                            fontWeight: 700,
                            letterSpacing: '.1rem',
                            color: theme.palette.primary.main,
                            textDecoration: 'none',
                            flexGrow: 1,
                            cursor: 'pointer'
                        }}
                    >
                        TransGraph-PPI
                    </Typography>

                    {/* Mobile Title (simplified) */}
                    <Typography
                        variant="h6"
                        noWrap
                        component="a"
                        href="/"
                        sx={{
                            mr: 2,
                            display: { xs: 'flex', md: 'none' },
                            flexGrow: 1,
                            fontFamily: 'monospace',
                            fontWeight: 700,
                            letterSpacing: '.1rem',
                            color: theme.palette.primary.main,
                            textDecoration: 'none',
                        }}
                    >
                        TG-PPI
                    </Typography>

                    {/* Navigation Items */}
                    <Box sx={{ flexGrow: 0, display: 'flex', gap: 2 }}>
                        {['Predict', 'Verify', 'About'].map((page) => (
                            <Button
                                key={page}
                                sx={{
                                    my: 2,
                                    color: 'text.primary',
                                    display: 'block',
                                    fontWeight: 500,
                                    '&:hover': {
                                        color: theme.palette.primary.main,
                                        backgroundColor: 'transparent'
                                    }
                                }}
                            >
                                {page}
                            </Button>
                        ))}
                    </Box>
                </Toolbar>
            </Container>
        </AppBar>
    );
};

export default Navbar;
