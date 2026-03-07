import React, { useContext } from 'react';
import { AppBar, Toolbar, Typography, Button, Box, Container, useTheme, IconButton } from '@mui/material';
import { motion } from 'framer-motion';
import { Link, useLocation } from 'react-router-dom';
import LightModeIcon from '@mui/icons-material/LightMode';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import { ThemeContext } from '../ThemeContext';

const Navbar = () => {
    const theme = useTheme();
    const { toggleTheme, mode } = useContext(ThemeContext);
    const location = useLocation();

    const navItems = [
        { name: 'Home', path: '/' },
        { name: 'Predict', path: '/predict' },
        { name: 'Verify', path: '/verify' },
        { name: 'About', path: '/about' }
    ];

    return (
        <AppBar position="sticky"
            sx={{
                background: mode === 'dark' ? 'rgba(10, 25, 47, 0.65)' : 'rgba(255, 255, 255, 0.85)',
                backdropFilter: 'blur(16px)',
                boxShadow: '0 4px 30px rgba(0, 0, 0, 0.1)',
                borderBottom: `1px solid ${mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)'}`,
                zIndex: theme.zIndex.drawer + 1
            }}
        >
            <Container maxWidth="lg">
                <Toolbar disableGutters>
                    {/* Logo / Title */}
                    <Typography
                        variant="h5"
                        noWrap
                        component={Link}
                        to="/"
                        sx={{
                            mr: 2,
                            display: { xs: 'none', md: 'flex' },
                            fontFamily: '"Outfit", sans-serif',
                            fontWeight: 800,
                            letterSpacing: '.1rem',
                            background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                            WebkitBackgroundClip: 'text',
                            WebkitTextFillColor: 'transparent',
                            textDecoration: 'none',
                            flexGrow: 1,
                            cursor: 'pointer',
                            textShadow: mode === 'dark' ? '0 0 20px rgba(0, 229, 255, 0.3)' : 'none'
                        }}
                    >
                        TransGraph-PPI
                    </Typography>

                    {/* Mobile Title */}
                    <Typography
                        variant="h6"
                        noWrap
                        component={Link}
                        to="/"
                        sx={{
                            mr: 2,
                            display: { xs: 'flex', md: 'none' },
                            flexGrow: 1,
                            fontFamily: '"Outfit", sans-serif',
                            fontWeight: 800,
                            background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                            WebkitBackgroundClip: 'text',
                            WebkitTextFillColor: 'transparent',
                            textDecoration: 'none',
                        }}
                    >
                        TG-PPI
                    </Typography>

                    {/* Navigation Items */}
                    <Box sx={{ flexGrow: 0, display: 'flex', gap: 2, alignItems: 'center' }}>
                        {navItems.map((item, i) => {
                            const isActive = location.pathname === item.path;
                            return (
                                <motion.div
                                    key={item.name}
                                    initial={{ y: -20, opacity: 0 }}
                                    animate={{ y: 0, opacity: 1 }}
                                    transition={{ duration: 0.5, delay: i * 0.1 }}
                                >
                                    <Button
                                        component={Link}
                                        to={item.path}
                                        sx={{
                                            my: 2,
                                            color: isActive
                                                ? theme.palette.primary.main
                                                : mode === 'dark' ? '#ccd6f6' : '#4a5568',
                                            display: 'block',
                                            fontWeight: isActive ? 700 : 600,
                                            fontFamily: '"Outfit", sans-serif',
                                            letterSpacing: '0.05em',
                                            position: 'relative',
                                            '&::after': {
                                                content: '""',
                                                position: 'absolute',
                                                bottom: 4,
                                                left: isActive ? '25%' : '50%',
                                                transform: isActive ? 'none' : 'translateX(-50%)',
                                                width: isActive ? '50%' : 0,
                                                height: '2px',
                                                backgroundColor: theme.palette.primary.main,
                                                transition: 'width 0.3s ease, left 0.3s ease, transform 0.3s ease',
                                            },
                                            '&:hover': {
                                                color: theme.palette.primary.main,
                                                backgroundColor: 'transparent',
                                                '&::after': {
                                                    width: '50%',
                                                    left: '25%',
                                                    transform: 'none'
                                                }
                                            }
                                        }}
                                    >
                                        {item.name}
                                    </Button>
                                </motion.div>
                            );
                        })}

                        {/* Theme Toggle Button */}
                        <motion.div
                            initial={{ y: -20, opacity: 0 }}
                            animate={{ y: 0, opacity: 1 }}
                            transition={{ duration: 0.5, delay: 0.4 }}
                        >
                            <IconButton
                                onClick={toggleTheme}
                                sx={{
                                    ml: 2,
                                    color: mode === 'dark' ? '#f6e05e' : '#4a5568',
                                    transition: 'transform 0.3s ease',
                                    '&:hover': {
                                        transform: 'rotate(45deg)'
                                    }
                                }}
                            >
                                {mode === 'dark' ? <LightModeIcon /> : <DarkModeIcon />}
                            </IconButton>
                        </motion.div>
                    </Box>
                </Toolbar>
            </Container>
        </AppBar>
    );
};

export default Navbar;
