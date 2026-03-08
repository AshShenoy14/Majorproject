import React, { useContext, useState, useEffect } from 'react';
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
    const [scrolled, setScrolled] = useState(false);

    useEffect(() => {
        const onScroll = () => setScrolled(window.scrollY > 20);
        window.addEventListener('scroll', onScroll, { passive: true });
        return () => window.removeEventListener('scroll', onScroll);
    }, []);

    const navItems = [
        { name: 'Home', path: '/' },
        { name: 'Predict', path: '/predict' },
        { name: 'Verify', path: '/verify' },
        { name: 'About', path: '/about' }
    ];

    return (
        <AppBar position="sticky"
            sx={{
                background: scrolled
                    ? (mode === 'dark' ? 'rgba(10, 25, 47, 0.85)' : 'rgba(255, 255, 255, 0.92)')
                    : (mode === 'dark' ? 'rgba(10, 25, 47, 0.4)' : 'rgba(255, 255, 255, 0.6)'),
                backdropFilter: scrolled ? 'blur(20px) saturate(180%)' : 'blur(12px)',
                boxShadow: scrolled
                    ? (mode === 'dark' ? '0 4px 30px rgba(0, 0, 0, 0.3)' : '0 4px 30px rgba(0, 0, 0, 0.08)')
                    : 'none',
                borderBottom: `1px solid ${scrolled
                    ? (mode === 'dark' ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.06)')
                    : 'transparent'}`,
                zIndex: theme.zIndex.drawer + 1,
                transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
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
                            textShadow: mode === 'dark' ? '0 0 20px rgba(0, 229, 255, 0.3)' : 'none',
                            transition: 'all 0.3s ease',
                            '&:hover': {
                                filter: 'brightness(1.2)',
                            },
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
                    <Box sx={{ flexGrow: 0, display: 'flex', gap: 1, alignItems: 'center' }}>
                        {navItems.map((item, i) => {
                            const isActive = location.pathname === item.path;
                            return (
                                <motion.div
                                    key={item.name}
                                    initial={{ y: -20, opacity: 0 }}
                                    animate={{ y: 0, opacity: 1 }}
                                    transition={{ duration: 0.5, delay: i * 0.08 }}
                                >
                                    <Button
                                        component={Link}
                                        to={item.path}
                                        sx={{
                                            my: 2,
                                            px: 2,
                                            color: isActive
                                                ? theme.palette.primary.main
                                                : mode === 'dark' ? '#ccd6f6' : '#4a5568',
                                            display: 'block',
                                            fontWeight: isActive ? 700 : 500,
                                            fontFamily: '"Outfit", sans-serif',
                                            letterSpacing: '0.04em',
                                            fontSize: '0.92rem',
                                            position: 'relative',
                                            borderRadius: '12px',
                                            background: isActive
                                                ? (mode === 'dark' ? 'rgba(0, 229, 255, 0.08)' : 'rgba(0, 105, 92, 0.06)')
                                                : 'transparent',
                                            '&::after': {
                                                content: '""',
                                                position: 'absolute',
                                                bottom: 6,
                                                left: '50%',
                                                transform: 'translateX(-50%)',
                                                width: isActive ? '40%' : 0,
                                                height: '2px',
                                                background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                                                borderRadius: '2px',
                                                transition: 'width 0.35s cubic-bezier(0.4, 0, 0.2, 1)',
                                            },
                                            '&:hover': {
                                                color: theme.palette.primary.main,
                                                backgroundColor: mode === 'dark' ? 'rgba(0, 229, 255, 0.06)' : 'rgba(0, 105, 92, 0.04)',
                                                '&::after': {
                                                    width: '40%',
                                                }
                                            },
                                            transition: 'all 0.3s ease',
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
                            transition={{ duration: 0.5, delay: 0.35 }}
                        >
                            <IconButton
                                onClick={toggleTheme}
                                sx={{
                                    ml: 1,
                                    color: mode === 'dark' ? '#f6e05e' : '#4a5568',
                                    transition: 'all 0.4s ease',
                                    background: mode === 'dark' ? 'rgba(246, 224, 94, 0.08)' : 'rgba(0, 0, 0, 0.04)',
                                    '&:hover': {
                                        transform: 'rotate(180deg) scale(1.1)',
                                        background: mode === 'dark' ? 'rgba(246, 224, 94, 0.15)' : 'rgba(0, 0, 0, 0.08)',
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
