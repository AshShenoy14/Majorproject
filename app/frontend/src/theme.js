import { createTheme } from '@mui/material/styles';

export const getThemeByMode = (mode) => createTheme({
    palette: {
        mode,
        ...(mode === 'dark'
            ? {
                primary: {
                    main: '#00e5ff',
                    light: '#6effff',
                    dark: '#00b2cc',
                    contrastText: '#0a192f',
                },
                secondary: {
                    main: '#d500f9',
                    light: '#ff5bff',
                    dark: '#9e00c5',
                    contrastText: '#ffffff',
                },
                background: {
                    default: '#0a192f',
                    paper: 'rgba(16, 33, 65, 0.7)',
                },
                text: {
                    primary: '#e6f1ff',
                    secondary: '#8892b0',
                },
                divider: 'rgba(255, 255, 255, 0.08)',
            }
            : {
                primary: {
                    main: '#00695c',
                    light: '#4db6ac',
                    dark: '#004d40',
                    contrastText: '#ffffff',
                },
                secondary: {
                    main: '#1565c0',
                    light: '#5e92f3',
                    dark: '#003c8f',
                    contrastText: '#ffffff',
                },
                background: {
                    default: '#f4f6f8',
                    paper: 'rgba(255, 255, 255, 0.7)',
                },
                text: {
                    primary: '#1a202c',
                    secondary: '#4a5568',
                },
                divider: 'rgba(0, 0, 0, 0.08)',
            }),
    },
    typography: {
        fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
        h1: { fontFamily: '"Outfit", sans-serif', fontWeight: 800, fontSize: '3rem', letterSpacing: '-0.02em', color: mode === 'dark' ? '#e6f1ff' : '#1a202c' },
        h2: { fontFamily: '"Outfit", sans-serif', fontWeight: 700, fontSize: '2.5rem', letterSpacing: '-0.01em', color: mode === 'dark' ? '#e6f1ff' : '#1a202c' },
        h3: { fontFamily: '"Outfit", sans-serif', fontWeight: 600, fontSize: '2rem', color: mode === 'dark' ? '#e6f1ff' : '#1a202c' },
        h4: { fontFamily: '"Outfit", sans-serif', fontWeight: 600, fontSize: '1.5rem', color: mode === 'dark' ? '#ccd6f6' : '#2d3748' },
        h6: { fontFamily: '"Outfit", sans-serif', fontWeight: 500, fontSize: '1.25rem', lineHeight: 1.6 },
        body1: { fontSize: '1rem', lineHeight: 1.6, color: mode === 'dark' ? '#8892b0' : '#4a5568' },
        button: { fontFamily: '"Outfit", sans-serif', textTransform: 'none', fontWeight: 600, letterSpacing: '0.02em' },
    },
    shape: {
        borderRadius: 16,
    },
    components: {
        MuiCssBaseline: {
            styleOverrides: `
                body {
                    background-color: ${mode === 'dark' ? '#0a192f' : '#f4f6f8'};
                    background-image: ${mode === 'dark'
                    ? `radial-gradient(circle at 15% 50%, rgba(213, 0, 249, 0.08), transparent 25%), radial-gradient(circle at 85% 30%, rgba(0, 229, 255, 0.08), transparent 25%)`
                    : `radial-gradient(circle at 15% 50%, rgba(21, 101, 192, 0.04), transparent 25%), radial-gradient(circle at 85% 30%, rgba(0, 105, 92, 0.04), transparent 25%)`
                };
                    background-attachment: fixed;
                    color: ${mode === 'dark' ? '#e6f1ff' : '#1a202c'};
                }
            `
        },
        MuiButton: {
            styleOverrides: {
                root: {
                    borderRadius: 24,
                    padding: '8px 24px',
                    boxShadow: 'none',
                    transition: 'all 0.3s ease-in-out',
                    '&:hover': {
                        transform: 'translateY(-2px)',
                        boxShadow: '0 8px 20px rgba(0, 0, 0, 0.15)',
                    },
                },
                containedPrimary: {
                    background: mode === 'dark' ? 'linear-gradient(135deg, #00e5ff 0%, #00b2cc 100%)' : 'linear-gradient(135deg, #00695c 0%, #4db6ac 100%)',
                    color: mode === 'dark' ? '#0a192f' : '#ffffff',
                    '&:hover': {
                        background: mode === 'dark' ? 'linear-gradient(135deg, #6effff 0%, #00e5ff 100%)' : 'linear-gradient(135deg, #4db6ac 0%, #00695c 100%)',
                        boxShadow: mode === 'dark' ? '0 0 15px rgba(0, 229, 255, 0.4)' : '0 8px 15px rgba(0, 105, 92, 0.3)',
                    }
                },
                containedSecondary: {
                    background: mode === 'dark' ? 'linear-gradient(135deg, #d500f9 0%, #9e00c5 100%)' : 'linear-gradient(135deg, #1565c0 0%, #42a5f5 100%)',
                    color: '#ffffff',
                    '&:hover': {
                        background: mode === 'dark' ? 'linear-gradient(135deg, #ff5bff 0%, #d500f9 100%)' : 'linear-gradient(135deg, #42a5f5 0%, #1565c0 100%)',
                        boxShadow: mode === 'dark' ? '0 0 15px rgba(213, 0, 249, 0.4)' : '0 8px 15px rgba(21, 101, 192, 0.3)',
                    }
                },
                outlinedPrimary: {
                    borderWidth: '2px',
                    borderColor: mode === 'dark' ? '#00e5ff' : '#00695c',
                    color: mode === 'dark' ? '#00e5ff' : '#00695c',
                    '&:hover': {
                        borderWidth: '2px',
                        backgroundColor: mode === 'dark' ? 'rgba(0, 229, 255, 0.1)' : 'rgba(0, 105, 92, 0.1)',
                        borderColor: mode === 'dark' ? '#6effff' : '#004d40',
                    }
                }
            },
        },
        MuiPaper: {
            styleOverrides: {
                root: {
                    backdropFilter: 'blur(12px)',
                    border: `1px solid ${mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)'}`,
                    backgroundImage: 'none',
                },
                elevation1: { boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.1)' },
                elevation3: { boxShadow: '0 12px 40px 0 rgba(0, 0, 0, 0.15)' }
            },
        },
        MuiCard: {
            styleOverrides: {
                root: {
                    overflow: 'hidden',
                    backdropFilter: 'blur(12px)',
                    backgroundColor: mode === 'dark' ? 'rgba(16, 33, 65, 0.7)' : 'rgba(255, 255, 255, 0.7)',
                    border: `1px solid ${mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)'}`,
                }
            }
        },
        MuiTextField: {
            styleOverrides: {
                root: {
                    '& .MuiOutlinedInput-root': {
                        borderRadius: 12,
                        backgroundColor: mode === 'dark' ? 'rgba(255, 255, 255, 0.03)' : 'rgba(0, 0, 0, 0.02)',
                        transition: 'all 0.2s',
                        '& fieldset': { borderColor: mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)' },
                        '&:hover fieldset': { borderColor: mode === 'dark' ? 'rgba(255, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.2)' },
                        '&.Mui-focused fieldset': { borderColor: mode === 'dark' ? '#00e5ff' : '#00695c' },
                    }
                }
            }
        }
    },
});

export default getThemeByMode;
