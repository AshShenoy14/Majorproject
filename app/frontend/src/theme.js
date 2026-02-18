import { createTheme } from '@mui/material/styles';

const theme = createTheme({
    palette: {
        mode: 'light',
        primary: {
            main: '#00695c', // Teal - often associated with biology/nature
            light: '#4db6ac',
            dark: '#004d40',
            contrastText: '#ffffff',
        },
        secondary: {
            main: '#1565c0', // Blue - typically associated with technology/science
            light: '#5e92f3',
            dark: '#003c8f',
            contrastText: '#ffffff',
        },
        background: {
            default: '#f4f6f8', // Very light grey/blue for a clean look
            paper: '#ffffff',
        },
        text: {
            primary: '#263238', // Dark blue-grey for readability
            secondary: '#546e7a',
        },
    },
    typography: {
        fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
        h1: {
            fontWeight: 700,
            fontSize: '2.5rem',
            letterSpacing: '-0.01562em',
            color: '#263238',
        },
        h2: {
            fontWeight: 600,
            fontSize: '2rem',
            letterSpacing: '-0.00833em',
            color: '#263238',
        },
        h3: {
            fontWeight: 600,
            fontSize: '1.75rem',
            color: '#37474f',
        },
        h4: {
            fontWeight: 500,
            fontSize: '1.5rem',
            color: '#37474f',
        },
        h6: {
            fontWeight: 500,
            fontSize: '1.25rem',
            lineHeight: 1.6,
        },
        body1: {
            fontSize: '1rem',
            lineHeight: 1.5,
            color: '#455a64',
        },
        button: {
            textTransform: 'none', // More modern feel than uppercase
            fontWeight: 600,
        },
    },
    shape: {
        borderRadius: 8, // Softens the UI
    },
    components: {
        MuiButton: {
            styleOverrides: {
                root: {
                    borderRadius: 20, // Pill-shaped buttons
                    boxShadow: 'none',
                    '&:hover': {
                        boxShadow: '0px 2px 4px -1px rgba(0,0,0,0.2), 0px 4px 5px 0px rgba(0,0,0,0.14), 0px 1px 10px 0px rgba(0,0,0,0.12)',
                    },
                },
                containedPrimary: {
                    background: 'linear-gradient(45deg, #00695c 30%, #4db6ac 90%)',
                },
                containedSecondary: {
                    background: 'linear-gradient(45deg, #1565c0 30%, #42a5f5 90%)',
                }
            },
        },
        MuiPaper: {
            styleOverrides: {
                elevation1: {
                    boxShadow: '0px 2px 8px rgba(0,0,0,0.05)', // Softer shadow
                },
                elevation3: {
                    boxShadow: '0px 4px 20px rgba(0,0,0,0.08)',
                }
            },
        },
        MuiCard: {
            styleOverrides: {
                root: {
                    overflow: 'visible', // Allow content to pop out if needed
                }
            }
        },
        MuiTextField: {
            styleOverrides: {
                root: {
                    '& .MuiOutlinedInput-root': {
                        borderRadius: 12,
                    }
                }
            }
        }
    },
});

export default theme;
