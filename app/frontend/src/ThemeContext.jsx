import React, { createContext, useState, useMemo, useEffect } from 'react';
import { ThemeProvider } from '@mui/material/styles';
import { getThemeByMode } from './theme';

export const ThemeContext = createContext({
    toggleTheme: () => { },
    mode: 'dark'
});

export const ThemeContextProvider = ({ children }) => {
    // Check local storage or default to dark
    const [mode, setMode] = useState(() => {
        const savedMode = localStorage.getItem('appTheme');
        return savedMode || 'dark';
    });

    useEffect(() => {
        localStorage.setItem('appTheme', mode);
    }, [mode]);

    const contextValue = useMemo(() => ({
        toggleTheme: () => {
            setMode((prevMode) => (prevMode === 'light' ? 'dark' : 'light'));
        },
        mode,
    }), [mode]);

    const theme = useMemo(() => getThemeByMode(mode), [mode]);

    return (
        <ThemeContext.Provider value={contextValue}>
            <ThemeProvider theme={theme}>
                {children}
            </ThemeProvider>
        </ThemeContext.Provider>
    );
};
