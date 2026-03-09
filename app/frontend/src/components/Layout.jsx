import { Box } from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import { useLocation } from 'react-router-dom';
import Navbar from './Navbar';
import Footer from './Footer';
import ParticlesBackground from './ParticlesBackground';

const pageVariants = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -10 },
};

const pageTransition = {
    type: 'tween',
    ease: [0.4, 0, 0.2, 1],
    duration: 0.45,
};

const Layout = ({ children }) => {
    const location = useLocation();

    return (
        <Box sx={{
            display: 'flex',
            flexDirection: 'column',
            minHeight: '100vh',
            bgcolor: 'background.default',
            position: 'relative',
        }}>
            {/* Global animated background */}
            <ParticlesBackground />

            <Navbar />
            <Box
                component="main"
                sx={{
                    flexGrow: 1,
                    position: 'relative',
                    zIndex: 1,
                }}
            >
                <AnimatePresence mode="wait">
                    <motion.div
                        key={location.pathname}
                        variants={pageVariants}
                        initial="initial"
                        animate="animate"
                        exit="exit"
                        transition={pageTransition}
                    >
                        {children}
                    </motion.div>
                </AnimatePresence>
            </Box>
            <Footer />
        </Box>
    );
};

export default Layout;
