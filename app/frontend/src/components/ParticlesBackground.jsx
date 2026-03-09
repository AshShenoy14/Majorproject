import React, { useContext, useMemo } from 'react';
import { motion } from 'framer-motion';
import { ThemeContext } from '../ThemeContext';

// Generates random particles that float and pulse
const ParticlesBackground = () => {
    const { mode } = useContext(ThemeContext);
    const isDark = mode === 'dark';

    const particles = useMemo(() => {
        return Array.from({ length: 35 }, (_, i) => ({
            id: i,
            x: Math.random() * 100,
            y: Math.random() * 100,
            size: Math.random() * 4 + 1.5,
            duration: Math.random() * 20 + 15,
            delay: Math.random() * 5,
            opacity: Math.random() * 0.4 + 0.1,
        }));
    }, []);

    const lines = useMemo(() => {
        return Array.from({ length: 12 }, (_, i) => ({
            id: i,
            x1: Math.random() * 100,
            y1: Math.random() * 100,
            x2: Math.random() * 100,
            y2: Math.random() * 100,
            duration: Math.random() * 25 + 20,
            delay: Math.random() * 8,
        }));
    }, []);

    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            zIndex: 0,
            pointerEvents: 'none',
            overflow: 'hidden',
        }}>
            {/* Background image layer */}
            <div style={{
                position: 'absolute',
                inset: 0,
                backgroundImage: 'url(/protein_bg.png)',
                backgroundSize: 'cover',
                backgroundPosition: 'center',
                backgroundRepeat: 'no-repeat',
                opacity: isDark ? 0.12 : 0.06,
                filter: isDark ? 'none' : 'grayscale(0.5)',
            }} />

            {/* Gradient overlay */}
            <div style={{
                position: 'absolute',
                inset: 0,
                background: isDark
                    ? `radial-gradient(ellipse at 20% 20%, rgba(0, 229, 255, 0.08) 0%, transparent 50%),
                       radial-gradient(ellipse at 80% 80%, rgba(213, 0, 249, 0.06) 0%, transparent 50%),
                       radial-gradient(ellipse at 50% 50%, rgba(0, 229, 255, 0.03) 0%, transparent 70%)`
                    : `radial-gradient(ellipse at 20% 20%, rgba(0, 105, 92, 0.05) 0%, transparent 50%),
                       radial-gradient(ellipse at 80% 80%, rgba(21, 101, 192, 0.04) 0%, transparent 50%)`,
            }} />

            {/* SVG layer for animated lines (network edges) */}
            <svg
                style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
                xmlns="http://www.w3.org/2000/svg"
            >
                {lines.map((line) => (
                    <motion.line
                        key={`line-${line.id}`}
                        x1={`${line.x1}%`}
                        y1={`${line.y1}%`}
                        x2={`${line.x2}%`}
                        y2={`${line.y2}%`}
                        stroke={isDark ? 'rgba(0, 229, 255, 0.06)' : 'rgba(0, 105, 92, 0.04)'}
                        strokeWidth="0.5"
                        initial={{ opacity: 0 }}
                        animate={{
                            opacity: [0, 0.4, 0],
                            x1: [`${line.x1}%`, `${(line.x1 + 10) % 100}%`],
                            y1: [`${line.y1}%`, `${(line.y1 + 8) % 100}%`],
                        }}
                        transition={{
                            duration: line.duration,
                            delay: line.delay,
                            repeat: Infinity,
                            ease: 'linear',
                        }}
                    />
                ))}
            </svg>

            {/* Floating particle nodes */}
            {particles.map((p) => (
                <motion.div
                    key={p.id}
                    style={{
                        position: 'absolute',
                        left: `${p.x}%`,
                        top: `${p.y}%`,
                        width: p.size,
                        height: p.size,
                        borderRadius: '50%',
                        background: isDark
                            ? `radial-gradient(circle, rgba(0, 229, 255, ${p.opacity}), transparent)`
                            : `radial-gradient(circle, rgba(0, 105, 92, ${p.opacity * 0.6}), transparent)`,
                        boxShadow: isDark
                            ? `0 0 ${p.size * 3}px rgba(0, 229, 255, ${p.opacity * 0.5})`
                            : `0 0 ${p.size * 2}px rgba(0, 105, 92, ${p.opacity * 0.3})`,
                    }}
                    animate={{
                        y: [0, -30, 0, 20, 0],
                        x: [0, 15, -10, 5, 0],
                        scale: [1, 1.3, 0.9, 1.1, 1],
                        opacity: [p.opacity, p.opacity * 1.5, p.opacity * 0.5, p.opacity * 1.2, p.opacity],
                    }}
                    transition={{
                        duration: p.duration,
                        delay: p.delay,
                        repeat: Infinity,
                        ease: 'easeInOut',
                    }}
                />
            ))}
        </div>
    );
};

export default ParticlesBackground;
