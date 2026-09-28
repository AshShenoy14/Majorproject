import React, { useState, useEffect, lazy, Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import ProteinPreloader from './components/ProteinPreloader';

// Primary pages loaded eagerly for instant navigation
import Home from './pages/Home';
import Predict from './pages/Predict';

// Heavy analytical and 3D sub-pages code-split on demand
const MutationAnalysis = lazy(() => import('./pages/MutationAnalysis'));
const StructureViewer = lazy(() => import('./pages/StructureViewer'));
const NetworkExplorer = lazy(() => import('./pages/NetworkExplorer'));
const NetworkExplorer3D = lazy(() => import('./pages/Network3D'));
const DrugInsights = lazy(() => import('./pages/DrugInsights'));
const Assistant = lazy(() => import('./pages/Assistant'));
const About = lazy(() => import('./pages/About'));
const Benchmark = lazy(() => import('./pages/Benchmark'));
const ComparisonMode = lazy(() => import('./pages/ComparisonMode'));
const CrossSpeciesTesting = lazy(() => import('./pages/CrossSpeciesTesting'));

function App() {
    const [loading, setLoading] = useState(() => {
        return !sessionStorage.getItem('transgraph_loaded');
    });
    const [progress, setProgress] = useState(0);

    useEffect(() => {
        if (!loading) return;

        const interval = setInterval(() => {
            setProgress((prev) => {
                if (prev >= 100) {
                    clearInterval(interval);
                    sessionStorage.setItem('transgraph_loaded', 'true');
                    return 100;
                }
                // Fast, smooth increments for responsive load feel
                const increment = Math.random() * 25 + 15;
                return Math.min(prev + increment, 100);
            });
        }, 150); // Snappy ~1 second initial load

        return () => clearInterval(interval);
    }, [loading]);

    const handleLoadingComplete = () => {
        sessionStorage.setItem('transgraph_loaded', 'true');
        setLoading(false);
    };

    return (
        <>
            {loading && (
                <ProteinPreloader 
                    progress={progress} 
                    onComplete={handleLoadingComplete} 
                />
            )}
            
            {!loading && (
                <div className="content-fade-in">
                    <Router>
                        <Layout>
                            <Suspense fallback={
                                <div className="flex flex-col items-center justify-center min-h-[350px] text-slate-500">
                                    <div className="w-8 h-8 rounded-full border-2 border-cyan-500 border-t-transparent animate-spin mb-3" />
                                    <span className="text-xs font-semibold tracking-wide uppercase text-slate-400">Loading module...</span>
                                </div>
                            }>
                                <Routes>
                                    <Route path="/" element={<Home />} />
                                    <Route path="/predict" element={<Predict />} />
                                    <Route path="/mutation" element={<MutationAnalysis />} />
                                    <Route path="/structure" element={<StructureViewer />} />
                                    <Route path="/network" element={<NetworkExplorer />} />
                                    <Route path="/network-3d" element={<NetworkExplorer3D />} />
                                    <Route path="/drug-targets" element={<DrugInsights />} />
                                    <Route path="/assistant" element={<Assistant />} />
                                    <Route path="/about" element={<About />} />
                                    <Route path="/zero-shot" element={<CrossSpeciesTesting />} />
                                    <Route path="/benchmark" element={<Benchmark />} />
                                    <Route path="/compare" element={<ComparisonMode />} />
                                </Routes>
                            </Suspense>
                        </Layout>
                    </Router>
                </div>
            )}
        </>
    );
}

export default App;
