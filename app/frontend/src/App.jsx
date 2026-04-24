import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import Predict from './pages/Predict';
import MutationAnalysis from './pages/MutationAnalysis';
import StructureViewer from './pages/StructureViewer';
import NetworkExplorer from './pages/NetworkExplorer';
import NetworkExplorer3D from './pages/Network3D';
import DrugInsights from './pages/DrugInsights';
import Assistant from './pages/Assistant';
import About from './pages/About';
import ProteinPreloader from './components/ProteinPreloader';
import CrossSpeciesTesting from './components/CrossSpeciesTesting';

function App() {
    const [loading, setLoading] = useState(true);
    const [progress, setProgress] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            setProgress((prev) => {
                if (prev >= 100) {
                    clearInterval(interval);
                    return 100;
                }
                // Random increments for a more natural feel
                const increment = Math.random() * 15 + 5;
                return Math.min(prev + increment, 100);
            });
        }, 400); // Progress over ~2.5 - 3 seconds

        return () => clearInterval(interval);
    }, []);

    const handleLoadingComplete = () => {
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
                            </Routes>
                        </Layout>
                    </Router>
                </div>
            )}
        </>
    );
}

export default App;
