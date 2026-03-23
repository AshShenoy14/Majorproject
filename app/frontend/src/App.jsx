import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import Predict from './pages/Predict';
import MutationAnalysis from './pages/MutationAnalysis';
import StructureViewer from './pages/StructureViewer';
import NetworkExplorer from './pages/NetworkExplorer';
import DrugInsights from './pages/DrugInsights';
import About from './pages/About';

function App() {
    return (
        <Router>
            <Layout>
                <Routes>
                    <Route path="/" element={<Home />} />
                    <Route path="/predict" element={<Predict />} />
                    <Route path="/mutation" element={<MutationAnalysis />} />
                    <Route path="/structure" element={<StructureViewer />} />
                    <Route path="/network" element={<NetworkExplorer />} />
                    <Route path="/drug-targets" element={<DrugInsights />} />
                    <Route path="/about" element={<About />} />
                </Routes>
            </Layout>
        </Router>
    );
}

export default App;
