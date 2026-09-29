import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, Zap, Pill, Dna, BarChart3, Lightbulb, Users, Share2, CheckCircle, Activity } from 'lucide-react';
import { Link } from 'react-router-dom';
import { ppiService } from '../services/api';
import HeroProtein3D from '../components/HeroProtein3D';
import { TechOnly } from '../components/TechDetails';

const StatCard = ({ icon: Icon, value, label, explanation, color }) => (
  <div className="glass-card p-6 flex flex-col items-start">
    <div className={`p-3 rounded-xl mb-4 ${color}`}>
      <Icon size={24} className="text-white" aria-hidden="true" />
    </div>
    <p className="text-3xl font-bold text-slate-900">{value}</p>
    <p className="text-base font-semibold text-slate-800 mt-1">{label}</p>
    <p className="text-sm text-slate-600 mt-2">{explanation}</p>
  </div>
);

const TASKS = [
  { to: '/predict', icon: Zap, title: 'Predict an interaction', text: 'Pick two proteins and see how likely they are to work together, and why.' },
  { to: '/drug-targets', icon: Pill, title: 'Find drug targets', text: 'See which proteins are most central in the network and already linked to medicines.' },
  { to: '/mutation', icon: Dna, title: 'Test a mutation', text: 'Change one building block of a protein and see whether the interaction gets weaker.' },
  { to: '/benchmark', icon: BarChart3, title: 'Check the accuracy', text: 'See how often the model is right, and where it still struggles.' },
];

const EXAMPLES = [
  { title: 'TP53 and MDM2', tag: 'Cancer', desc: 'A famous pair: MDM2 switches off TP53, the cell\'s "guardian" against cancer.', p1: 'ENSP00000269305', p2: 'ENSP00000258149' },
  { title: 'AP2A2 and CLTC', tag: 'Brain health', desc: 'Two proteins that help cells take in material, linked to Alzheimer\'s research.', p1: 'ENSP00000300161', p2: 'ENSP00000267029' },
  { title: 'BAX and BCL2L1', tag: 'Cell survival', desc: 'Proteins that decide whether a damaged cell lives or dies.', p1: 'ENSP00000293879', p2: 'ENSP00000307677' },
  { title: 'A less-studied pair', tag: 'New protein', desc: 'Shows how the app handles a protein with few known connections.', p1: 'ENSP00000385802', p2: 'ENSP00000361000' },
];

const Home = () => {
  const [stats, setStats] = useState({ proteins: '—', interactions: '—', accuracy: '—', rocAuc: '—', testPairs: null });

  useEffect(() => {
    // All figures come from the backend; nothing is pre-filled with placeholder values.
    const fetchStats = async () => {
      try {
        const response = await ppiService.getNetworkStats('known');  // dataset figures: training interactions
        if (response.data?.num_nodes != null) {
          setStats(prev => ({
            ...prev,
            proteins: response.data.num_nodes.toLocaleString(),
            interactions: response.data.num_edges.toLocaleString()
          }));
        }
      } catch (error) {
        console.error("Error fetching network stats:", error);
      }
      try {
        const evalRes = await ppiService.getFinalEvaluation();
        const ensemble = Object.entries(evalRes.data?.models || {}).find(([name]) => name.includes('Ensemble'));
        if (ensemble) {
          setStats(prev => ({
            ...prev,
            accuracy: `${(ensemble[1].accuracy * 100).toFixed(2)}%`,
            rocAuc: ensemble[1].roc_auc.toFixed(3),
            testPairs: evalRes.data?.dataset_rows?.test_evaluated ?? null
          }));
        }
      } catch (error) {
        console.error("Error fetching final evaluation:", error);
      }
    };
    fetchStats();
  }, []);

  return (
    <div className="space-y-12">
      {/* Hero */}
      <section className="relative overflow-hidden rounded-[3rem] bg-white min-h-[440px] flex flex-col lg:flex-row items-center justify-between shadow-xl border border-slate-100">
        {/* Background image with a white fade so the text stays readable */}
        <div
          aria-hidden="true"
          className="absolute inset-0 z-0 bg-cover bg-center opacity-[0.05] pointer-events-none"
          style={{ backgroundImage: "url('/ppi_hero_bg.png')" }}
        />
        <div aria-hidden="true" className="absolute inset-0 z-0 bg-gradient-to-r from-white via-white/50 to-transparent pointer-events-none" />
        <div aria-hidden="true" className="absolute top-1/4 right-20 w-48 h-48 bg-emerald-100/20 blur-[80px] rounded-full pointer-events-none" />
        <div aria-hidden="true" className="absolute bottom-1/4 right-40 w-64 h-64 bg-teal-100/15 blur-[100px] rounded-full pointer-events-none" />

        <div className="relative z-10 p-10 lg:p-16 max-w-2xl">
          <p className="text-emerald-700 text-2xl font-bold mb-4" style={{ fontFamily: "'Dancing Script', cursive" }}>TransGraph PPI</p>
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-5xl lg:text-6xl font-black mb-6 leading-[1.1] text-slate-900 tracking-tight"
          >
            Will these two proteins <span className="text-emerald-700">work together?</span>
          </motion.h1>
          <p className="text-xl text-slate-700 mb-4 leading-relaxed">
            Proteins rarely work alone. TransGraph predicts whether two human proteins interact, explains
            why, and points to proteins that could be targets for new medicines.
          </p>
          <TechOnly>
            <p className="text-sm text-slate-700 mb-4 p-3 rounded-xl bg-slate-50 border border-slate-200">
              <strong>Technical:</strong> ESM-2 protein language model (sequence branch) + GraphSAGE graph neural
              network over the STRING network (graph branch), stacked by an XGBoost meta-learner and explained with SHAP.
              A standard GAT version is kept as a controlled comparison.
            </p>
          </TechOnly>
          <div className="flex flex-wrap gap-4 mt-6">
            <Link to="/predict" className="btn-primary shadow-2xl shadow-emerald-200">
              Try a prediction <ArrowRight size={20} aria-hidden="true" />
            </Link>
            <Link to="/how-it-works" className="px-7 py-3 bg-white text-slate-800 rounded-2xl font-bold border border-slate-300 hover:bg-slate-50 flex items-center gap-2">
              <Lightbulb size={18} aria-hidden="true" /> How it works
            </Link>
          </div>
        </div>
        <div className="relative z-10 w-full lg:w-[480px] h-[360px] lg:h-[460px] flex items-center justify-center lg:pr-8" aria-hidden="true">
          <HeroProtein3D />
        </div>
      </section>

      {/* What you can do */}
      <section aria-labelledby="tasks-heading" className="space-y-5">
        <h2 id="tasks-heading" className="text-2xl font-black text-slate-900">What you can do</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {TASKS.map(({ to, icon: Icon, title, text }) => (
            <Link key={to} to={to} className="glass-card p-6 bg-white hover:border-emerald-400 hover:shadow-lg transition-all flex flex-col gap-3 group focus-visible:outline focus-visible:outline-2 focus-visible:outline-emerald-600">
              <Icon size={28} className="text-emerald-700" aria-hidden="true" />
              <h3 className="text-lg font-bold text-slate-900">{title}</h3>
              <p className="text-sm text-slate-700 leading-relaxed flex-1">{text}</p>
              <span className="text-sm font-bold text-emerald-700 flex items-center gap-1">
                Open <ArrowRight size={15} aria-hidden="true" className="group-hover:translate-x-1 transition-transform" />
              </span>
            </Link>
          ))}
        </div>
      </section>

      {/* Numbers in plain words */}
      <section aria-labelledby="numbers-heading" className="space-y-5">
        <h2 id="numbers-heading" className="text-2xl font-black text-slate-900">The project in numbers</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <StatCard icon={Users} value={stats.proteins} label="Human proteins" explanation="Proteins in the network the model learned from." color="bg-teal-600" />
          <StatCard icon={Share2} value={stats.interactions} label="Known interactions" explanation="High-confidence pairs from the STRING database used for training." color="bg-blue-600" />
          <StatCard
            icon={CheckCircle}
            value={stats.accuracy}
            label="Predictions correct"
            explanation={`On ${stats.testPairs ? stats.testPairs.toLocaleString() : 'the'} test pairs the model never saw during training.`}
            color="bg-purple-600"
          />
          <StatCard icon={Activity} value={stats.rocAuc} label="Ranking quality" explanation="How well it tells interacting pairs from non-interacting ones (1.0 is perfect). Technical name: ROC-AUC." color="bg-orange-600" />
        </div>
      </section>

      {/* Examples */}
      <section aria-labelledby="examples-heading" className="glass-card p-8 bg-white space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-3">
          <div>
            <h2 id="examples-heading" className="text-2xl font-black text-slate-900">Try an example</h2>
            <p className="text-slate-700 mt-1">One click fills in both proteins on the Predict page. No IDs needed.</p>
          </div>
          <Link to="/predict" className="text-sm font-bold text-emerald-700 hover:underline flex items-center gap-1">
            Enter your own proteins <ArrowRight size={15} aria-hidden="true" />
          </Link>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {EXAMPLES.map((ex) => (
            <Link
              key={ex.title}
              to={`/predict?p1=${ex.p1}&p2=${ex.p2}`}
              className="p-5 rounded-2xl bg-white border border-slate-200 hover:border-emerald-400 hover:shadow-lg transition-all flex flex-col justify-between gap-4 focus-visible:outline focus-visible:outline-2 focus-visible:outline-emerald-600"
            >
              <div>
                <div className="flex items-center justify-between gap-2 mb-2">
                  <h3 className="font-bold text-slate-900">{ex.title}</h3>
                  <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200">{ex.tag}</span>
                </div>
                <p className="text-sm text-slate-700 leading-relaxed">{ex.desc}</p>
              </div>
              <span className="text-sm font-bold text-emerald-700 flex items-center gap-1">Run this example <ArrowRight size={15} aria-hidden="true" /></span>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
};

export default Home;
