import React from 'react';
import { motion } from 'framer-motion';
import { 
  GitBranch, 
  Cpu, 
  Globe, 
  ShieldCheck, 
  Mail, 
  Github,
  Award,
  BookOpen,
  PieChart as PieIcon
} from 'lucide-react';

const About = () => {
  return (
    <div className="max-w-5xl mx-auto space-y-12 pb-20">
      {/* Platform Intro */}
      <section className="text-center space-y-6">
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="inline-flex items-center gap-2 px-4 py-2 bg-scientific-primary/10 rounded-full text-scientific-primary text-xs font-bold uppercase tracking-widest"
        >
          <Globe size={14} /> Global Biological Research Platform
        </motion.div>
        <h1 className="text-4xl font-extrabold text-slate-800">About TransGraph-PPI</h1>
        <p className="text-lg text-slate-500 max-w-2xl mx-auto leading-relaxed">
           A next-generation AI framework designed to accelerate the discovery of protein-protein interactions 
           using geometric deep learning and large-scale language models.
        </p>
      </section>

      {/* Methodology Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {[
          { 
            icon: Cpu, 
            title: "ESM-2 Embeddings", 
            text: "Utilizing 650M parameter protein language models to extract high-dimensional semantic features from sequences.",
            color: "bg-teal-500"
          },
          { 
            icon: GitBranch, 
            title: "GAT Networks", 
            text: "Graph Attention Networks analyze topological proximity and functional clusters in the protein interactome.",
            color: "bg-blue-500"
          },
          { 
            icon: ShieldCheck, 
            title: "XGBoost Ensemble", 
            text: "A robust meta-learner that combines sequence and graph evidence to provide confident, calibrated predictions.",
            color: "bg-purple-500"
          }
        ].map((item, i) => (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            key={i} 
            className="glass-card p-8 flex flex-col items-center text-center"
          >
            <div className={`p-4 rounded-2xl ${item.color} text-white mb-6 shadow-lg`}>
              <item.icon size={32} />
            </div>
            <h3 className="text-xl font-bold text-slate-800 mb-4">{item.title}</h3>
            <p className="text-sm text-slate-500 leading-relaxed font-medium">{item.text}</p>
          </motion.div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Architecture Section */}
        <div className="glass-card p-8">
          <div className="flex items-center gap-3 mb-8">
             <BookOpen className="text-scientific-primary" />
             <h3 className="text-xl font-bold text-slate-800">Research Architecture</h3>
          </div>
          <div className="space-y-6">
             <div className="flex gap-4">
                <div className="w-1.5 h-auto bg-scientific-primary rounded-full shrink-0" />
                <div>
                   <h4 className="text-sm font-bold text-slate-700 mb-1 uppercase tracking-tight">Sequence Module</h4>
                   <p className="text-xs text-slate-500 leading-relaxed">
                      Multi-layer perceptron architecture processing latent vectors from Meta's ESM-2 transformer. 
                      Captures conserved domains and binding motifs.
                   </p>
                </div>
             </div>
             <div className="flex gap-4">
                <div className="w-1.5 h-auto bg-scientific-secondary rounded-full shrink-0" />
                <div>
                   <h4 className="text-sm font-bold text-slate-700 mb-1 uppercase tracking-tight">Graph Module</h4>
                   <p className="text-xs text-slate-500 leading-relaxed">
                      Two-layer Graph Attention Network with 8-head multi-head attention. 
                      Learns structural dependencies on the Human STRING DB network.
                   </p>
                </div>
             </div>
             <div className="flex gap-4">
                <div className="w-1.5 h-auto bg-scientific-accent rounded-full shrink-0" />
                <div>
                   <h4 className="text-sm font-bold text-slate-700 mb-1 uppercase tracking-tight">SHAP Explainer</h4>
                   <p className="text-xs text-slate-500 leading-relaxed">
                      Feature attribution module using Shapley values to identify which model components 
                      drove the final interaction decision.
                   </p>
                </div>
             </div>
          </div>
        </div>

        {/* Contact & Community */}
        <div className="space-y-8 flex flex-col justify-center">
           <div className="glass-card p-8 bg-slate-900 text-white relative overflow-hidden">
              <div className="relative z-10">
                 <h3 className="text-2xl font-bold mb-4">Open-Source Intelligence</h3>
                 <p className="text-slate-400 text-sm mb-8 leading-relaxed">
                    This project is part of the ongoing effort to open-source advanced bioinformatics tools for drug discovery and molecular biology.
                 </p>
                 <div className="flex gap-4">
                    <button className="px-6 py-2 bg-white text-slate-900 rounded-xl font-bold text-sm flex items-center gap-2 hover:bg-slate-100 transition-colors">
                       <Github size={18} /> GITHUB REPO
                    </button>
                    <button className="px-6 py-2 bg-slate-800 text-white border border-slate-700 rounded-xl font-bold text-sm flex items-center gap-2 hover:bg-slate-700 transition-colors">
                       <Mail size={18} /> CONTACT LAB
                    </button>
                 </div>
              </div>
              <PieIcon size={120} className="absolute -right-8 -bottom-8 text-white/5 rotate-12" />
           </div>

           <div className="flex justify-between items-center px-4">
              <div className="flex items-center gap-2">
                 <Award className="text-warning" size={20} />
                 <span className="text-xs font-bold text-slate-500 uppercase tracking-widest">Research Citation v2.4</span>
              </div>
              <p className="text-[10px] text-slate-400 font-medium italic">Powered by DeepMind, Meta AI, and STRING</p>
           </div>
        </div>
      </div>
    </div>
  );
};

export default About;
