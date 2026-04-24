import React from 'react';
import { motion } from 'framer-motion';
import { 
  BookOpen, 
  Cpu, 
  Activity, 
  ShieldCheck, 
  Zap,
  Globe,
  Users,
  Lightbulb
} from 'lucide-react';

const About = () => {
  return (
    <div className="max-w-6xl mx-auto space-y-16 pb-24">
      
      {/* Friendly Header */}
      <section className="text-center space-y-8 pt-8">
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="inline-flex items-center gap-2 px-6 py-2 bg-emerald-500/10 rounded-full text-emerald-600 text-xs font-black uppercase tracking-[0.2em]"
        >
          <Lightbulb size={14} className="animate-pulse" /> How it Works: Simplified
        </motion.div>
        
        <h1 className="text-5xl font-black text-slate-800 tracking-tight leading-tight">
           Understanding the Magic behind <span className="logo-cursive text-emerald-500">TransGraph PPI</span>
        </h1>
        
        <p className="text-xl text-slate-500 max-w-3xl mx-auto leading-relaxed">
          Predicting how proteins interact is like solving a massive puzzle. Our system uses three "Specialized Experts" to solve this puzzle together.
        </p>
      </section>

      {/* The 3 Experts Section */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {[
          { 
            icon: BookOpen, 
            title: "Expert 1: The Language Reader", 
            text: "Imagine an AI that has read millions of protein 'books'. It understands the secret alphabet of amino acids and can tell if two proteins are 'speaking the same language' to bind together.",
            color: "bg-emerald-500",
            label: "ESM-2 Intelligence"
          },
          { 
            icon: Users, 
            title: "Expert 2: The Social Mapper", 
            text: "Proteins live in complex 'neighborhoods'. This expert looks at the social network of the cell to see if two proteins have the same friends or live close enough to ever meet and interact.",
            color: "bg-indigo-500",
            label: "Graph Relationship AI"
          },
          { 
            icon: ShieldCheck, 
            title: "Expert 3: The Final Jury", 
            text: "Finally, these two experts present their evidence to a head judge. The judge combines their opinions to give you a single, high-confidence score that you can trust.",
            color: "bg-amber-500",
            label: "Ensemble Master"
          }
        ].map((item, i) => (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            key={i} 
            className="glass-card p-10 flex flex-col items-center text-center group"
          >
            <div className={`w-20 h-20 rounded-3xl ${item.color} text-white mb-8 flex items-center justify-center shadow-2xl group-hover:scale-110 transition-transform duration-500`}>
              <item.icon size={40} />
            </div>
            <h3 className="text-2xl font-black text-slate-800 mb-4">{item.title}</h3>
            <p className="text-base text-slate-500 leading-relaxed font-medium mb-6">{item.text}</p>
            <span className="mt-auto text-[10px] font-black text-slate-300 uppercase tracking-widest border-t border-slate-100 pt-4 w-full">
              {item.label}
            </span>
          </motion.div>
        ))}
      </div>

      {/* Technical Blueprint (Simplified) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
        
        <div className="glass-card p-12 bg-white relative overflow-hidden group">
          <div className="absolute top-0 left-0 w-2 h-full bg-emerald-500" />
          <div className="flex items-center gap-4 mb-10">
             <div className="p-3 bg-emerald-50 rounded-2xl text-emerald-600"><Cpu size={24} /></div>
             <h3 className="text-2xl font-black text-slate-800">The Biological Brain</h3>
          </div>
          
          <div className="space-y-10">
             <div className="flex gap-6 group">
                <div className="w-12 h-12 rounded-2xl bg-slate-50 flex items-center justify-center text-slate-400 group-hover:bg-emerald-50 group-hover:text-emerald-500 transition-colors shrink-0 font-black">01</div>
                <div>
                   <h4 className="text-lg font-bold text-slate-800 mb-2">Reading the Blueprint</h4>
                   <p className="text-sm text-slate-500 leading-relaxed">
                      We take raw protein sequences and turn them into mathematical 'fingerprints'. 
                      This allows the AI to compare them just like a fingerprint scanner.
                   </p>
                </div>
             </div>

             <div className="flex gap-6 group">
                <div className="w-12 h-12 rounded-2xl bg-slate-50 flex items-center justify-center text-slate-400 group-hover:bg-indigo-50 group-hover:text-indigo-500 transition-colors shrink-0 font-black">02</div>
                <div>
                   <h4 className="text-lg font-bold text-slate-800 mb-2">Drawing the Network</h4>
                   <p className="text-sm text-slate-500 leading-relaxed">
                      The AI builds a massive map of how proteins interact across the whole human body. 
                      It uses this map to find hidden patterns that humans might miss.
                   </p>
                </div>
             </div>

             <div className="flex gap-6 group">
                <div className="w-12 h-12 rounded-2xl bg-slate-50 flex items-center justify-center text-slate-400 group-hover:bg-amber-50 group-hover:text-amber-500 transition-colors shrink-0 font-black">03</div>
                <div>
                   <h4 className="text-lg font-bold text-slate-800 mb-2">Calibrated Results</h4>
                   <p className="text-sm text-slate-500 leading-relaxed">
                      Instead of a simple 'Yes' or 'No', we give you a percentage. 
                      This tells you exactly how confident the AI is about its discovery.
                   </p>
                </div>
             </div>
          </div>
        </div>

        <div className="flex flex-col justify-between gap-8">
           <div className="glass-card p-12 bg-indigo-600 text-white relative overflow-hidden flex-1 shadow-indigo-200">
              <div className="relative z-10">
                 <div className="w-16 h-16 bg-white/10 rounded-2xl flex items-center justify-center mb-8"><Activity size={32} /></div>
                 <h3 className="text-3xl font-black mb-6 leading-tight">Ready for Modern Research</h3>
                 <p className="text-indigo-100 text-base mb-10 leading-relaxed opacity-80">
                    Our platform is designed to be as fast as a search engine but as accurate as a laboratory experiment. 
                    It helps researchers find the next big breakthrough in record time.
                 </p>
                 <div className="flex gap-4">
                    <button className="px-8 py-3 bg-white text-indigo-600 rounded-2xl font-black text-xs uppercase tracking-widest hover:bg-emerald-50 transition-all active:scale-95 shadow-xl">
                       View Case Studies
                    </button>
                    <button className="px-8 py-3 bg-indigo-500/50 text-white border border-white/10 rounded-2xl font-black text-xs uppercase tracking-widest hover:bg-indigo-500 transition-all active:scale-95">
                       Contact Support
                    </button>
                 </div>
              </div>
              <Globe size={200} className="absolute -right-20 -bottom-20 text-white/5 rotate-12" />
           </div>

           <div className="flex items-center justify-between px-6 p-4 glass-card bg-white">
              <div className="flex items-center gap-3">
                 <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                 <span className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em]">Model: v3.1_LATEST</span>
              </div>
              <div className="flex items-center gap-4">
                <span className="text-[10px] font-black text-indigo-400 uppercase tracking-widest">Accuracy Target: 95.0%</span>
                <div className="w-px h-4 bg-slate-100" />
                <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest italic">Academic Build</span>
              </div>
           </div>
        </div>
      </div>
    </div>
  );
};

export default About;
