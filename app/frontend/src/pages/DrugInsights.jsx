import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ShieldAlert, 
  Filter, 
  Search, 
  ExternalLink, 
  TrendingUp, 
  PieChart as PieIcon,
  Tag,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Award,
  Info,
  Sparkles
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Cell 
} from 'recharts';
import { ppiService } from '../services/api';

const DrugInsights = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        // Fetch computational Therapeutic Target Priority Scores (TTPS)
        const res = await ppiService.getTherapeuticTargets(50, 0.40, 0.35, 0.25);
        setData(res.data || []);
      } catch (err) {
        console.error("Drug insights error:", err);
        setError("Failed to load computational therapeutic target priority scores.");
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const filteredData = data.filter(item => {
    const matchesSearch = 
      item.protein_id?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.uniprot_id?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (item.chembl_id && item.chembl_id.toLowerCase().includes(searchQuery.toLowerCase()));

    if (filter === 'all') return matchesSearch;
    if (filter === 'verified') return matchesSearch && item.is_chembl_target;
    if (filter === 'novel') return matchesSearch && !item.is_chembl_target;
    if (filter === 'high_priority') return matchesSearch && (item.ttps_score || 0) >= 0.40;
    return matchesSearch;
  });

  const verifiedCount = data.filter(d => d.is_chembl_target).length;
  const novelCount = data.filter(d => !d.is_chembl_target && (d.ttps_score || 0) >= 0.40).length;
  const avgTtps = data.length > 0 ? (data.reduce((acc, d) => acc + (d.ttps_score || 0), 0) / data.length).toFixed(3) : '0.000';

  const getChartData = () => {
    return [...filteredData]
      .sort((a, b) => (b.ttps_score || 0) - (a.ttps_score || 0))
      .slice(0, 8)
      .map(item => ({
        name: (item.protein_id || 'N/A').substring(0, 12),
        score: (Number(item.ttps_score || 0) * 100).toFixed(1),
        isVerified: item.is_chembl_target
      }));
  };

  return (
    <div className="space-y-8 pb-12">
      {/* Explanatory Guide Banner */}
      <div className="bg-slate-900 border border-emerald-500/30 text-slate-200 p-6 rounded-[2rem] flex items-start gap-4 shadow-xl">
        <div className="p-3 bg-emerald-500/20 rounded-2xl text-emerald-400 font-black text-xs uppercase tracking-widest shrink-0 flex items-center gap-1.5">
          <Sparkles size={16} />
          TTPS Scoring
        </div>
        <div className="space-y-2 text-xs leading-relaxed flex-1">
          <h4 className="font-bold text-white text-sm">Computational Therapeutic Target Priority Score (TTPS)</h4>
          <p className="text-slate-300">
            TransGraph prioritizes disease targets by integrating topological hubness with verified drug evidence:
            <code className="bg-slate-800 text-emerald-300 px-2 py-0.5 rounded text-[11px] font-mono ml-2">
              TTPS = 0.40 × NormDegree + 0.35 × NormBetweenness + 0.25 × ChEMBL_Target
            </code>
          </p>
          <div className="flex flex-wrap gap-4 text-[11px] text-slate-400 pt-1">
            <span><strong className="text-emerald-400">NormDegree (0.40):</strong> Direct interactome connections</span>
            <span><strong className="text-teal-400">NormBetweenness (0.35):</strong> Bottleneck traffic control</span>
            <span><strong className="text-purple-400">ChEMBL Target (0.25):</strong> Existing drug evidence</span>
          </div>
          <div className="mt-2 text-[10px] text-amber-300/90 font-medium flex items-center gap-1.5 border-t border-slate-800 pt-2">
            <Info size={12} className="shrink-0" />
            <span>Note: TTPS is a computational prioritization metric derived from graph topology and biological database cross-referencing to guide candidate selection. Downstream experimental validation is required.</span>
          </div>
        </div>
      </div>

      {/* Header & Summary Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 glass-card p-10 bg-scientific-gradient text-white flex justify-between items-center relative overflow-hidden shadow-2xl shadow-emerald-200">
           <div className="relative z-10 space-y-4">
              <div className="flex items-center gap-4">
                 <div className="p-3 bg-white/20 backdrop-blur-md rounded-2xl">
                    <Award size={28} />
                 </div>
                 <div>
                    <h2 className="text-3xl font-black tracking-tight leading-none">Therapeutic <span className="font-cursive text-emerald-100">Prioritization</span></h2>
                    <p className="text-[10px] font-black uppercase tracking-[0.4em] text-emerald-200/60 mt-1">Computational Target Ranking</p>
                 </div>
              </div>
              <p className="text-emerald-50/90 max-w-md text-sm font-medium leading-relaxed">
                 Identify high-priority target candidates combining topological centrality in the PPI interactome with ChEMBL drug cross-referencing.
              </p>
           </div>
           <div className="relative z-10 flex gap-4">
              <div className="text-center p-5 bg-white/10 backdrop-blur-xl rounded-[2rem] border border-white/20 min-w-[110px] shadow-xl">
                 <p className="text-3xl font-black">{loading ? '...' : verifiedCount}</p>
                 <p className="text-[9px] font-black uppercase tracking-widest text-emerald-200">Verified Targets</p>
              </div>
              <div className="text-center p-5 bg-emerald-400/30 backdrop-blur-xl rounded-[2rem] border border-white/20 min-w-[110px] shadow-xl">
                 <p className="text-3xl font-black">{loading ? '...' : novelCount}</p>
                 <p className="text-[9px] font-black uppercase tracking-widest text-emerald-200">Novel Leads</p>
              </div>
           </div>
           {/* Decorative elements */}
           <div className="absolute -right-16 -top-16 w-64 h-64 rounded-full bg-white/10 blur-3xl" />
           <div className="absolute -right-8 -bottom-8 w-32 h-32 rounded-full bg-emerald-400/10 blur-2xl" />
        </div>

        <div className="glass-card p-6 flex flex-col justify-center">
           <div className="flex items-center justify-between mb-4 text-slate-800">
              <div className="flex items-center gap-2">
                 <TrendingUp size={20} className="text-scientific-primary" />
                 <h3 className="text-sm font-bold uppercase tracking-widest">Top TTPS Targets</h3>
              </div>
              <span className="text-[10px] font-black text-slate-400">Mean: {avgTtps}</span>
           </div>
           <div className="h-40">
              {loading ? (
                 <div className="h-full flex items-center justify-center">
                    <Loader2 className="animate-spin text-scientific-primary" size={24} />
                 </div>
              ) : (
                 <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={getChartData()}>
                       <Bar dataKey="score" radius={[4, 4, 0, 0]}>
                          {getChartData().map((entry, index) => (
                             <Cell key={`cell-${index}`} fill={entry.isVerified ? '#0D9488' : '#7C3AED'} />
                          ))}
                       </Bar>
                       <Tooltip cursor={{ fill: 'transparent' }} content={({ payload }) => {
                          if (!payload || !payload.length) return null;
                          const d = payload[0].payload;
                          return (
                             <div className="bg-slate-900 text-white p-2 rounded text-xs shadow-lg border border-slate-700">
                                <p className="font-bold">{d.name}</p>
                                <p className="text-emerald-400">TTPS Score: {d.score}%</p>
                                <p className="text-slate-400 text-[10px]">{d.isVerified ? 'ChEMBL Verified' : 'Novel Lead Candidate'}</p>
                             </div>
                          );
                       }} />
                    </BarChart>
                 </ResponsiveContainer>
              )}
           </div>
           <div className="mt-4 flex items-center justify-between text-[10px] text-slate-400 font-medium">
              <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-teal-600 rounded-sm inline-block" /> Verified</span>
              <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-purple-600 rounded-sm inline-block" /> Novel Lead</span>
           </div>
        </div>
      </div>

      {/* Filter & Search Bar */}
      <div className="glass-card p-4 flex flex-col md:flex-row gap-4 items-center justify-between">
        <div className="flex items-center gap-2 w-full md:w-96 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
          <input 
            type="text" 
            placeholder="Search Protein ID, UniProt, ChEMBL..." 
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-scientific-primary outline-none transition-all text-sm"
          />
        </div>

        <div className="flex items-center gap-2 overflow-x-auto w-full md:w-auto pb-2 md:pb-0">
           <Filter size={18} className="text-slate-400 mr-2" />
           {[
             { id: 'all', label: 'All Candidates' },
             { id: 'verified', label: 'ChEMBL Verified' },
             { id: 'novel', label: 'Novel Candidates' },
             { id: 'high_priority', label: 'High Priority (TTPS ≥ 0.40)' }
           ].map(f => (
             <button
               key={f.id}
               onClick={() => setFilter(f.id)}
               className={`px-4 py-2 rounded-xl text-xs font-bold transition-all whitespace-nowrap ${filter === f.id ? 'bg-scientific-primary text-white shadow-md' : 'bg-slate-100 text-slate-500 hover:bg-slate-200'}`}
             >
               {f.label}
             </button>
           ))}
        </div>
      </div>

      {/* Data Table */}
      <div className="glass-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-100">
                <th className="px-6 py-5 text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">Rank & ID</th>
                <th className="px-6 py-5 text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">TTPS Priority Score</th>
                <th className="px-6 py-5 text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">Degree (Norm)</th>
                <th className="px-6 py-5 text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">Betweenness (Norm)</th>
                <th className="px-6 py-5 text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">ChEMBL Status</th>
                <th className="px-6 py-5 text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {loading ? (
                <tr>
                  <td colSpan="6" className="px-6 py-12 text-center">
                    <div className="flex flex-col items-center gap-3">
                       <Loader2 className="animate-spin text-scientific-primary" size={32} />
                       <p className="text-sm font-bold text-slate-400 uppercase tracking-widest">Computing Target Priority Scores...</p>
                    </div>
                  </td>
                </tr>
              ) : filteredData.map((item, i) => (
                <motion.tr 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: i * 0.03 }}
                  key={item.protein_id} 
                  className="hover:bg-slate-50/50 transition-colors group"
                >
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                       <span className="w-6 h-6 rounded-full bg-slate-100 font-bold text-xs text-slate-600 flex items-center justify-center shrink-0">
                          #{item.rank}
                       </span>
                       <div className="flex flex-col">
                          <span className="text-sm font-bold text-slate-700">{item.protein_id}</span>
                          <span className="text-[10px] text-slate-400 font-medium">UniProt: {item.uniprot_id || 'N/A'}</span>
                       </div>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                       <div className="flex-1 h-2.5 w-24 bg-slate-100 rounded-full overflow-hidden">
                          <div 
                            className={`h-full ${item.is_chembl_target ? 'bg-teal-600' : 'bg-purple-600'}`} 
                            style={{ width: `${Math.min(100, Math.max(0, (item.ttps_score || 0) * 100))}%` }} 
                          />
                       </div>
                       <span className="text-xs font-black text-slate-700 font-mono">
                          {Number(item.ttps_score || 0).toFixed(3)}
                       </span>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                     <span className="text-xs font-semibold text-slate-600 font-mono">{Number(item.norm_degree || 0).toFixed(3)}</span>
                  </td>
                  <td className="px-6 py-4">
                     <span className="text-xs font-semibold text-slate-600 font-mono">{Number(item.norm_betweenness || 0).toFixed(3)}</span>
                  </td>
                  <td className="px-6 py-4">
                    {item.is_chembl_target ? (
                      <div className="flex flex-col gap-1">
                        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-green-50 text-green-700 text-[10px] font-bold border border-green-200 shrink-0 w-fit">
                          <CheckCircle2 size={11} /> VERIFIED TARGET
                        </span>
                        {item.chembl_id && (
                          <span className="text-[10px] text-slate-400 font-mono">{item.chembl_id}</span>
                        )}
                      </div>
                    ) : (
                      <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-purple-50 text-purple-700 text-[10px] font-bold border border-purple-200 shrink-0 w-fit">
                        <AlertCircle size={11} /> NOVEL CANDIDATE
                      </span>
                    )}
                  </td>
                  <td className="px-6 py-4">
                    {item.chembl_id ? (
                      <a 
                        href={`https://www.ebi.ac.uk/chembl/target_report_card/${item.chembl_id}/`} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="p-2 text-slate-400 hover:text-scientific-primary hover:bg-scientific-primary/10 rounded-lg transition-all inline-block"
                        title="View on ChEMBL"
                      >
                         <ExternalLink size={16} />
                      </a>
                    ) : (
                      <span className="text-xs text-slate-300">—</span>
                    )}
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
        {!loading && filteredData.length === 0 && (
          <div className="p-12 text-center space-y-4">
             <Search size={48} className="text-slate-200 mx-auto" />
             <p className="text-slate-400 font-bold uppercase tracking-widest">No target matches found</p>
             <button onClick={() => {setSearchQuery(''); setFilter('all');}} className="text-scientific-primary text-xs font-bold hover:underline">
                RESET ALL FILTERS
             </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default DrugInsights;
