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
  Loader2
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
      try {
        // 1. Fetch top centrality proteins first to define the "landscape"
        const centralityRes = await ppiService.getCentrality(50);
        const centralProteins = centralityRes.data;
        
        // 2. Fetch drug targets for these proteins
        const proteinsCsv = centralProteins.map(p => p.protein_id || p.protein).join(',');
        const drugRes = await ppiService.getDrugTargets(proteinsCsv);
        const drugData = drugRes.data;

        // 3. Map drug data onto centrality data
        const enrichedData = centralProteins.map(p => {
          const proteinId = p.protein_id || p.protein;
          const drugTarget = drugData.find(d => d.protein_id === proteinId);
          return {
            ...p,
            protein_id: proteinId,
            centrality_score: p.degree_centrality || p.degree || 0,
            is_drug_target: !!drugTarget,
            chembl_id: drugTarget?.chembl_id || null,
            uniprot_id: p.uniprot_id || drugTarget?.uniprot_id || 'N/A'
          };
        });

        setData(enrichedData);
      } catch (err) {
        console.error("Drug insights error:", err);
        setError("Failed to load drug target insights.");
        // #region agent log
        fetch('http://127.0.0.1:7656/ingest/a2c6930f-0198-499d-9920-7d735f885f13', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Debug-Session-Id': 'e579db'
          },
          body: JSON.stringify({
            sessionId: 'e579db',
            runId: 'pre-fix',
            hypothesisId: 'H1',
            location: 'DrugInsights.jsx:fetchData',
            message: 'Error fetching drug targets',
            data: {
              baseURL: ppiService?.defaults?.baseURL || null,
              endpoint: '/drug_targets',
              errorMessage: err?.message || null,
              errorCode: err?.code || null
            },
            timestamp: Date.now()
          })
        }).catch(() => {});
        // #endregion
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const filteredData = data.filter(item => {
    const matchesSearch = item.protein_id?.toLowerCase().includes(searchQuery.toLowerCase());
    if (filter === 'all') return matchesSearch;
    if (filter === 'drug_target') return matchesSearch && item.is_drug_target;
    if (filter === 'high_centrality') return matchesSearch && (item.centrality_score || 0) > 0.05; // Adjusted threshold for hub detection
    return matchesSearch;
  });

  const getChartData = () => {
    return filteredData
      .sort((a, b) => b.centrality_score - a.centrality_score)
      .slice(0, 8)
      .map(item => ({
        name: (item.protein_id || 'N/A').substring(0, 10),
        score: (Number(item.centrality_score || 0) * 100).toFixed(1),
        druggability: item.is_drug_target ? 100 : 40
      }));
  };

  return (
    <div className="space-y-8 pb-12">
      {/* Header & Stats */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 glass-card p-10 bg-scientific-gradient text-white flex justify-between items-center relative overflow-hidden shadow-2xl shadow-emerald-200">
           <div className="relative z-10 space-y-4">
              <div className="flex items-center gap-4">
                 <div className="p-3 bg-white/20 backdrop-blur-md rounded-2xl">
                    <ShieldAlert size={28} />
                 </div>
                 <div>
                    <h2 className="text-3xl font-black tracking-tight leading-none">Therapeutic <span className="font-cursive text-emerald-100">Insights</span></h2>
                    <p className="text-[10px] font-black uppercase tracking-[0.4em] text-emerald-200/60 mt-1">Druggability Landscape</p>
                 </div>
              </div>
              <p className="text-emerald-50/90 max-w-sm text-sm font-medium leading-relaxed">
                 Discover high-centrality proteins that represent critical nodes in pathological pathways, identifying them as prime candidates for therapeutic targeting.
              </p>
           </div>
           <div className="relative z-10 flex gap-4">
              <div className="text-center p-5 bg-white/10 backdrop-blur-xl rounded-[2rem] border border-white/20 min-w-[110px] shadow-xl">
                 <p className="text-3xl font-black">42</p>
                 <p className="text-[9px] font-black uppercase tracking-widest text-emerald-200">Known Targets</p>
              </div>
              <div className="text-center p-5 bg-emerald-400/30 backdrop-blur-xl rounded-[2rem] border border-white/20 min-w-[110px] shadow-xl">
                 <p className="text-3xl font-black">18</p>
                 <p className="text-[9px] font-black uppercase tracking-widest text-emerald-200">Novel Leads</p>
              </div>
           </div>
           {/* Decorative elements */}
           <div className="absolute -right-16 -top-16 w-64 h-64 rounded-full bg-white/10 blur-3xl" />
           <div className="absolute -right-8 -bottom-8 w-32 h-32 rounded-full bg-emerald-400/10 blur-2xl" />
        </div>

        <div className="glass-card p-6 flex flex-col justify-center">
           <div className="flex items-center gap-2 mb-4 text-slate-800">
              <TrendingUp size={20} className="text-scientific-primary" />
              <h3 className="text-sm font-bold uppercase tracking-widest">Target Ranking</h3>
           </div>
           <div className="h-40">
              <ResponsiveContainer width="100%" height="100%">
                 <BarChart data={getChartData()}>
                    <Bar dataKey="score" radius={[4, 4, 0, 0]}>
                       {getChartData().map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={index % 2 === 0 ? '#0D9488' : '#7C3AED'} />
                       ))}
                    </Bar>
                    <Tooltip cursor={{ fill: 'transparent' }} content={() => null} />
                 </BarChart>
              </ResponsiveContainer>
           </div>
           <p className="mt-4 text-[10px] text-slate-400 font-medium italic">Relative centrality scores for top nodes.</p>
        </div>
      </div>

      {/* Filter & Search Bar */}
      <div className="glass-card p-4 flex flex-col md:flex-row gap-4 items-center justify-between">
        <div className="flex items-center gap-2 w-full md:w-96 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
          <input 
            type="text" 
            placeholder="Search Protein ID..." 
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-scientific-primary outline-none transition-all text-sm"
          />
        </div>

        <div className="flex items-center gap-2 overflow-x-auto w-full md:w-auto pb-2 md:pb-0">
           <Filter size={18} className="text-slate-400 mr-2" />
           {[
             { id: 'all', label: 'All Clusters' },
             { id: 'drug_target', label: 'Verified Targets' },
             { id: 'high_centrality', label: 'High Centrality' }
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
                <th className="px-6 py-5 text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">Protein ID</th>
                <th className="px-6 py-5 text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">Criticality</th>
                <th className="px-6 py-5 text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">Pathway</th>
                <th className="px-6 py-5 text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">Target Status</th>
                <th className="px-6 py-5 text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">Compounds</th>
                <th className="px-6 py-5 text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {loading ? (
                <tr>
                  <td colSpan="6" className="px-6 py-12 text-center">
                    <div className="flex flex-col items-center gap-3">
                       <Loader2 className="animate-spin text-scientific-primary" size={32} />
                       <p className="text-sm font-bold text-slate-400 uppercase tracking-widest">Querying ChEMBL Latent Space...</p>
                    </div>
                  </td>
                </tr>
              ) : filteredData.map((item, i) => (
                <motion.tr 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: i * 0.05 }}
                  key={item.protein_id} 
                  className="hover:bg-slate-50/50 transition-colors group"
                >
                  <td className="px-6 py-4">
                    <div className="flex flex-col">
                       <span className="text-sm font-bold text-slate-700">{item.protein_id}</span>
                       <span className="text-[10px] text-slate-400 font-medium">UniProt: {item.uniprot_id || 'N/A'}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-2">
                       <div className="flex-1 h-1.5 w-16 bg-slate-100 rounded-full overflow-hidden">
                          <div className="h-full bg-scientific-primary" style={{ width: `${(item.centrality_score || 0) * 100}%` }} />
                       </div>
                       <span className="text-xs font-bold text-slate-500">{Number(item.centrality_score || 0).toFixed(3)}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-2">
                       <Tag size={12} className="text-scientific-accent" />
                       <span className="text-xs font-semibold text-slate-600 truncate max-w-[120px]">{item.pathway || 'Metabolic'}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    {item.is_drug_target ? (
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-green-50 text-green-600 text-[10px] font-bold border border-green-100">
                        <CheckCircle2 size={12} /> VERIFIED
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-100 text-slate-500 text-[10px] font-bold border border-slate-200">
                        <AlertCircle size={12} /> POTENTIAL
                      </span>
                    )}
                  </td>
                  <td className="px-6 py-4">
                     <span className="text-xs font-bold text-slate-700">{item.chembl_id ? 1 : 0} Lead(s)</span>
                  </td>
                  <td className="px-6 py-4">
                    <button className="p-2 text-slate-400 hover:text-scientific-primary hover:bg-scientific-primary/5 rounded-lg transition-all">
                       <ExternalLink size={18} />
                    </button>
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
        {!loading && filteredData.length === 0 && (
          <div className="p-12 text-center space-y-4">
             <Search size={48} className="text-slate-100 mx-auto" />
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
