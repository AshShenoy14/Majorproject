import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Filter,
  Search,
  ExternalLink,
  TrendingUp,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Award,
  Info
} from 'lucide-react';
import {
  BarChart,
  Bar,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';
import { ppiService } from '../services/api';
import { TechOnly } from '../components/TechDetails';

const DrugInsights = () => {
  const [searchParams] = useSearchParams();
  const urlQuery = searchParams.get('q') || searchParams.get('query') || searchParams.get('protein') || '';
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState(urlQuery);
  const [network, setNetwork] = useState('predicted');
  const [networkStats, setNetworkStats] = useState(null);

  useEffect(() => {
    if (urlQuery) {
      setSearchQuery(urlQuery);
    }
  }, [urlQuery]);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        // Fetch computational Therapeutic Target Priority Scores (TTPS)
        const [res, stats] = await Promise.all([
          ppiService.getTherapeuticTargets(50, 0.40, 0.35, 0.25, network),
          ppiService.getNetworkStats(network),
        ]);
        setData(res.data || []);
        setNetworkStats(stats.data || null);
      } catch (err) {
        console.error("Drug insights error:", err);
        setError("Failed to load computational therapeutic target priority scores.");
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [network]);

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
      {error && (
        <div role="alert" className="p-4 bg-red-50 border border-red-200 rounded-2xl text-sm text-red-700 flex items-center gap-2">
          <AlertCircle size={16} className="shrink-0" /> {error}
        </div>
      )}

      <header className="space-y-2">
        <h1 className="text-3xl font-black text-slate-900">Drug targets</h1>
        <p className="text-lg text-slate-700 max-w-3xl">
          Which proteins could be good targets for medicines? Proteins that sit at the centre of the protein network
          often matter most to the cell, so we rank them by how central they are and whether a medicine already targets them.
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <section aria-labelledby="ranking-heading" className="lg:col-span-2 glass-card p-8 bg-white space-y-4">
          <h2 id="ranking-heading" className="text-xl font-bold text-slate-900 flex items-center gap-2">
            <Award size={22} className="text-emerald-700" aria-hidden="true" /> How the ranking works
          </h2>
          <p className="text-slate-700">Each protein gets a <strong>priority score from 0 to 1</strong>, built from three clues:</p>
          <ul className="grid sm:grid-cols-3 gap-3 text-sm">
            <li className="p-3 rounded-xl bg-emerald-50 border border-emerald-200"><strong className="block text-emerald-900">40% · Connections</strong><span className="text-slate-700">How many partners it has in the network.</span></li>
            <li className="p-3 rounded-xl bg-teal-50 border border-teal-200"><strong className="block text-teal-900">35% · Bridge role</strong><span className="text-slate-700">How often it links other proteins together.</span></li>
            <li className="p-3 rounded-xl bg-purple-50 border border-purple-200"><strong className="block text-purple-900">25% · Drug record</strong><span className="text-slate-700">Whether the ChEMBL drug database lists it as a target.</span></li>
          </ul>
          {networkStats && (
            <p className="text-sm text-slate-700">
              {networkStats.network === 'predicted' ? (
                <>Network used: the <strong>{networkStats.known_edges?.toLocaleString()}</strong> known interactions plus <strong>{networkStats.predicted_edges?.toLocaleString()}</strong> new ones predicted by our model ({networkStats.num_nodes?.toLocaleString()} proteins).</>
              ) : (
                <>Network used: the <strong>{networkStats.num_edges?.toLocaleString()}</strong> known interactions only ({networkStats.num_nodes?.toLocaleString()} proteins).
                  {network === 'predicted' && !networkStats.predicted_network_available && ' (Predicted network not built yet: run scripts/build_predicted_network.py.)'}</>
              )}
            </p>
          )}
          <TechOnly>
            <div className="text-sm text-slate-800 bg-slate-50 border border-slate-200 rounded-xl p-3 space-y-1">
              <p><strong>Technical:</strong> Therapeutic Target Priority Score <code className="font-mono">TTPS = 0.40 × NormDegree + 0.35 × NormBetweenness + 0.25 × ChEMBL</code> (min-max normalized degree and betweenness centrality).</p>
              {networkStats?.network === 'predicted' && (
                <p>Predicted network = training positives + held-out pairs the ensemble scores above the validation threshold; {networkStats.predicted_edges_confirmed_by_string?.toLocaleString()} of the predicted edges are STRING interactions and {networkStats.predicted_edges_not_in_string?.toLocaleString()} are not.</p>
              )}
            </div>
          </TechOnly>
          <p className="text-sm text-amber-900 bg-amber-50 border border-amber-200 rounded-xl p-3 flex gap-2">
            <Info size={16} className="shrink-0 mt-0.5" aria-hidden="true" />
            A high score means "worth studying first". It does not prove a protein is a good drug target; that needs laboratory work.
          </p>
          <div className="flex gap-4 pt-1">
            <div className="p-4 rounded-2xl bg-emerald-50 border border-emerald-200 min-w-[140px]">
              <p className="text-3xl font-black text-emerald-900">{loading ? '…' : verifiedCount}</p>
              <p className="text-sm font-semibold text-emerald-900">have a drug record</p>
            </div>
            <div className="p-4 rounded-2xl bg-purple-50 border border-purple-200 min-w-[140px]">
              <p className="text-3xl font-black text-purple-900">{loading ? '…' : novelCount}</p>
              <p className="text-sm font-semibold text-purple-900">high score, no drug record yet</p>
            </div>
          </div>
        </section>

        <div className="glass-card p-6 flex flex-col justify-center">
           <div className="flex items-center justify-between mb-4 text-slate-800">
              <div className="flex items-center gap-2">
                 <TrendingUp size={20} className="text-emerald-700" aria-hidden="true" />
                 <h2 className="text-base font-bold">Top 8 scores</h2>
              </div>
              <span className="text-xs font-semibold text-slate-600">Average: {avgTtps}</span>
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
                                <p className="text-emerald-300">Score: {d.score}%</p>
                                <p className="text-slate-300 text-xs">{d.isVerified ? 'Has a drug record' : 'No drug record yet'}</p>
                             </div>
                          );
                       }} />
                    </BarChart>
                 </ResponsiveContainer>
              )}
           </div>
           <div className="mt-4 flex items-center justify-between text-xs text-slate-700 font-medium">
              <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-teal-600 rounded-sm inline-block" aria-hidden="true" /> Has drug record</span>
              <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-purple-600 rounded-sm inline-block" aria-hidden="true" /> No drug record yet</span>
           </div>
        </div>
      </div>

      {/* Filter & Search Bar */}
      <div className="glass-card p-4 flex flex-col md:flex-row gap-4 items-center justify-between">
        <div className="flex items-center gap-2 w-full md:w-96 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={18} aria-hidden="true" />
          <input
            type="text"
            aria-label="Search proteins"
            placeholder="Search by name, protein ID or UniProt ID"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-scientific-primary outline-none transition-all text-sm"
          />
        </div>

        <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-xl shrink-0" role="group" aria-label="Interaction network">
          {[
            { id: 'predicted', label: 'Include predicted interactions' },
            { id: 'known', label: 'Known interactions only' },
          ].map(n => (
            <button
              key={n.id}
              onClick={() => setNetwork(n.id)}
              aria-pressed={network === n.id}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all whitespace-nowrap ${network === n.id ? 'bg-white text-emerald-800 shadow-sm' : 'text-slate-700 hover:text-slate-900'}`}
            >
              {n.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2 overflow-x-auto w-full md:w-auto pb-2 md:pb-0">
           <Filter size={18} className="text-slate-500 mr-2" aria-hidden="true" />
           {[
             { id: 'all', label: 'All' },
             { id: 'verified', label: 'Has drug record' },
             { id: 'novel', label: 'No drug record yet' },
             { id: 'high_priority', label: 'High priority (score ≥ 0.40)' }
           ].map(f => (
             <button
               key={f.id}
               onClick={() => setFilter(f.id)}
               aria-pressed={filter === f.id}
               className={`px-4 py-2 rounded-xl text-xs font-bold transition-all whitespace-nowrap ${filter === f.id ? 'bg-emerald-700 text-white shadow-md' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'}`}
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
                <th scope="col" className="px-6 py-4 text-xs font-bold text-slate-700">Protein</th>
                <th scope="col" className="px-6 py-4 text-xs font-bold text-slate-700">Priority score</th>
                <th scope="col" className="px-6 py-4 text-xs font-bold text-slate-700">Connections</th>
                <th scope="col" className="px-6 py-4 text-xs font-bold text-slate-700">Bridge role</th>
                {network === 'predicted' && (
                  <th scope="col" className="px-6 py-4 text-xs font-bold text-slate-700" title="Interactions of this protein that our model predicted">New predicted partners</th>
                )}
                <th scope="col" className="px-6 py-4 text-xs font-bold text-slate-700">Drug record</th>
                <th scope="col" className="px-6 py-4 text-xs font-bold text-slate-700">Link</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {loading ? (
                <tr>
                  <td colSpan={network === 'predicted' ? 7 : 6} className="px-6 py-12 text-center">
                    <div className="flex flex-col items-center gap-3">
                       <Loader2 className="animate-spin text-scientific-primary" size={32} />
                       <p className="text-sm font-semibold text-slate-700">Calculating scores… this can take up to half a minute the first time.</p>
                    </div>
                  </td>
                </tr>
              ) : filteredData.map((item) => (
                <tr
                  key={item.protein_id} 
                  className="hover:bg-slate-50/50 transition-colors group"
                >
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                       <span className="w-6 h-6 rounded-full bg-slate-100 font-bold text-xs text-slate-600 flex items-center justify-center shrink-0">
                          #{item.rank}
                       </span>
                       <div className="flex flex-col">
                          {item.target_name && <span className="text-sm font-bold text-slate-900">{item.target_name}</span>}
                          <span className="text-xs font-mono text-slate-700">{item.protein_id}</span>
                          <span className="text-xs text-slate-600">UniProt: {item.uniprot_id || 'N/A'}</span>
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
                  {network === 'predicted' && (
                    <td className="px-6 py-4">
                      <div className="flex flex-col">
                        <span className="text-xs font-semibold text-slate-600 font-mono">{item.predicted_interactions ?? 0}</span>
                        {(item.novel_predicted_interactions ?? 0) > 0 && (
                          <span className="text-xs text-sky-800 font-medium">{item.novel_predicted_interactions} new (not in STRING)</span>
                        )}
                      </div>
                    </td>
                  )}
                  <td className="px-6 py-4">
                    {item.is_chembl_target ? (
                      <div className="flex flex-col gap-1">
                        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-green-50 text-green-700 text-[10px] font-bold border border-green-200 shrink-0 w-fit">
                          <CheckCircle2 size={11} aria-hidden="true" /> Has drug record
                        </span>
                        {item.chembl_id && (
                          <span className="text-[10px] text-slate-400 font-mono">{item.chembl_id}</span>
                        )}
                      </div>
                    ) : (
                      <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-purple-50 text-purple-700 text-[10px] font-bold border border-purple-200 shrink-0 w-fit">
                        <AlertCircle size={11} aria-hidden="true" /> No drug record yet
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
                        title="View on ChEMBL" aria-label={`View ${item.chembl_id} on ChEMBL`}
                      >
                         <ExternalLink size={16} />
                      </a>
                    ) : (
                      <span className="text-xs text-slate-300">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!loading && filteredData.length === 0 && (
          <div className="p-12 text-center space-y-4">
             <Search size={48} className="text-slate-200 mx-auto" />
             <p className="text-slate-700 font-semibold">No protein matches your search.</p>
             <button onClick={() => {setSearchQuery(''); setFilter('all');}} className="text-scientific-primary text-xs font-bold hover:underline">
                Clear search and filters
             </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default DrugInsights;
