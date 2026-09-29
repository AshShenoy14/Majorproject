import { useEffect, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart2,
  Database,
  AlertTriangle,
  Loader2,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  Layers,
  Activity,
  ArrowUpRight
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';
import { ppiService } from '../services/api';

const METRIC_COLUMNS = [
  { key: 'accuracy', label: 'Accuracy' },
  { key: 'precision', label: 'Precision' },
  { key: 'recall', label: 'Recall' },
  { key: 'f1', label: 'F1' },
  { key: 'roc_auc', label: 'ROC-AUC' },
  { key: 'pr_auc', label: 'PR-AUC' },
];

const PLAIN_MODEL_NAMES = {
  'Full Ensemble (XGBoost)': 'Final model (both opinions combined)',
  'Graph-Only (GraphSAGE)': 'Network only',
  'Sequence-Only (ESM-MLP)': 'Sequence only',
  'Random Forest Baseline': 'Simple baseline (random forest)',
};

const Benchmark = () => {
  const [evaluation, setEvaluation] = useState(null);
  const [benchmarks, setBenchmarks] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('models'); // 'models' | 'bootstrap' | 'cold_start' | 'external'

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [resFinal, resBench] = await Promise.all([
        ppiService.getFinalEvaluation().catch(() => ({ data: null })),
        ppiService.getAllBenchmarks().catch(() => ({ data: null }))
      ]);
      setEvaluation(resFinal.data);
      setBenchmarks(resBench.data || {});
    } catch (err) {
      console.error('Failed to load evaluation results:', err);
      setError(err?.response?.data?.detail || 'Could not reach backend for evaluation artifacts.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const rows = evaluation ? Object.entries(evaluation.models || {}) : [];
  const chartData = rows.map(([name, m]) => ({
    name,
    'ROC-AUC': +(m.roc_auc * 100).toFixed(2),
    'PR-AUC': +(m.pr_auc * 100).toFixed(2),
    F1: +(m.f1 * 100).toFixed(2),
  }));
  const counts = evaluation?.dataset_rows;

  const ens = evaluation?.models?.['Full Ensemble (XGBoost)'];
  const bs = benchmarks?.bootstrap_ci;
  const cs = benchmarks?.cold_start;
  const shs = benchmarks?.shs27k;
  const huri = benchmarks?.huri;

  return (
    <div className="max-w-7xl mx-auto space-y-8 py-2">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-gradient-to-br from-indigo-500 to-violet-600 rounded-2xl shadow-lg shadow-indigo-100">
            <BarChart2 size={28} className="text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-3xl font-black text-slate-900 tracking-tight">Model results</h1>
            </div>
            <p className="text-base text-slate-700">
              How often the model is right, how it compares with simpler models, and where it still struggles.
              Every number is loaded from the project's saved result files.
            </p>
          </div>
        </div>

        <button
          onClick={load}
          disabled={loading}
          className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-bold text-slate-600 bg-white border border-slate-200 rounded-xl hover:bg-slate-50 transition-all shadow-2xs self-start md:self-auto"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin text-indigo-600' : ''} />
          Refresh Metrics
        </button>
      </div>

      {/* Top Stat Banners */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white p-5 rounded-2xl border border-slate-100 shadow-sm relative overflow-hidden">
          <div className="absolute top-0 right-0 p-3 text-indigo-500/10">
            <TrendingUp size={64} />
          </div>
          <span className="text-xs font-black uppercase tracking-widest text-slate-600">Ensemble Accuracy</span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-3xl font-black text-slate-800 tracking-tight">{ens ? `${(ens.accuracy * 100).toFixed(2)}%` : '—'}</span>
            <span className="text-xs font-bold text-emerald-800 bg-emerald-50 px-1.5 py-0.5 rounded">Held-out test</span>
          </div>
          {bs?.accuracy && (
            <div className="mt-2 text-[11px] font-mono text-slate-500 bg-slate-50 px-2 py-1 rounded-lg border border-slate-100 inline-block">
              95% CI: [{(bs.accuracy.ci_lo * 100).toFixed(2)}%, {(bs.accuracy.ci_hi * 100).toFixed(2)}%]
            </div>
          )}
        </div>

        <div className="bg-white p-5 rounded-2xl border border-slate-100 shadow-sm relative overflow-hidden">
          <div className="absolute top-0 right-0 p-3 text-emerald-500/10">
            <ShieldCheck size={64} />
          </div>
          <span className="text-xs font-black uppercase tracking-widest text-slate-600">Ensemble ROC-AUC</span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-3xl font-black text-slate-800 tracking-tight">{ens ? ens.roc_auc.toFixed(4) : '—'}</span>
            <span className="text-xs font-bold text-indigo-600 bg-indigo-50 px-1.5 py-0.5 rounded">PR-AUC {ens ? ens.pr_auc.toFixed(4) : '—'}</span>
          </div>
          {bs?.roc_auc && (
            <div className="mt-2 text-[11px] font-mono text-slate-500 bg-slate-50 px-2 py-1 rounded-lg border border-slate-100 inline-block">
              95% CI: [{(bs.roc_auc.ci_lo).toFixed(4)}, {(bs.roc_auc.ci_hi).toFixed(4)}]
            </div>
          )}
        </div>

        <div className="bg-white p-5 rounded-2xl border border-slate-100 shadow-sm relative overflow-hidden">
          <div className="absolute top-0 right-0 p-3 text-cyan-500/10">
            <Activity size={64} />
          </div>
          <span className="text-xs font-black uppercase tracking-widest text-slate-600">Cold-Start Accuracy</span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-3xl font-black text-slate-800 tracking-tight">
              {cs ? `${(cs.cold_start_novel_protein_via_knn?.accuracy * 100).toFixed(1)}%` : '—'}
            </span>
            <span className="text-xs font-bold text-cyan-700 bg-cyan-50 px-1.5 py-0.5 rounded">KNN Inductive</span>
          </div>
          <p className="mt-2 text-[11px] text-slate-600 font-medium">Unseen proteins without prior edges</p>
        </div>

        <div className="bg-white p-5 rounded-2xl border border-slate-100 shadow-sm relative overflow-hidden">
          <div className="absolute top-0 right-0 p-3 text-amber-500/10">
            <Sparkles size={64} />
          </div>
          <span className="text-xs font-black uppercase tracking-widest text-slate-600">External Transfer (SHS27k)</span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-3xl font-black text-slate-800 tracking-tight">
              {shs ? shs.overall?.roc_auc.toFixed(4) : '—'}
            </span>
            <span className="text-xs font-bold text-amber-700 bg-amber-50 px-1.5 py-0.5 rounded">ROC-AUC</span>
          </div>
          <p className="mt-2 text-[11px] text-slate-600 font-medium">15,248 pairs scored on external snapshot</p>
        </div>
      </div>

      {loading && (
        <div className="flex items-center gap-3 text-slate-500 p-8 bg-white rounded-2xl border border-slate-100">
          <Loader2 className="animate-spin" size={20} />
          <span className="text-sm font-semibold">Loading evaluation artifacts…</span>
        </div>
      )}

      {!loading && error && (
        <div className="p-6 bg-rose-50 border border-rose-200 rounded-2xl flex items-start gap-4">
          <AlertTriangle className="text-rose-500 flex-shrink-0" size={22} />
          <div className="flex-1">
            <p className="text-sm font-black text-rose-700">Evaluation results unavailable</p>
            <p className="text-sm text-rose-600 mt-1">{error}</p>
            <button
              onClick={load}
              className="mt-3 inline-flex items-center gap-2 text-xs font-bold text-rose-700 bg-white border border-rose-200 rounded-lg px-3 py-1.5 hover:bg-rose-100"
            >
              <RefreshCw size={12} /> Retry
            </button>
          </div>
        </div>
      )}

      {/* Tabs Navigation */}
      <div className="flex items-center gap-2 border-b border-slate-200/80 pb-3">
        {[
          { id: 'models', label: '1. Which model is best?', icon: Layers },
          { id: 'bootstrap', label: '2. How certain are these numbers?', icon: ShieldCheck },
          { id: 'cold_start', label: '3. Proteins missing from the network', icon: Activity },
          { id: 'external', label: '4. Other datasets', icon: ArrowUpRight }
        ].map(tab => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              aria-pressed={isActive}
              className={`px-4 py-2.5 rounded-xl text-sm font-bold flex items-center gap-2 transition-all ${
                isActive 
                  ? 'bg-slate-900 text-white shadow-sm' 
                  : 'bg-white text-slate-600 hover:bg-slate-100/70 border border-slate-200/60'
              }`}
            >
              <Icon size={14} aria-hidden="true" className={isActive ? 'text-emerald-400' : 'text-slate-500'} />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Tab 1: In-Domain Models */}
      {activeTab === 'models' && evaluation && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
          <p className="text-base text-slate-700">
            All models were tested on the same <strong>20,172 protein pairs</strong> that none of them saw during training.
            Combining both opinions gives the best result on every measure.
          </p>
          <details className="bg-white rounded-2xl border border-slate-200 p-4">
            <summary className="font-bold text-slate-900 cursor-pointer">What do these measures mean?</summary>
            <dl className="mt-3 grid sm:grid-cols-2 gap-3 text-sm text-slate-700">
              <div><dt className="font-bold text-slate-900">Accuracy</dt><dd>Share of all predictions that were correct.</dd></div>
              <div><dt className="font-bold text-slate-900">Precision</dt><dd>Of the pairs predicted to interact, how many really do.</dd></div>
              <div><dt className="font-bold text-slate-900">Recall</dt><dd>Of the pairs that really interact, how many the model found.</dd></div>
              <div><dt className="font-bold text-slate-900">F1</dt><dd>A single score balancing precision and recall.</dd></div>
              <div><dt className="font-bold text-slate-900">ROC-AUC and PR-AUC</dt><dd>How well the model ranks interacting pairs above non-interacting ones (100% = perfect ranking).</dd></div>
              <div><dt className="font-bold text-slate-900">Threshold</dt><dd>The probability above which a pair counts as "interacting", chosen on separate validation data.</dd></div>
            </dl>
          </details>
          <div className="bg-white rounded-2xl border border-slate-100 shadow-sm overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs font-bold text-slate-700 border-b border-slate-200">
                  <th scope="col" className="px-5 py-4">Model</th>
                  {METRIC_COLUMNS.map(c => (
                    <th scope="col" key={c.key} className="px-4 py-4 text-right">{c.label}</th>
                  ))}
                  <th scope="col" className="px-4 py-4 text-right">Threshold</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(([name, m]) => (
                  <tr key={name} className="border-b border-slate-50 last:border-0 hover:bg-slate-50/50">
                    <td className="px-5 py-3 font-bold text-slate-800 flex items-center gap-2">
                      {name.includes('Ensemble') && <span className="w-2 h-2 rounded-full bg-indigo-600" aria-hidden="true"></span>}
                      <span>
                        <span className="block">{PLAIN_MODEL_NAMES[name] ?? name}</span>
                        {PLAIN_MODEL_NAMES[name] && <span className="block text-xs font-normal text-slate-600">{name}</span>}
                      </span>
                    </td>
                    {METRIC_COLUMNS.map(c => (
                      <td key={c.key} className={`px-4 py-3 text-right font-mono text-sm ${name.includes('Ensemble') ? 'font-black text-indigo-800' : 'text-slate-700'}`}>
                        {(m[c.key] * 100).toFixed(2)}%
                      </td>
                    ))}
                    <td className="px-4 py-3 text-right font-mono text-slate-700">{m.val_selected_threshold.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="bg-white rounded-2xl border border-slate-100 p-6 shadow-sm">
            <h2 className="text-base font-bold text-slate-900 mb-6">Side-by-side comparison (higher is better)</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} />
                <YAxis domain={[70, 100]} tick={{ fontSize: 11, fill: '#64748b' }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="ROC-AUC" fill="#6366f1" radius={[4, 4, 0, 0]} />
                <Bar dataKey="PR-AUC" fill="#14b8a6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="F1" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      )}

      {/* Tab 2: Bootstrap Confidence Intervals */}
      {activeTab === 'bootstrap' && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
          <div className="bg-white rounded-2xl border border-slate-100 p-6 shadow-sm">
            <div className="flex items-center gap-2 mb-4">
              <ShieldCheck className="text-indigo-600" size={20} />
              <h2 className="text-lg font-bold text-slate-900">How certain are these numbers?</h2>
            </div>
            <p className="text-sm text-slate-600 mb-6 leading-relaxed">
              A score measured on one test set could be a little lucky or unlucky. We re-sampled the 20,172 test results
              2,000 times (a "bootstrap") to see how much the scores move. The true score is very likely (95%) inside the range shown.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {[
                { name: 'Accuracy', mean: bs?.accuracy?.mean, lo: bs?.accuracy?.ci_lo, hi: bs?.accuracy?.ci_hi },
                { name: 'ROC-AUC', mean: bs?.roc_auc?.mean, lo: bs?.roc_auc?.ci_lo, hi: bs?.roc_auc?.ci_hi },
                { name: 'F1 Score', mean: bs?.f1?.mean, lo: bs?.f1?.ci_lo, hi: bs?.f1?.ci_hi }
              ].map(item => (
                <div key={item.name} className="p-4 rounded-xl bg-slate-50 border border-slate-100">
                  <span className="text-sm font-bold text-slate-700">{item.name}</span>
                  <div className="text-2xl font-black text-slate-800 mt-1">{item.mean != null ? `${(item.mean * 100).toFixed(2)}%` : '—'}</div>
                  <div className="mt-2 text-xs font-mono font-bold text-indigo-600 bg-white px-2.5 py-1 rounded border border-slate-200/60 inline-block">
                    Likely range: {item.lo != null ? `[${(item.lo * 100).toFixed(2)}%, ${(item.hi * 100).toFixed(2)}%]` : 'not available'}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      )}

      {/* Tab 3: Cold-Start Evaluation */}
      {activeTab === 'cold_start' && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
          <div className="bg-white rounded-2xl border border-slate-100 p-6 shadow-sm">
            <div className="flex items-center gap-2 mb-4">
              <Activity className="text-cyan-600" size={20} />
              <h2 className="text-lg font-bold text-slate-900">What about proteins missing from the network?</h2>
            </div>
            <p className="text-sm text-slate-600 mb-6 leading-relaxed">
              If a protein is not in the network, the app borrows the network position of the most similar proteins it does know.
              To test this, we removed 400 proteins from the network and predicted their pairs again, then compared with the normal result.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="p-5 rounded-2xl bg-cyan-50/60 border border-cyan-100">
                <span className="text-sm font-bold text-cyan-900">Protein removed from the network</span>
                <div className="text-3xl font-black text-slate-800 mt-2">
                  {cs ? `${(cs.cold_start_novel_protein_via_knn?.accuracy * 100).toFixed(2)}%` : '—'}
                </div>
                <div className="mt-2 text-xs text-slate-600 space-y-1">
                  <div>ROC-AUC: <span className="font-mono font-bold text-cyan-700">{cs ? cs.cold_start_novel_protein_via_knn?.roc_auc.toFixed(4) : '—'}</span></div>
                  <div>F1 Score: <span className="font-mono font-bold text-cyan-700">{cs ? cs.cold_start_novel_protein_via_knn?.f1.toFixed(4) : '—'}</span></div>
                  <div>Evaluated pairs: <span className="font-mono text-slate-700">{cs?.cold_start_novel_protein_via_knn?.n ?? '—'}</span></div>
                </div>
              </div>

              <div className="p-5 rounded-2xl bg-slate-50 border border-slate-200">
                <span className="text-sm font-bold text-slate-800">Same pairs, protein in the network</span>
                <div className="text-3xl font-black text-slate-800 mt-2">
                  {cs ? `${(cs.warm_baseline_same_pairs_full_graph?.accuracy * 100).toFixed(2)}%` : '—'}
                </div>
                <div className="mt-2 text-xs text-slate-600 space-y-1">
                  <div>ROC-AUC: <span className="font-mono font-bold text-slate-700">{cs ? cs.warm_baseline_same_pairs_full_graph?.roc_auc.toFixed(4) : '—'}</span></div>
                  <div>F1 Score: <span className="font-mono font-bold text-slate-700">{cs ? cs.warm_baseline_same_pairs_full_graph?.f1.toFixed(4) : '—'}</span></div>
                  <div>Same test pairs with full topological connectivity</div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Tab 4: External Benchmarks */}
      {activeTab === 'external' && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* SHS27k Card */}
            <div className="bg-white rounded-2xl border border-slate-100 p-6 shadow-sm flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-3">
                  <span className="px-2.5 py-1 rounded-lg text-xs font-bold bg-amber-50 text-amber-800 border border-amber-200">
                    Same Source, Alternate Curation
                  </span>
                  <span className="text-xs font-mono text-slate-600">{shs?.overall?.n != null ? `${shs.overall.n.toLocaleString()} pairs` : ''}</span>
                </div>
                <h3 className="text-lg font-bold text-slate-900">SHS27k (from the same STRING database)</h3>
                <p className="text-xs text-slate-500 mt-2 leading-relaxed">
                  A different selection of interactions taken from the same database the model learned from. It shows how the model does on similar, but not identical, data.
                </p>

                <div className="grid grid-cols-3 gap-2 mt-5">
                  <div className="p-3 bg-slate-50 rounded-xl text-center">
                    <span className="text-xs font-bold text-slate-700">Accuracy</span>
                    <div className="text-base font-black text-slate-800 mt-0.5">
                      {shs ? `${(shs.overall?.accuracy * 100).toFixed(1)}%` : '—'}
                    </div>
                  </div>
                  <div className="p-3 bg-slate-50 rounded-xl text-center">
                    <span className="text-xs font-bold text-slate-700">ROC-AUC</span>
                    <div className="text-base font-black text-amber-700 mt-0.5">
                      {shs ? shs.overall?.roc_auc.toFixed(4) : '—'}
                    </div>
                  </div>
                  <div className="p-3 bg-slate-50 rounded-xl text-center">
                    <span className="text-xs font-bold text-slate-700">F1</span>
                    <div className="text-base font-black text-slate-800 mt-0.5">
                      {shs ? shs.overall?.f1.toFixed(4) : '—'}
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-4 p-3 bg-slate-50 rounded-xl text-sm text-slate-800 border border-slate-200">
                Pairs with at least one protein the model never trained on: ROC-AUC{' '}
                <strong>{shs?.at_least_one_novel_protein_subset?.roc_auc != null ? shs.at_least_one_novel_protein_subset.roc_auc.toFixed(4) : '—'}</strong>
                {' '}(accuracy {shs?.at_least_one_novel_protein_subset?.accuracy != null ? `${(shs.at_least_one_novel_protein_subset.accuracy * 100).toFixed(1)}%` : '—'}).
              </div>
            </div>

            {/* HuRI Card */}
            <div className="bg-white rounded-2xl border border-slate-100 p-6 shadow-sm flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-3">
                  <span className="px-2.5 py-1 rounded-lg text-xs font-bold bg-violet-50 text-violet-800 border border-violet-200">
                    Independent Source (Yeast-2-Hybrid)
                  </span>
                  <span className="text-xs font-mono text-slate-600">{huri?.overall?.n != null ? `${huri.overall.n.toLocaleString()} pairs` : ''}</span>
                </div>
                <h3 className="text-lg font-bold text-slate-900">HuRI (independent laboratory screen)</h3>
                <p className="text-xs text-slate-500 mt-2 leading-relaxed">
                  Interactions measured in a separate, large laboratory experiment (yeast two-hybrid, Luck et al. 2020), not taken from STRING. The hardest test.
                </p>

                <div className="grid grid-cols-3 gap-2 mt-5">
                  <div className="p-3 bg-slate-50 rounded-xl text-center">
                    <span className="text-xs font-bold text-slate-700">Accuracy</span>
                    <div className="text-base font-black text-slate-800 mt-0.5">
                      {huri ? `${(huri.overall?.accuracy * 100).toFixed(1)}%` : '—'}
                    </div>
                  </div>
                  <div className="p-3 bg-slate-50 rounded-xl text-center">
                    <span className="text-xs font-bold text-slate-700">ROC-AUC</span>
                    <div className="text-base font-black text-violet-700 mt-0.5">
                      {huri ? huri.overall?.roc_auc.toFixed(4) : '—'}
                    </div>
                  </div>
                  <div className="p-3 bg-slate-50 rounded-xl text-center">
                    <span className="text-xs font-bold text-slate-700">Both proteins known</span>
                    <div className="text-base font-black text-slate-800 mt-0.5">
                      {huri ? `${(huri.both_proteins_seen_in_training_subset?.accuracy * 100).toFixed(1)}%` : '—'}
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-4 p-3 bg-amber-50 rounded-xl text-sm text-amber-900 border border-amber-200">
                Close to guessing: the model called only{' '}
                <strong>{huri?.overall?.predicted_positive_rate != null ? `${(huri.overall.predicted_positive_rate * 100).toFixed(1)}%` : '—'}</strong>
                {' '}of these pairs interacting, while half of them really do. It does not yet work well on interactions found by this different lab method.
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Dataset & Artifact Provenance */}
      {evaluation && (
        <div className="bg-white rounded-2xl border border-slate-100 p-6 shadow-sm space-y-3">
          <div className="flex items-center gap-2">
            <Database size={16} className="text-slate-600" />
            <h2 className="text-base font-bold text-slate-900">Where these numbers come from</h2>
          </div>
          {counts && (
            <p className="text-sm text-slate-600">
              Rows: {counts.train?.toLocaleString()} train / {counts.validation?.toLocaleString()} validation / {counts.test?.toLocaleString()} test
              {' '}({counts.total?.toLocaleString()} total). Test rows evaluated: {counts.test_evaluated?.toLocaleString()}.
            </p>
          )}
          {evaluation.split_design && <p className="text-sm text-slate-600">Split: {evaluation.split_design}</p>}
          {evaluation.protocol && <p className="text-sm text-slate-600">Protocol: {evaluation.protocol}</p>}
          {evaluation.checkpoints && (
            <p className="text-xs text-slate-600 font-mono break-all">
              Verified Checkpoints (sha256):{' '}
              {Object.entries(evaluation.checkpoints)
                .filter(([, c]) => c)
                .map(([k, c]) => `${k}=${c.sha256_16}`)
                .join(' · ')}
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default Benchmark;
