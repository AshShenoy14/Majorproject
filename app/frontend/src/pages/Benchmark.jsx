import React, { useEffect, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { BarChart2, Database, AlertTriangle, Loader2, RefreshCw } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
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

const Benchmark = () => {
  const [evaluation, setEvaluation] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await ppiService.getFinalEvaluation();
      setEvaluation(res.data);
    } catch (err) {
      console.error('Failed to load final evaluation:', err);
      setEvaluation(null);
      setError(err?.response?.data?.detail || 'Could not reach the backend to load the final evaluation results.');
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

  return (
    <div className="max-w-7xl mx-auto space-y-8 py-2">
      <div className="flex items-center gap-4">
        <div className="p-3 bg-gradient-to-br from-indigo-500 to-violet-600 rounded-2xl shadow-lg shadow-indigo-100">
          <BarChart2 size={28} className="text-white" />
        </div>
        <div>
          <h1 className="text-3xl font-black text-slate-800 tracking-tight">Model Benchmarks</h1>
          <p className="text-sm text-slate-400 font-medium">Final held-out test evaluation (read from the backend evaluation artifact)</p>
        </div>
      </div>

      {loading && (
        <div className="flex items-center gap-3 text-slate-500 p-8 bg-white rounded-2xl border border-slate-100">
          <Loader2 className="animate-spin" size={20} />
          <span className="text-sm font-semibold">Loading final evaluation results…</span>
        </div>
      )}

      {!loading && error && (
        <div className="p-6 bg-rose-50 border border-rose-200 rounded-2xl flex items-start gap-4">
          <AlertTriangle className="text-rose-500 flex-shrink-0" size={22} />
          <div className="flex-1">
            <p className="text-sm font-black text-rose-700">Final evaluation results unavailable</p>
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

      {!loading && !error && evaluation && (
        <>
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-2xl border border-slate-100 shadow-sm overflow-x-auto"
          >
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[10px] font-black uppercase tracking-widest text-slate-400 border-b border-slate-100">
                  <th className="px-5 py-4">Model</th>
                  {METRIC_COLUMNS.map(c => (
                    <th key={c.key} className="px-4 py-4 text-right">{c.label}</th>
                  ))}
                  <th className="px-4 py-4 text-right">Val threshold</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(([name, m]) => (
                  <tr key={name} className="border-b border-slate-50 last:border-0">
                    <td className="px-5 py-3 font-bold text-slate-700">{name}</td>
                    {METRIC_COLUMNS.map(c => (
                      <td key={c.key} className="px-4 py-3 text-right font-mono text-slate-600">{m[c.key].toFixed(4)}</td>
                    ))}
                    <td className="px-4 py-3 text-right font-mono text-slate-400">{m.val_selected_threshold.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </motion.div>

          <div className="bg-white rounded-2xl border border-slate-100 p-6 shadow-sm">
            <h3 className="text-sm font-black text-slate-700 uppercase tracking-widest mb-6">ROC-AUC, PR-AUC and F1 by model (%)</h3>
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

          <div className="bg-white rounded-2xl border border-slate-100 p-6 shadow-sm space-y-3">
            <div className="flex items-center gap-2">
              <Database size={16} className="text-slate-400" />
              <h3 className="text-sm font-black text-slate-700 uppercase tracking-widest">Evaluation details</h3>
            </div>
            {counts && (
              <p className="text-sm text-slate-600">
                Rows: {counts.train?.toLocaleString()} train / {counts.validation?.toLocaleString()} validation / {counts.test?.toLocaleString()} test
                {' '}({counts.total?.toLocaleString()} total). Test rows evaluated: {counts.test_evaluated?.toLocaleString()}
                {' '}(filtered: {counts.test_rows_filtered}).
              </p>
            )}
            {evaluation.split_design && <p className="text-sm text-slate-600">Split: {evaluation.split_design}</p>}
            {evaluation.protocol && <p className="text-sm text-slate-600">Protocol: {evaluation.protocol}</p>}
            {evaluation.timestamp_utc && (
              <p className="text-xs text-slate-400">Generated: {new Date(evaluation.timestamp_utc).toLocaleString()}</p>
            )}
            {evaluation.checkpoints && (
              <p className="text-xs text-slate-400 font-mono break-all">
                Checkpoints (sha256 prefix):{' '}
                {Object.entries(evaluation.checkpoints)
                  .filter(([, c]) => c)
                  .map(([k, c]) => `${k}=${c.sha256_16}`)
                  .join(' · ')}
              </p>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default Benchmark;
