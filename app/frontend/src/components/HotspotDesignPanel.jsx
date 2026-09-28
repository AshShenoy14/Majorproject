import { useEffect, useRef, useState } from 'react';
import { Crosshair, FlaskConical, Loader2, AlertTriangle, ArrowRight, Info } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { ppiService } from '../services/api';

// Hotspot map (occlusion scan) and mutation design, both run as backend jobs because they need
// dozens of ESM-2 passes (about 1-2 minutes on CPU). The panel polls the job and shows its progress.

const POLL_MS = 1500;

const useJob = () => {
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const timer = useRef(null);

  const stop = () => { if (timer.current) { clearInterval(timer.current); timer.current = null; } };
  useEffect(() => stop, []);

  const start = async (startFn) => {
    stop();
    setError(null);
    setJob(null);
    try {
      const { data } = await startFn();
      setJob(data);
      timer.current = setInterval(async () => {
        try {
          const res = await ppiService.getJob(data.job_id);
          setJob(res.data);
          if (res.data.status === 'done' || res.data.status === 'error') stop();
        } catch (err) {
          stop();
          setError(err.response?.data?.detail || 'Lost contact with the analysis job.');
        }
      }, POLL_MS);
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not start the analysis.');
    }
  };

  const running = job && (job.status === 'queued' || job.status === 'running');
  return { job, error: error || (job?.status === 'error' ? job.error : null), running, start };
};

const Progress = ({ job }) => {
  const { done = 0, total = 0, stage } = job.progress || {};
  const pct = total ? Math.round((done / total) * 100) : 0;
  return (
    <div className="space-y-2" role="status" aria-live="polite">
      <div className="flex justify-between text-xs text-slate-500">
        <span className="flex items-center gap-2"><Loader2 size={14} className="animate-spin text-scientific-accent" />{stage || 'Queued'}</span>
        <span className="font-mono">{total ? `${done}/${total} ESM passes` : ''} · {job.elapsed_seconds}s</span>
      </div>
      <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
        <div className="h-full bg-scientific-accent transition-all duration-500" style={{ width: `${pct}%` }} />
      </div>
      <p className="text-[11px] text-slate-400">Runs in the background on the server; you can keep using other pages.</p>
    </div>
  );
};

const HotspotChart = ({ label, impact }) => {
  const data = impact.map((v, i) => ({ pos: i + 1, impact: v }));
  // Long proteins are scanned with ~30 windows, so one window's impact covers a run of residues: report runs.
  const runs = [];
  impact.forEach((v, i) => {
    const last = runs[runs.length - 1];
    if (last && last.value === v && last.end === i) last.end = i + 1;
    else runs.push({ value: v, start: i + 1, end: i + 1 });
  });
  const top = runs.filter(r => r.value > 0).sort((a, b) => b.value - a.value).slice(0, 3);
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-baseline">
        <h5 className="text-xs font-bold text-slate-600 uppercase tracking-wider">{label}</h5>
        <span className="text-[11px] text-slate-400">max drop {(Math.max(0, ...impact) * 100).toFixed(2)} pts</span>
      </div>
      <div className="h-40">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
            <XAxis dataKey="pos" tick={{ fontSize: 10 }} axisLine={false} tickLine={false} minTickGap={30} />
            <YAxis tick={{ fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={v => (v * 100).toFixed(1)} width={36} />
            <Tooltip formatter={v => [`${(v * 100).toFixed(2)} pts`, 'Score drop when masked']} labelFormatter={l => `Residue ${l}`} />
            <Area type="stepAfter" dataKey="impact" stroke="#7C3AED" fill="#7C3AED" fillOpacity={0.15} strokeWidth={2} isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      {top.length > 0 && (
        <p className="text-[11px] text-slate-500">
          Top regions: {top.map(t => `${t.start === t.end ? t.start : `${t.start}–${t.end}`} (${(t.value * 100).toFixed(2)} pts)`).join(', ')}
        </p>
      )}
    </div>
  );
};

const HotspotDesignPanel = ({ protein1, protein2, onLoadMutations }) => {
  const hotspots = useJob();
  const design = useJob();
  const [mode, setMode] = useState('disrupt');
  const hs = hotspots.job?.status === 'done' ? hotspots.job.result : null;
  const suggestions = design.job?.status === 'done' ? design.job.result : null;

  return (
    <div className="glass-card p-8 space-y-6">
      <div>
        <h3 className="text-lg font-bold text-slate-800">Hotspot Map &amp; Mutation Design</h3>
        <p className="text-xs text-slate-500 mt-1 max-w-3xl">
          Hotspots: each 5-residue window is masked and the protein re-embedded with ESM-2; the drop in the sequence
          model's interaction score marks residues the prediction depends on. Design: the three strongest hotspots of
          Protein A are mutated to all 19 alternatives and scored exactly.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <section className="space-y-4" aria-label="Hotspot map">
          <button type="button" disabled={hotspots.running || !protein1 || !protein2}
            onClick={() => hotspots.start(() => ppiService.startHotspotJob(protein1, protein2))}
            className="w-full py-3 bg-slate-800 text-white rounded-xl font-bold text-sm flex items-center justify-center gap-2 hover:bg-slate-700 transition-all disabled:opacity-50">
            <Crosshair size={18} /> Map interaction hotspots
          </button>
          {hotspots.running && <Progress job={hotspots.job} />}
          {hotspots.error && <p className="text-xs text-red-600 flex items-center gap-2"><AlertTriangle size={14} />{hotspots.error}</p>}
          {hs && (
            <div className="space-y-5">
              <p className="text-xs text-slate-500">Sequence-model score for this pair: <b>{(hs.base_score * 100).toFixed(1)}%</b></p>
              {hs.protein1 && <HotspotChart label={`Protein A (${hs.protein1.id})`} impact={hs.protein1.residue_impact} />}
              {hs.protein2 && <HotspotChart label={`Protein B (${hs.protein2.id})`} impact={hs.protein2.residue_impact} />}
            </div>
          )}
        </section>

        <section className="space-y-4" aria-label="Mutation design">
          <div className="flex gap-2">
            <select value={mode} onChange={e => setMode(e.target.value)} disabled={design.running}
              aria-label="Design goal"
              className="px-3 py-3 bg-white border border-slate-200 rounded-xl text-sm font-bold outline-none">
              <option value="disrupt">Disrupt binding</option>
              <option value="stabilize">Stabilize binding</option>
            </select>
            <button type="button" disabled={design.running || !protein1 || !protein2}
              onClick={() => design.start(() => ppiService.startOptimizeJob(protein1, protein2, mode))}
              className="flex-1 py-3 bg-scientific-accent text-white rounded-xl font-bold text-sm flex items-center justify-center gap-2 hover:bg-purple-700 transition-all disabled:opacity-50">
              <FlaskConical size={18} /> Suggest mutations
            </button>
          </div>
          {design.running && <Progress job={design.job} />}
          {design.error && <p className="text-xs text-red-600 flex items-center gap-2"><AlertTriangle size={14} />{design.error}</p>}
          {suggestions && (
            <div className="space-y-3">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-[10px] uppercase tracking-wider text-slate-400">
                    <th className="py-1">Protein A mutation</th><th className="py-1 text-right">Score</th><th className="py-1 text-right">Δ</th>
                  </tr>
                </thead>
                <tbody>
                  {suggestions.suggestions.map(s => (
                    <tr key={`${s.pos}${s.mut}`} className="border-t border-slate-100">
                      <td className="py-1.5 font-mono">{s.orig}{s.pos}{s.mut}</td>
                      <td className="py-1.5 text-right font-mono">{(s.score * 100).toFixed(1)}%</td>
                      <td className={`py-1.5 text-right font-mono ${s.delta < 0 ? 'text-rose-600' : 'text-emerald-600'}`}>
                        {s.delta > 0 ? '+' : ''}{(s.delta * 100).toFixed(2)} pts
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <button type="button"
                onClick={() => onLoadMutations(suggestions.suggestions.map(s => ({ protein: 1, pos: s.pos, orig: s.orig, mut: s.mut })))}
                className="text-xs font-bold text-scientific-accent flex items-center gap-1 hover:underline">
                Load into the mutation batch above <ArrowRight size={14} />
              </button>
              <p className="text-[11px] text-slate-400 flex gap-1.5"><Info size={12} className="shrink-0 mt-0.5" />
                Single substitutions usually move this model's score by well under one point; treat these as ranked candidates, not predicted binders.
              </p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
};

export default HotspotDesignPanel;
