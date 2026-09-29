import { useState } from 'react';
import { Eye, Info } from 'lucide-react';

// Attention-based explanation from the GAT comparison model (ESMGAT backend, `attention_explanation`).
// For each protein: the graph neighbours its GAT encoding attends to most, per layer (head-averaged).

const MAX_SPOKES = 8;
const SIZE = 280;
const CENTER = SIZE / 2;
const RADIUS = 100;

const short = (id) => (id && id.length > 10 ? `${id.slice(0, 10)}…` : id);

const StarDiagram = ({ title, info, entries, sharedIds, uniformWeight }) => {
  const self = entries.find(e => e.is_self);
  const spokes = entries.filter(e => !e.is_self).slice(0, MAX_SPOKES);
  const maxW = Math.max(...spokes.map(e => e.weight), 1e-6);

  return (
    <div className="flex-1 min-w-[260px] bg-slate-50 border border-slate-100 rounded-2xl p-4">
      <div className="flex items-baseline justify-between gap-2 mb-1">
        <h4 className="text-sm font-black text-slate-800">{title}</h4>
        <span className="text-xs font-mono text-slate-600">{info.num_neighbors} neighbours</span>
      </div>
      {info.surrogate && (
        <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-2 py-1 mb-2">
          Not in the training graph: shown via its most similar graph protein ({info.uniprot_id}).
        </p>
      )}
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="w-full max-w-[280px] mx-auto" role="img"
        aria-label={`Attention of ${info.uniprot_id} over its top ${spokes.length} neighbours`}>
        {spokes.map((e, i) => {
          const angle = (2 * Math.PI * i) / spokes.length - Math.PI / 2;
          const x = CENTER + RADIUS * Math.cos(angle);
          const y = CENTER + RADIUS * Math.sin(angle);
          const shared = sharedIds.has(e.protein_id);
          const tone = e.is_partner ? 'amber' : shared ? 'violet' : 'teal';
          const stroke = { amber: 'stroke-amber-500', violet: 'stroke-violet-500', teal: 'stroke-teal-500' }[tone];
          const fill = { amber: 'fill-amber-400', violet: 'fill-violet-400', teal: 'fill-teal-400' }[tone];
          return (
            <g key={e.protein_id}>
              <title>{`${e.uniprot_id} (${e.protein_id}): weight ${(e.weight * 100).toFixed(1)}%, ${e.lift.toFixed(1)}× an even split`}</title>
              <line x1={CENTER} y1={CENTER} x2={x} y2={y} className={stroke}
                strokeWidth={1 + 9 * (e.weight / maxW)} strokeOpacity={0.35 + 0.6 * (e.weight / maxW)} strokeLinecap="round" />
              <circle cx={x} cy={y} r={11} className={fill} />
              <text x={x} y={y + (y > CENTER ? 24 : -16)} textAnchor="middle" className="fill-slate-600 text-[11px] font-mono">
                {short(e.uniprot_id)}
              </text>
            </g>
          );
        })}
        <circle cx={CENTER} cy={CENTER} r={20} className="fill-slate-800" />
        <text x={CENTER} y={CENTER + 3} textAnchor="middle" className="fill-white text-[8px] font-bold">
          {short(info.uniprot_id)}
        </text>
      </svg>
      <table className="w-full text-[11px] mt-2">
        <thead>
          <tr className="text-slate-600 text-left">
            <th className="font-bold py-1">Neighbour</th>
            <th className="font-bold py-1 text-right">Weight</th>
            <th className="font-bold py-1 text-right" title="Weight relative to an even split over all neighbours">Lift</th>
          </tr>
        </thead>
        <tbody>
          {spokes.map(e => (
            <tr key={e.protein_id} className="border-t border-slate-100">
              <td className="py-1 font-mono text-slate-700">
                {e.uniprot_id}
                {e.is_partner && <span className="ml-1 text-[11px] font-bold text-amber-600">PARTNER</span>}
                {!e.is_partner && sharedIds.has(e.protein_id) && <span className="ml-1 text-[11px] font-bold text-violet-600">SHARED</span>}
              </td>
              <td className="py-1 text-right font-mono text-slate-700">{(e.weight * 100).toFixed(1)}%</td>
              <td className="py-1 text-right font-mono text-slate-500">{e.lift.toFixed(1)}×</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs text-slate-600 mt-2">
        Self-attention: {self ? `${(self.weight * 100).toFixed(1)}%` : 'n/a'} · even split: {(uniformWeight * 100).toFixed(1)}%
      </p>
    </div>
  );
};

const GATAttentionPanel = ({ attention }) => {
  const [layerIdx, setLayerIdx] = useState(attention.layers.length - 1);
  const layer = attention.layers[layerIdx];
  const sharedIds = new Set(layer.shared.map(s => s.protein_id));

  return (
    <div className="bg-white p-6 rounded-[2rem] border border-slate-100 shadow-sm space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-base font-black text-slate-800 flex items-center gap-2">
            <Eye size={18} className="text-violet-600" /> GAT Attention: where the graph model looked
          </h3>
          <p className="text-xs text-slate-500 mt-1 max-w-2xl">
            Graph neighbours each protein's GAT encoding attended to most (averaged over 4 attention heads).
            Thicker spokes mean more attention.
          </p>
        </div>
        <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-xl" role="group" aria-label="GAT layer">
          {attention.layers.map((l, i) => (
            <button key={l.layer} onClick={() => setLayerIdx(i)} aria-pressed={i === layerIdx}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer ${i === layerIdx ? 'bg-white text-violet-700 shadow-sm' : 'text-slate-700 hover:text-slate-900'}`}>
              Layer {l.layer}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap gap-4">
        <StarDiagram title="Protein 1" info={attention.protein1} entries={layer.protein1_top}
          sharedIds={sharedIds} uniformWeight={layer.protein1_uniform_weight} />
        <StarDiagram title="Protein 2" info={attention.protein2} entries={layer.protein2_top}
          sharedIds={sharedIds} uniformWeight={layer.protein2_uniform_weight} />
      </div>

      <div className="flex flex-wrap items-center gap-4 text-xs text-slate-500">
        <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-amber-400 inline-block" /> Interaction partner</span>
        <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-violet-400 inline-block" /> Neighbour of both proteins</span>
        <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-teal-400 inline-block" /> Other neighbour</span>
      </div>

      <div>
        <h4 className="text-sm font-bold text-slate-800 mb-2">
          Shared neighbours ({layer.num_shared})
        </h4>
        {layer.shared.length === 0 ? (
          <p className="text-xs text-slate-600">The two proteins have no graph neighbours in common.</p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {layer.shared.map(s => (
              <span key={s.protein_id} className="px-2.5 py-1 rounded-lg bg-violet-50 border border-violet-200 text-[11px] font-mono text-violet-800"
                title={s.protein_id}>
                {s.uniprot_id} · {(s.weight_protein1 * 100).toFixed(1)}% / {(s.weight_protein2 * 100).toFixed(1)}%
              </span>
            ))}
          </div>
        )}
      </div>

      <p className="text-xs text-slate-600 flex items-start gap-1.5 border-t border-slate-100 pt-3">
        <Info size={12} className="shrink-0 mt-0.5" />
        {attention.method} Attention shows what the graph encoder weighted, not a causal explanation of the final ensemble score (see SHAP above).
      </p>
    </div>
  );
};

export default GATAttentionPanel;
