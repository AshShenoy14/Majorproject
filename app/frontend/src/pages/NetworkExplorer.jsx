import React, { useEffect, useRef, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import cytoscape from 'cytoscape';
import {
  Share2, Maximize2, RotateCcw, Info, Search,
  ChevronRight, Activity, Database, Star, Filter,
  SlidersHorizontal, Navigation, Circle, GitMerge
} from 'lucide-react';
import { ppiService } from '../services/api';

// ─── BFS shortest path on cytoscape elements ───────────────────────────────
function bfs(cy, startId, endId) {
  if (!cy || !startId || !endId) return [];
  const visited = new Set();
  const queue = [[startId, [startId]]];
  while (queue.length) {
    const [nodeId, path] = queue.shift();
    if (nodeId === endId) return path;
    if (visited.has(nodeId)) continue;
    visited.add(nodeId);
    cy.getElementById(nodeId).neighborhood('node').forEach(n => {
      if (!visited.has(n.id())) {
        queue.push([n.id(), [...path, n.id()]]);
      }
    });
  }
  return [];
}

const NetworkExplorer = () => {
  const containerRef = useRef(null);
  const cyRef = useRef(null);

  const [selectedNode, setSelectedNode] = useState(null);
  const [metrics, setMetrics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Search & filter state
  const [searchQ, setSearchQ] = useState('');
  const [confidenceMin, setConfidenceMin] = useState(0);
  const [showHubsOnly, setShowHubsOnly] = useState(false);
  const [allNodes, setAllNodes] = useState([]);   // raw nodes from API
  const [allEdges, setAllEdges] = useState([]);   // raw edges from API

  // Shortest path state
  const [pathStart, setPathStart] = useState('');
  const [pathEnd, setPathEnd] = useState('');
  const [pathResult, setPathResult] = useState([]);

  // Build cytoscape elements from current filter state
  const rebuildGraph = useCallback((nodes, edges, minConf, hubsOnly, topHubs) => {
    if (!cyRef.current) return;

    const hubSet = new Set(topHubs.slice(0, 5).map(m => m.protein));

    // Filter nodes
    let filteredNodes = nodes;
    if (hubsOnly) filteredNodes = filteredNodes.filter(n => hubSet.has(n.id));

    // Filter edges by confidence
    const filteredEdges = edges.filter(e => (e.weight || 0.5) >= minConf);

    // Only keep nodes that appear in filtered edges
    const activeIds = new Set();
    filteredEdges.forEach(e => { activeIds.add(e.source); activeIds.add(e.target); });
    if (!hubsOnly) filteredNodes = filteredNodes.filter(n => activeIds.has(n.id));

    const elements = [
      ...filteredNodes.map(n => ({
        data: {
          id: n.id,
          label: n.label || n.id.slice(0, 8),
          isHub: hubSet.has(n.id),
          degree: cyRef.current.getElementById(n.id)?.degree?.() || 0,
        }
      })),
      ...filteredEdges.map(e => ({
        data: {
          source: e.source,
          target: e.target,
          weight: e.weight || 0.5,
        }
      }))
    ];

    cyRef.current.elements().remove();
    cyRef.current.add(elements);

    // Style hubs differently
    cyRef.current.style()
      .selector('node[?isHub]')
      .style({ 'background-color': '#7c3aed', 'width': '40px', 'height': '40px', 'border-color': '#7c3aed', 'border-width': '3px' })
      .update();

    cyRef.current.layout({ name: 'cose', animate: true, componentSpacing: 100 }).run();
  }, []);

  useEffect(() => {
    const initNetwork = async () => {
      setLoading(true);
      try {
        const [netRes, centralityRes] = await Promise.all([
          ppiService.getNetwork(80),
          ppiService.getCentrality(15)
        ]);

        const nodes = netRes.data.nodes;
        const edges = netRes.data.edges;
        setAllNodes(nodes);
        setAllEdges(edges);
        setMetrics(centralityRes.data);

        if (containerRef.current) {
          const topHubSet = new Set(centralityRes.data.slice(0, 5).map(m => m.protein));

          const elements = [
            ...nodes.map(n => ({
              data: { id: n.id, label: n.label || n.id.slice(0, 8), isHub: topHubSet.has(n.id) }
            })),
            ...edges.map(e => ({
              data: { source: e.source, target: e.target, weight: e.weight || 0.5 }
            }))
          ];

          cyRef.current = cytoscape({
            container: containerRef.current,
            elements,
            style: [
              {
                selector: 'node',
                style: {
                  'background-color': '#0D9488',
                  'label': 'data(label)',
                  'color': '#475569',
                  'font-size': '10px',
                  'text-valign': 'bottom',
                  'text-margin-y': '5px',
                  'width': '30px',
                  'height': '30px',
                  'border-width': '2px',
                  'border-color': '#ffffff',
                  'transition-property': 'background-color, width, height',
                  'transition-duration': '0.25s'
                }
              },
              {
                selector: 'node[?isHub]',
                style: {
                  'background-color': '#7c3aed',
                  'width': '42px',
                  'height': '42px',
                  'border-color': '#ddd6fe',
                  'border-width': '3px',
                  'font-weight': '700'
                }
              },
              {
                selector: 'edge',
                style: {
                  'width': 'mapData(weight, 0, 1, 1, 4)',
                  'line-color': '#CBD5E1',
                  'curve-style': 'bezier',
                  'opacity': 0.6
                }
              },
              {
                selector: 'node:selected',
                style: {
                  'background-color': '#f59e0b',
                  'width': '44px',
                  'height': '44px',
                  'border-color': '#f59e0b',
                  'border-width': '4px',
                  'border-opacity': 0.4
                }
              },
              {
                selector: '.path-highlight',
                style: {
                  'background-color': '#f43f5e',
                  'border-color': '#f43f5e',
                  'border-width': '3px',
                  'z-index': 999
                }
              },
              {
                selector: 'edge.path-highlight',
                style: {
                  'line-color': '#f43f5e',
                  'width': 3,
                  'opacity': 1
                }
              }
            ],
            layout: { name: 'cose', animate: true, componentSpacing: 100, nodeRepulsion: 400000, idealEdgeLength: 100 }
          });

          cyRef.current.on('tap', 'node', evt => setSelectedNode(evt.target.data()));
          cyRef.current.on('tap', evt => {
            if (evt.target === cyRef.current) setSelectedNode(null);
          });
        }
      } catch (err) {
        setError('Failed to load network data. Check backend connectivity.');
      } finally {
        setLoading(false);
      }
    };

    initNetwork();
    return () => { if (cyRef.current) cyRef.current.destroy(); };
  }, []);

  // ── Search ─────────────────────────────────────────────────────────────────
  const handleSearch = () => {
    if (!cyRef.current || !searchQ.trim()) return;
    const q = searchQ.trim().toLowerCase();
    cyRef.current.nodes().forEach(n => {
      const match = n.data('label')?.toLowerCase().includes(q) || n.data('id')?.toLowerCase().includes(q);
      n.style('background-color', match ? '#f59e0b' : (n.data('isHub') ? '#7c3aed' : '#0D9488'));
    });
  };

  // ── Filter by confidence ───────────────────────────────────────────────────
  const applyConfidenceFilter = (val) => {
    setConfidenceMin(val);
    if (!cyRef.current) return;
    cyRef.current.edges().forEach(e => {
      const w = e.data('weight') || 0.5;
      e.style('opacity', w >= val ? 0.6 : 0.05);
      e.style('line-color', w >= val ? '#0D9488' : '#e2e8f0');
    });
  };

  // ── Shortest path ──────────────────────────────────────────────────────────
  const findPath = () => {
    if (!cyRef.current || !pathStart || !pathEnd) return;
    setPathResult([]);
    cyRef.current.elements().removeClass('path-highlight');
    const path = bfs(cyRef.current, pathStart, pathEnd);
    setPathResult(path);
    if (path.length === 0) return;
    // Highlight nodes
    path.forEach(id => cyRef.current.getElementById(id).addClass('path-highlight'));
    // Highlight edges along path
    for (let i = 0; i < path.length - 1; i++) {
      cyRef.current.edges(`[source="${path[i]}"][target="${path[i+1]}"], [source="${path[i+1]}"][target="${path[i]}"]`)
        .addClass('path-highlight');
    }
  };

  const resetLayout = () => {
    if (!cyRef.current) return;
    cyRef.current.elements().removeClass('path-highlight');
    cyRef.current.elements().style('opacity', '');
    cyRef.current.elements().style('background-color', '');
    cyRef.current.elements().style('line-color', '');
    cyRef.current.layout({ name: 'cose', animate: true }).run();
    cyRef.current.fit();
    setPathResult([]);
  };

  return (
    <div className="h-[calc(100vh-160px)] flex gap-6 overflow-hidden">
      {/* Network Canvas */}
      <div className="flex-1 bg-slate-50/50 rounded-2xl border border-slate-100 relative overflow-hidden shadow-sm">
        {/* Canvas header badge */}
        <div className="absolute top-5 left-5 z-20 flex gap-3 flex-wrap">
          <div className="px-4 py-2 bg-white/90 backdrop-blur shadow-sm border border-slate-200 rounded-xl flex items-center gap-3">
            <div className="p-1.5 bg-teal-50 rounded-lg text-teal-600"><Share2 size={18} /></div>
            <div>
              <h2 className="text-sm font-bold text-slate-800">Topological Explorer</h2>
              <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Interactive PPI Schema</p>
            </div>
          </div>
          {/* Legend */}
          <div className="px-3 py-2 bg-white/90 backdrop-blur shadow-sm border border-slate-200 rounded-xl flex items-center gap-3 text-[10px] font-bold">
            <span className="flex items-center gap-1"><Circle size={8} className="fill-violet-500 text-violet-500" /> Hub</span>
            <span className="flex items-center gap-1"><Circle size={8} className="fill-teal-500 text-teal-500" /> Protein</span>
            <span className="flex items-center gap-1"><Circle size={8} className="fill-amber-400 text-amber-400" /> Selected</span>
            <span className="flex items-center gap-1"><Circle size={8} className="fill-rose-500 text-rose-500" /> Path</span>
          </div>
        </div>

        {/* Canvas controls */}
        <div className="absolute bottom-5 left-5 z-20 flex flex-col gap-2">
          <button onClick={resetLayout} title="Reset Layout"
            className="p-2.5 bg-white shadow-lg border border-slate-200 rounded-xl text-slate-600 hover:text-teal-600 transition-all">
            <RotateCcw size={18} />
          </button>
          <button onClick={() => cyRef.current?.fit()} title="Fit to View"
            className="p-2.5 bg-white shadow-lg border border-slate-200 rounded-xl text-slate-600 hover:text-teal-600 transition-all">
            <Maximize2 size={18} />
          </button>
        </div>

        {loading && (
          <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-white/60 backdrop-blur-sm">
            <div className="w-12 h-12 border-4 border-teal-200 border-t-teal-500 rounded-full animate-spin mb-4" />
            <p className="text-sm font-bold text-teal-600 uppercase tracking-widest">Loading Network...</p>
          </div>
        )}
        {error && (
          <div className="absolute inset-0 z-30 flex items-center justify-center">
            <p className="text-rose-500 text-sm font-bold">{error}</p>
          </div>
        )}

        <div ref={containerRef} className="w-full h-full rounded-2xl cursor-grab active:cursor-grabbing" />
      </div>

      {/* Control Panel */}
      <div className="w-80 space-y-4 flex flex-col overflow-y-auto pr-1 scrollbar-thin">
        {/* Search */}
        <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-4 space-y-3">
          <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-widest flex items-center gap-1.5"><Search size={11} /> Search Protein</h3>
          <div className="flex gap-2">
            <input value={searchQ} onChange={e => setSearchQ(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSearch()}
              placeholder="ID or name..."
              className="flex-1 px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl text-xs font-medium focus:ring-2 focus:ring-teal-200 outline-none"
            />
            <button onClick={handleSearch}
              className="px-3 py-2 bg-teal-500 text-white rounded-xl text-xs font-bold hover:bg-teal-600 transition-all">
              Go
            </button>
          </div>
        </div>

        {/* Filters */}
        <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-4 space-y-4">
          <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-widest flex items-center gap-1.5"><SlidersHorizontal size={11} /> Filters</h3>

          {/* Confidence slider */}
          <div>
            <div className="flex justify-between items-center mb-1.5">
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">Min Confidence</span>
              <span className="text-[10px] font-black text-teal-600">{confidenceMin.toFixed(1)}</span>
            </div>
            <input type="range" min="0" max="1" step="0.05" value={confidenceMin}
              onChange={e => applyConfidenceFilter(parseFloat(e.target.value))}
              className="w-full accent-teal-500" />
          </div>

          {/* Hub proteins toggle */}
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1.5">
              <Star size={11} /> Hub Proteins Only
            </span>
            <button onClick={() => setShowHubsOnly(v => !v)}
              className={`relative w-10 h-5 rounded-full transition-colors ${showHubsOnly ? 'bg-violet-500' : 'bg-slate-200'}`}>
              <div className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-all ${showHubsOnly ? 'left-5' : 'left-0.5'}`} />
            </button>
          </div>
        </div>

        {/* Shortest Path */}
        <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-4 space-y-3">
          <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-widest flex items-center gap-1.5"><Navigation size={11} /> Shortest Path</h3>
          <input value={pathStart} onChange={e => setPathStart(e.target.value)}
            placeholder="Start node ID"
            className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl text-xs font-medium focus:ring-2 focus:ring-rose-200 outline-none"
          />
          <input value={pathEnd} onChange={e => setPathEnd(e.target.value)}
            placeholder="End node ID"
            className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl text-xs font-medium focus:ring-2 focus:ring-rose-200 outline-none"
          />
          <button onClick={findPath}
            className="w-full py-2 bg-rose-500 hover:bg-rose-600 text-white rounded-xl text-xs font-bold transition-all flex items-center justify-center gap-1.5">
            <GitMerge size={13} /> Find Path
          </button>
          {pathResult.length > 0 && (
            <div className="mt-2 p-3 bg-rose-50 rounded-xl border border-rose-100">
              <p className="text-[10px] font-black text-rose-600 uppercase tracking-widest mb-2">Path ({pathResult.length} hops)</p>
              <div className="flex flex-wrap gap-1">
                {pathResult.map((id, i) => (
                  <React.Fragment key={i}>
                    <span className="text-[9px] bg-white border border-rose-200 px-1.5 py-0.5 rounded font-bold text-slate-700">{id.slice(0,10)}</span>
                    {i < pathResult.length - 1 && <span className="text-rose-400 text-xs">›</span>}
                  </React.Fragment>
                ))}
              </div>
            </div>
          )}
          {pathStart && pathEnd && pathResult.length === 0 && (
            <p className="text-[10px] text-slate-400 text-center">No path found between nodes.</p>
          )}
        </div>

        {/* Node Detail / Centrality */}
        <AnimatePresence mode="wait">
          {selectedNode ? (
            <motion.div key="details"
              initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 20 }}
              className="bg-white rounded-2xl border-l-4 border-teal-400 border border-slate-100 shadow-sm p-5">
              <div className="flex items-center gap-2 mb-3 text-teal-600">
                <Database size={14} />
                <span className="text-[10px] font-black uppercase tracking-widest">Protein Metadata</span>
              </div>
              <h3 className="text-lg font-black text-slate-800 mb-1">{selectedNode.label}</h3>
              <p className="text-[10px] text-slate-400 font-medium mb-4 font-mono">{selectedNode.id}</p>

              {/* Centrality metrics from API */}
              {(() => {
                const m = metrics.find(x => x.protein === selectedNode.id || x.protein === selectedNode.label);
                return m ? (
                  <div className="grid grid-cols-2 gap-2 mb-4">
                    {[
                      ['Degree', m.degree],
                      ['Betweenness', m.betweenness?.toFixed(3)],
                      ['Closeness', m.closeness?.toFixed(3)],
                      ['PageRank', m.pagerank?.toFixed(4)],
                    ].map(([k, v]) => v != null && (
                      <div key={k} className="p-2.5 bg-slate-50 rounded-xl border border-slate-100">
                        <p className="text-[9px] font-black text-slate-400 uppercase mb-0.5">{k}</p>
                        <p className="text-sm font-black text-slate-700">{v}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="grid grid-cols-2 gap-2 mb-4">
                    <div className="p-2.5 bg-slate-50 rounded-xl border border-slate-100">
                      <p className="text-[9px] font-black text-slate-400 uppercase mb-0.5">Degree</p>
                      <p className="text-sm font-black text-slate-700">
                        {cyRef.current?.getElementById(selectedNode.id)?.degree() ?? '—'}
                      </p>
                    </div>
                  </div>
                );
              })()}

              <button
                onClick={() => { setPathStart(selectedNode.id); }}
                className="w-full py-2 bg-teal-50 hover:bg-teal-100 border border-teal-200 rounded-xl text-xs font-bold text-teal-700 transition-all flex items-center justify-center gap-1.5">
                <Navigation size={12} /> Set as Path Start
              </button>
            </motion.div>
          ) : (
            <motion.div key="placeholder"
              initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              className="bg-white rounded-2xl border border-slate-100 shadow-sm p-8 flex flex-col items-center justify-center text-center">
              <Info size={28} className="text-slate-200 mb-3" />
              <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">Click a node to view properties</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Network Hubs */}
        <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-5 flex-1 min-h-0 flex flex-col">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-widest flex items-center gap-1.5"><Activity size={11} /> Hub Centrality</h3>
          </div>
          <div className="space-y-3 overflow-y-auto flex-1 pr-1 scrollbar-thin">
            {metrics.slice(0, 10).map((m, i) => (
              <div key={i} onClick={() => setSelectedNode({ id: m.protein, label: m.protein })}
                className="group p-3 hover:bg-slate-50 rounded-xl border border-transparent hover:border-slate-100 transition-all cursor-pointer">
                <div className="flex justify-between items-center mb-1.5">
                  <span className="text-xs font-black text-slate-700 truncate max-w-[140px]">{m.protein}</span>
                  {i < 3 && <Star size={12} className="fill-amber-400 text-amber-400 flex-shrink-0" />}
                </div>
                <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(m.degree / (metrics[0]?.degree || 1)) * 100}%` }}
                    transition={{ delay: i * 0.07, duration: 0.8 }}
                    className="h-full bg-gradient-to-r from-teal-400 to-cyan-500 rounded-full"
                  />
                </div>
                <div className="flex justify-between mt-1 text-[9px] font-bold text-slate-400 uppercase tracking-tight">
                  <span>Degree: {m.degree}</span>
                  <span>BC: {m.betweenness?.toFixed(3) || '—'}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default NetworkExplorer;
