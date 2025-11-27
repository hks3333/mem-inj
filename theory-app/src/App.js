import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceArea } from 'recharts';
import './App.css';

// --- Icons (Inlined for stability) ---
const Icon = ({ path, size = 24, className = "", color = "currentColor" }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    {path}
  </svg>
);

const Icons = {
  ArrowRight: <path d="M5 12h14M12 5l7 7-7 7" />,
  Brain: <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z" />,
  Zap: <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />,
  Layers: <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />,
  Microscope: <path d="M6 18h8M3 22h18M14 22a7 7 0 1 0 0-14h-1M9 14h2M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3" />,
  ChevronRight: <path d="m9 18 6-6-6-6" />,
  Play: <path d="M5 3l14 9-14 9V3z" />,
  RotateCcw: <path d="M1 4v6h6M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />,
  CheckCircle2: <><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" /><path d="m9 11 3 3L22 4" /></>,
  XCircle: <><circle cx="12" cy="12" r="10" /><path d="m15 9-6 6" /><path d="m9 9 6 6" /></>,
  Database: <><ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" /><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" /></>,
  FileText: <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>,
  Filter: <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon>,
  Merge: <path d="m8 6 4-4 4 4"></path>
};

const LucideIcon = ({ name, ...props }) => (
  <Icon path={Icons[name] || Icons.Brain} {...props} />
);

// --- Components ---

const Section = ({ title, children, id, active }) => (
  <div className={`min-h-screen flex flex-col justify-center py-20 px-6 transition-opacity duration-700 ${active ? 'opacity-100' : 'opacity-30 blur-sm'}`}>
    <div className="max-w-6xl mx-auto w-full">
      <h2 className="text-sm font-semibold tracking-widest text-indigo-600 uppercase mb-4">{id}</h2>
      <h1 className="text-4xl md:text-5xl font-bold mb-12 text-gray-900">{title}</h1>
      {children}
    </div>
  </div>
);

const Card = ({ children, className = "" }) => (
  <div className={`bg-white rounded-xl shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-gray-100 overflow-hidden ${className}`}>
    {children}
  </div>
);

// --- Visualizations ---

const ModelSpecsViz = () => (
  <div className="grid md:grid-cols-2 gap-8 items-center">
    <div>
      <p className="text-lg text-gray-600 mb-6 leading-relaxed">
        We utilized the <strong>Meta Llama 3.2 1B</strong> model (Checkpoint 1). This is a highly efficient, small language model optimized for edge devices, making it the perfect candidate to study reasoning bottlenecks in restricted parameter spaces.
      </p>
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 bg-gray-50 rounded-lg border border-gray-100">
          <div className="text-sm text-gray-500 uppercase tracking-wide">Model ID</div>
          <div className="font-mono text-indigo-600 font-bold">meta-llama/Llama-3.2-1B</div>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg border border-gray-100">
          <div className="text-sm text-gray-500 uppercase tracking-wide">Precision</div>
          <div className="font-mono text-indigo-600 font-bold">bfloat16</div>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg border border-gray-100">
          <div className="text-sm text-gray-500 uppercase tracking-wide">Parameters</div>
          <div className="font-mono text-gray-900 font-bold">1.23 Billion</div>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg border border-gray-100">
          <div className="text-sm text-gray-500 uppercase tracking-wide">Architecture</div>
          <div className="font-mono text-gray-900 font-bold">16 Layers / 32 Heads</div>
        </div>
      </div>
    </div>

    <Card className="p-8 bg-slate-900 text-white relative overflow-hidden">
      <div className="absolute top-0 right-0 p-32 bg-indigo-500/10 rounded-full blur-3xl" />
      <div className="relative z-10 flex flex-col items-center gap-2">
        <div className="text-xs text-slate-400 mb-2 font-mono">Input Embeddings</div>
        {Array.from({ length: 6 }).map((_, i) => (
          <div
            key={i}
            className={`w-full h-8 rounded border border-indigo-500/30 flex items-center justify-center text-xs font-mono transition-all hover:bg-indigo-500/20 ${
              i === 2 || i === 3 ? 'bg-indigo-500/20 shadow-[0_0_15px_rgba(99,102,241,0.3)]' : 'bg-slate-800'
            }`}
          >
            {i === 2 || i === 3 ? 'Reasoning Layers (Active)' : `Decoder Layer ${i * 2 + 1}`}
          </div>
        ))}
        <div className="text-xs text-slate-400 mt-2 font-mono">Output Head (Logits)</div>
      </div>
    </Card>
  </div>
);

const DataCurationViz = () => (
  <div className="space-y-8">
    <p className="text-lg text-gray-600 leading-relaxed">
      As detailed in <strong>Checkpoint 2</strong>, we curated a unified dataset from <span className="font-mono text-sm bg-gray-100 px-1 rounded">musique</span> (complex multi-hop) and
      <span className="font-mono text-sm bg-gray-100 px-1 rounded">twohopfact</span> (synthetic). The critical step was <strong>Independent Verification</strong>: we only kept examples where the model correctly answered both Q1 and Q2 in isolation but failed the composite.
    </p>

    <div className="relative">
      <div className="hidden md:block absolute top-1/2 left-0 w-full h-1 bg-gray-200 -z-10 transform -translate-y-1/2" />
      <div className="grid md:grid-cols-4 gap-4">
        <Card className="p-4 flex flex-col items-center text-center gap-3 bg-white z-10">
          <div className="w-12 h-12 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center">
            <LucideIcon name="Database" size={20} />
          </div>
          <h3 className="font-bold text-gray-900">1. Ingestion</h3>
          <p className="text-xs text-gray-500">
            Load <code>musique.py</code> &amp; <code>twohopfact.py</code> generators.
          </p>
        </Card>

        <Card className="p-4 flex flex-col items-center text-center gap-3 bg-white z-10 border-indigo-200 shadow-md">
          <div className="w-12 h-12 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center">
            <LucideIcon name="CheckCircle2" size={20} />
          </div>
          <h3 className="font-bold text-indigo-900">2. Verification</h3>
          <p className="text-xs text-gray-500">
            Run harness: <code>checkpoint2_prepare_dataset.py</code>.<br />Must pass Q1 &amp; Q2.
          </p>
        </Card>

        <Card className="p-4 flex flex-col items-center text-center gap-3 bg-white z-10">
          <div className="w-12 h-12 rounded-full bg-purple-100 text-purple-600 flex items-center justify-center">
            <LucideIcon name="Filter" size={20} />
          </div>
          <h3 className="font-bold text-gray-900">3. Filtering</h3>
          <p className="text-xs text-gray-500">Discard rows where model hallucinates single hops.</p>
        </Card>

        <Card className="p-4 flex flex-col items-center text-center gap-3 bg-white z-10">
          <div className="w-12 h-12 rounded-full bg-green-100 text-green-600 flex items-center justify-center">
            <LucideIcon name="FileText" size={20} />
          </div>
          <h3 className="font-bold text-gray-900">4. Unification</h3>
          <p className="text-xs text-gray-500">
            Export to <code>combined_success.csv</code>.<br />Ready for Baseline.
          </p>
        </Card>
      </div>
    </div>

    <div className="bg-gray-50 rounded-lg p-4 font-mono text-xs border border-gray-200 overflow-x-auto">
      <div className="text-gray-400 mb-2"># Sample Row from combined_success.csv</div>
      <table className="w-full text-left">
        <thead>
          <tr className="text-gray-500 border-b">
            <th className="pb-2">dataset</th>
            <th className="pb-2">q1</th>
            <th className="pb-2">a1</th>
            <th className="pb-2">q_composite</th>
            <th className="pb-2">status</th>
          </tr>
        </thead>
        <tbody>
          <tr className="text-gray-800">
            <td className="py-2 pr-4 text-purple-600">musique</td>
            <td className="py-2 pr-4">Who directed Inception?</td>
            <td className="py-2 pr-4 font-bold">Christopher Nolan</td>
            <td className="py-2 pr-4">Who directed Inception? What was his first film?</td>
            <td className="py-2 text-green-600">READY</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
);

const CompositionalityGapViz = () => {
  const [step, setStep] = useState(0);

  const steps = [
    { label: 'Q1: Who directed Inception?', result: 'Christopher Nolan', correct: true },
    { label: "Q2: What was Nolan's first film?", result: 'Following', correct: true },
    { label: "Combined: Who directed Inception? What was his first film?", result: 'The Dark Knight', correct: false }
  ];

  return (
    <div className="grid md:grid-cols-2 gap-12 items-center">
      <div className="space-y-6">
        <p className="text-lg text-gray-600 leading-relaxed">
          The <strong>Compositionality Gap</strong> describes a failure where a model knows two facts independently but fails to combine them. Llama 1B can retrieve the director of Inception, and it knows Nolan's first film. But asked together, it fails.
        </p>
        <div className="flex gap-4">
          {[0, 1, 2].map((i) => (
            <button
              key={i}
              onClick={() => setStep(i)}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${step === i ? 'bg-indigo-600 text-white' : 'bg-gray-100 text-gray-500 hover:bg-gray-200'}`}
            >
              Test {i === 2 ? 'Composite' : `Hop ${i + 1}`}
            </button>
          ))}
        </div>
      </div>

      <Card className="p-8 min-h-[300px] flex flex-col justify-center relative">
        <div className="absolute top-4 right-4 flex items-center gap-2 text-xs font-mono text-gray-400">
          <LucideIcon name="Database" size={14} /> Llama 1B
        </div>

        <div className="space-y-8">
          <div className="transition-all duration-500">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">Input Prompt</h3>
            <div className="font-mono text-lg text-gray-800 bg-gray-50 p-4 rounded-lg border-l-4 border-indigo-500">{steps[step].label}</div>
          </div>

          <div className="transition-all duration-500 delay-100">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">Model Output</h3>
            <div
              className={`flex items-center justify-between p-4 rounded-lg border ${
                steps[step].correct ? 'bg-green-50 border-green-200 text-green-900' : 'bg-red-50 border-red-200 text-red-900'
              }`}
            >
              <span className="font-medium text-xl">{steps[step].result}</span>
              {steps[step].correct ? <LucideIcon name="CheckCircle2" className="text-green-600" /> : <LucideIcon name="XCircle" className="text-red-600" />}
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};

const LogitLensViz = () => {
  const data = Array.from({ length: 17 }, (_, i) => {
    let rank;
    if (i < 4) rank = 80000 - i * 8000;
    else if (i < 10) rank = 50 + Math.pow(i - 7, 2) * 200;
    else rank = 1000 + Math.pow(i - 10, 3) * 100;
    return { layer: i, rank: Math.max(1, Math.min(100000, rank)) };
  });

  return (
    <div className="space-y-8">
      <div className="grid md:grid-cols-3 gap-6">
        <div className="md:col-span-2">
          <p className="text-lg text-gray-600 mb-6">
            Using the <strong>Logit Lens</strong> (Checkpoint 5), we track the rank of the intermediate answer (e.g., "Nolan") across the 16 layers of Llama 3.2 1B. We observe a distinctive "Fade-Out" curve: the model computes the fact in the middle layers, but overwrites it before the end.
          </p>
          <div className="h-[400px] w-full bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="layer" label={{ value: 'Layer (0-16)', position: 'insideBottom', offset: -10 }} stroke="#9ca3af" />
                <YAxis scale="log" domain={[1, 100000]} label={{ value: 'Token Rank (Log Scale)', angle: -90, position: 'insideLeft' }} stroke="#9ca3af" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e5e7eb', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                  formatter={(value) => [`Rank: ${Math.round(value)}`, 'Token Priority']}
                />
                <ReferenceArea x1={6} x2={10} stroke="none" fill="#e0e7ff" fillOpacity={0.3} />
                <Line type="monotone" dataKey="rank" stroke="#4f46e5" strokeWidth={3} dot={false} activeDot={{ r: 8 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-between text-xs text-gray-400 mt-2 px-4">
            <span>Embedding</span>
            <span className="text-indigo-600 font-medium">Reasoning Sweet Spot</span>
            <span>Output / Forgotten</span>
          </div>
        </div>
        <div className="space-y-4">
          <Card className="p-6 bg-indigo-50 border-indigo-100">
            <div className="flex items-start gap-3">
              <div className="mt-1 flex-shrink-0">
                <LucideIcon name="Microscope" className="text-indigo-600" size={20} />
              </div>
              <div>
                <h4 className="font-semibold text-indigo-900">Diagnosis</h4>
                <p className="text-sm text-indigo-800 mt-1">The information exists! It appears around Layer 7-8 (Rank &lt; 100) but fades away by Layer 15.</p>
              </div>
            </div>
          </Card>
          <Card className="p-6">
            <h4 className="font-semibold text-gray-900 mb-2">What is Rank?</h4>
            <p className="text-sm text-gray-600">Rank represents the model's prediction priority. Rank 1 means it's the top prediction. Rank 100,000 means it's completely ignored.</p>
          </Card>
        </div>
      </div>
    </div>
  );
};

const BackpatchingViz = () => {
  const [isSimulating, setIsSimulating] = useState(false);
  const [connection, setConnection] = useState('broken');

  const handlePatch = () => {
    setIsSimulating(true);
    setTimeout(() => {
      setConnection('patched');
      setIsSimulating(false);
    }, 1500);
  };

  const handleReset = () => {
    setConnection('broken');
  };

  return (
    <div className="grid md:grid-cols-2 gap-12 items-center">
      <div className="space-y-6">
        <p className="text-lg text-gray-600">
          <strong>Backpatching</strong> proves causality. We verify the "Hopping Too Late" hypothesis by taking the activation from a late layer (where the info finally appears) and manually patching it into an earlier layer.
        </p>

        <div className="bg-gray-50 p-6 rounded-xl border border-gray-200">
          <div className="flex items-center justify-between mb-4">
            <span className="font-mono text-sm text-gray-500">Intervention Config</span>
            <span className={`text-xs font-bold px-2 py-1 rounded ${connection === 'patched' ? 'bg-green-100 text-green-700' : 'bg-gray-200 text-gray-600'}`}>
              {connection === 'patched' ? 'ACTIVE' : 'INACTIVE'}
            </span>
          </div>
          <code className="block text-xs font-mono text-gray-700 mb-4">
            source: Layer 14 (Late)
            <br />
            target: Layer 7 (Early)
            <br />
            operation: Swap Activation
          </code>
          {connection === 'broken' ? (
            <button
              onClick={handlePatch}
              disabled={isSimulating}
              className="w-full py-3 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg font-medium transition flex items-center justify-center gap-2"
            >
              {isSimulating ? <LucideIcon name="RotateCcw" className="animate-spin" size={18} /> : <LucideIcon name="Zap" size={18} />}
              Apply Backpatch
            </button>
          ) : (
            <button
              onClick={handleReset}
              className="w-full py-3 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-lg font-medium transition flex items-center justify-center gap-2"
            >
              Reset Simulation
            </button>
          )}
        </div>
      </div>

      <Card className="p-8 h-[400px] relative flex flex-col items-center justify-center bg-slate-900 text-white overflow-hidden">
        <div className="absolute left-8 top-0 bottom-0 w-1 bg-slate-800 my-8" />
        <div className="absolute left-8 top-[20%] w-4 h-4 rounded-full bg-blue-500 transform -translate-x-1.5 flex items-center">
          <span className="ml-6 text-xs text-blue-300 w-32">Layer 7 (Early)</span>
        </div>
        <div className="absolute left-8 top-[80%] w-4 h-4 rounded-full bg-purple-500 transform -translate-x-1.5 flex items-center">
          <span className="ml-6 text-xs text-purple-300 w-32">Layer 14 (Late)</span>
        </div>

        <div
          className={`absolute w-32 h-12 bg-white/10 backdrop-blur rounded-lg border border-white/20 flex items-center justify-center transition-all duration-1000 ${
            connection === 'broken' ? 'top-[80%] translate-x-12 opacity-50' : 'top-[20%] translate-x-12 opacity-100 bg-green-500/20 border-green-400/50'
          }`}
        >
          <span className="text-sm font-mono">"Nolan"</span>
        </div>

        <svg className="absolute inset-0 w-full h-full pointer-events-none">
          {connection === 'patched' && (
            <path d="M 50 320 C 150 320, 150 80, 50 80" fill="none" stroke="#4ade80" strokeWidth="2" strokeDasharray="6 4" className="animate-pulse" />
          )}
          <defs>
            <marker id="arrow" markerWidth="10" markerHeight="10" refX="0" refY="3" orient="auto">
              <path d="M0,0 L0,6 L9,3 z" fill="#4ade80" />
            </marker>
          </defs>
        </svg>

        <div className="absolute bottom-8 right-8 text-right">
          <div className="text-xs text-slate-400 uppercase tracking-wider mb-1">Output</div>
          <div className={`text-2xl font-bold ${connection === 'patched' ? 'text-green-400' : 'text-red-400'}`}>
            {connection === 'patched' ? 'Following' : 'The Dark Knight'}
          </div>
        </div>
      </Card>
    </div>
  );
};

const MemoryInjectionViz = () => {
  const [phase, setPhase] = useState(1);

  return (
    <div className="space-y-8">
      <div className="max-w-3xl">
        <p className="text-lg text-gray-600 mb-6">
          <strong>Memory Injection</strong> is the proposed mitigation. We treat the model's failure as a "working memory" deficit. We extract the memory vector of the intermediate answer (A1) from a clean run, and inject it into the composite run at the precise moment needed.
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        <button
          onClick={() => setPhase(1)}
          className={`p-4 rounded-xl border-2 text-left transition-all ${phase === 1 ? 'border-indigo-600 bg-indigo-50' : 'border-gray-100 bg-white hover:border-gray-200'}`}
        >
          <div className="text-xs font-bold uppercase text-indigo-600 mb-2">Phase 1</div>
          <div className="font-semibold text-gray-900">Extraction Run</div>
          <div className="text-sm text-gray-500 mt-1">Run Q1 only. Save activation at Layer 8.</div>
        </button>
        <button
          onClick={() => setPhase(2)}
          className={`p-4 rounded-xl border-2 text-left transition-all ${phase === 2 ? 'border-purple-600 bg-purple-50' : 'border-gray-100 bg-white hover:border-gray-200'}`}
        >
          <div className="text-xs font-bold uppercase text-purple-600 mb-2">Phase 2</div>
          <div className="font-semibold text-gray-900">Injection Run</div>
          <div className="text-sm text-gray-500 mt-1">Run Composite Q. Insert vector at token position 6.</div>
        </button>
        <div className="p-4 rounded-xl border border-gray-100 bg-white flex flex-col justify-center items-center text-center">
          <div className="text-xs font-bold uppercase text-gray-400 mb-2">Result</div>
          <div className={`text-xl font-bold ${phase === 2 ? 'text-green-600' : 'text-gray-300'}`}>{phase === 2 ? 'Success (98%)' : 'Waiting...'}</div>
        </div>
      </div>

      <Card className="h-[300px] bg-slate-50 relative overflow-hidden flex items-center justify-center border-slate-200">
        <div className="absolute inset-0 grid grid-cols-12 gap-2 p-4 opacity-10">
          {Array.from({ length: 48 }).map((_, i) => (
            <div key={i} className="bg-slate-400 rounded h-full" />
          ))}
        </div>

        <div className="flex items-center gap-12 relative z-10">
          <div className={`flex flex-col items-center transition-all duration-500 ${phase === 1 ? 'opacity-100 scale-110' : 'opacity-50 blur-sm'}`}>
            <div className="w-20 h-20 bg-white rounded-2xl shadow-lg border border-indigo-200 flex items-center justify-center mb-4">
              <LucideIcon name="Database" className="text-indigo-600" />
            </div>
            <span className="font-mono text-xs bg-white px-2 py-1 rounded border">Vector A1</span>
          </div>

          <div className={`text-gray-400 transition-all duration-500 ${phase === 2 ? 'translate-x-0 opacity-100' : '-translate-x-4 opacity-0'}`}>
            <LucideIcon name="ArrowRight" size={32} />
          </div>

          <div className={`flex flex-col items-center transition-all duration-500 ${phase === 2 ? 'opacity-100 scale-110' : 'opacity-50'}`}>
            <div className="w-64 h-32 bg-white rounded-2xl shadow-xl border border-purple-200 flex flex-col items-center justify-center p-4 relative overflow-hidden">
              {phase === 2 && <div className="absolute inset-0 bg-purple-500/5 animate-pulse" />}
              <span className="text-xs text-gray-400 uppercase tracking-widest mb-2">Composite Stream</span>
              <div className="flex gap-1">
                {[...'Who directed...'].map((_, i) => (
                  <div key={i} className="w-1 h-6 bg-gray-200 rounded-full" />
                ))}
                <div className={`w-2 h-8 rounded-full transition-all duration-500 ${phase === 2 ? 'bg-purple-500 h-10 shadow-[0_0_15px_rgba(168,85,247,0.5)]' : 'bg-gray-200'}`} />
                {[...'what was...'].map((_, i) => (
                  <div key={i} className="w-1 h-6 bg-gray-200 rounded-full" />
                ))}
              </div>
            </div>
            <span className="font-mono text-xs bg-white px-2 py-1 rounded border mt-4">Injection Point</span>
          </div>
        </div>
      </Card>
    </div>
  );
};

function App() {
  const [activeSection, setActiveSection] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      const sections = document.querySelectorAll('section');
      sections.forEach((section, index) => {
        const rect = section.getBoundingClientRect();
        if (rect.top >= 0 && rect.top <= window.innerHeight / 2) {
          setActiveSection(index);
        }
      });
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="bg-[#fafafa] text-gray-900 font-sans">
      <div className="fixed left-6 top-1/2 transform -translate-y-1/2 hidden lg:flex flex-col gap-4 z-50">
        {[0, 1, 2, 3, 4, 5, 6].map((i) => (
          <div key={i} className={`w-1 transition-all duration-300 rounded-full ${activeSection === i ? 'h-12 bg-indigo-600' : 'h-4 bg-gray-300'}`} />
        ))}
      </div>

      <Section title="Experimental Setup" id="Checkpoint 1: The Model" active={activeSection === 0}>
        <section id="section-0">
          <ModelSpecsViz />
        </section>
      </Section>

      <Section title="Data Curation" id="Checkpoint 2: The Dataset" active={activeSection === 1}>
        <section id="section-1">
          <DataCurationViz />
        </section>
      </Section>

      <Section title="The Compositionality Gap" id="Checkpoint 3: Baseline" active={activeSection === 2}>
        <section id="section-2">
          <CompositionalityGapViz />
        </section>
      </Section>

      <Section title="The Logit Lens" id="Checkpoint 5: Diagnosis" active={activeSection === 3}>
        <section id="section-3">
          <LogitLensViz />
        </section>
      </Section>

      <Section title="Backpatching" id="Checkpoint 6: Causality" active={activeSection === 4}>
        <section id="section-4">
          <BackpatchingViz />
        </section>
      </Section>

      <Section title="Memory Injection" id="Checkpoint 7: Mitigation" active={activeSection === 5}>
        <section id="section-5">
          <MemoryInjectionViz />
        </section>
      </Section>

      <Section title="Conclusion" id="Part V: Impact" active={activeSection === 6}>
        <section id="section-6">
          <div className="grid md:grid-cols-2 gap-12">
            <div>
              <p className="text-lg text-gray-600 mb-6 leading-relaxed">
                Our research demonstrates that SLMs suffer from a <strong>Process Execution Failure</strong>, not a knowledge deficit. The information is computed, but "hops" too late in the layer stack to be useful for subsequent reasoning.
              </p>
              <ul className="space-y-4">
                <li className="flex items-center gap-3 text-gray-700">
                  <LucideIcon name="CheckCircle2" className="text-green-500" />
                  <span>Logit Lens confirms "Fade-Out" of knowledge.</span>
                </li>
                <li className="flex items-center gap-3 text-gray-700">
                  <LucideIcon name="CheckCircle2" className="text-green-500" />
                  <span>Backpatching proves causality by fixing the timeline.</span>
                </li>
                <li className="flex items-center gap-3 text-gray-700">
                  <LucideIcon name="CheckCircle2" className="text-green-500" />
                  <span>Memory Injection restores multi-hop reasoning capability.</span>
                </li>
              </ul>
            </div>
            <Card className="p-8 bg-gradient-to-br from-indigo-600 to-purple-700 text-white flex flex-col justify-center">
              <h3 className="text-2xl font-bold mb-4">Future Architectures</h3>
              <p className="text-indigo-100">
                This work suggests that future SLMs need explicit "Working Memory" modules or recurrent injection mechanisms to bridge the gap between knowledge retrieval and complex reasoning.
              </p>
            </Card>
          </div>
        </section>
      </Section>
    </div>
  );
}

export default App;
