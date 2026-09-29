import { Link } from 'react-router-dom';
import { FileText, Users, Network, Scale, Lightbulb, Pill, ArrowRight, CheckCircle2, AlertTriangle } from 'lucide-react';
import { TechOnly } from '../components/TechDetails';
import { useTechDetails } from '../techDetails';

const STEPS = [
  {
    icon: FileText,
    title: 'Read each protein',
    plain: 'Every protein is a chain of building blocks (amino acids). An AI model that has read millions of proteins turns each chain into a numeric "fingerprint" that captures what the protein is like.',
    analogy: 'Like reading two people\'s profiles.',
    tech: 'ESM-2 protein language model (esm2_t30_150M_UR50D) → one 640-dimensional embedding per protein.',
  },
  {
    icon: Users,
    title: 'Opinion 1: compare the two proteins',
    plain: 'A first model looks only at the two fingerprints and gives its opinion: how likely are these two to interact?',
    analogy: 'Do their profiles suggest they would get along?',
    tech: 'Sequence branch: symmetric pair features (2,560-d) → residual MLP → p_seq.',
  },
  {
    icon: Network,
    title: 'Opinion 2: look at their neighbours',
    plain: 'A second model looks at the network of 80,685 known interactions and at which proteins each one already works with.',
    analogy: 'Do they have many friends in common?',
    tech: 'Graph branch: 2-layer GraphSAGE over the STRING network (640 + 3 network features per protein) → p_graph, Platt-calibrated. A standard GAT (graph attention) version was also built for comparison.',
  },
  {
    icon: Scale,
    title: 'Final decision',
    plain: 'A "judge" model combines the two opinions. It has learned when to trust each one, especially when they disagree, and gives the final probability.',
    analogy: 'A referee who knows which opinion is usually right.',
    tech: 'XGBoost meta-learner on 7 meta-features (both probabilities, their confidence, difference and agreement), trained on 5-fold out-of-fold predictions to avoid leakage.',
  },
  {
    icon: Lightbulb,
    title: 'Explain and use the result',
    plain: 'The app shows which opinion drove each decision. Across all proteins, it ranks the most connected ones that already have medicines, as candidate drug targets.',
    analogy: 'Explaining the referee\'s decision, and finding the most influential people.',
    tech: 'SHAP attributions for every prediction; GAT attention weights for the GAT model; target score = 0.40·degree + 0.35·betweenness + 0.25·ChEMBL record, on the predicted network.',
  },
];

const HowItWorks = () => {
  const { showTech } = useTechDetails();
  return (
    <div className="max-w-4xl mx-auto space-y-10 pb-12">
      <header className="space-y-4">
        <h1 className="text-4xl font-black text-slate-900">How it works</h1>
        <p className="text-lg text-slate-700 leading-relaxed">
          Proteins do almost everything in our bodies, and most of them work in pairs or teams. Testing every
          possible pair in a laboratory is slow and expensive, so TransGraph <strong>predicts</strong> which pairs
          are likely to interact. Scientists can then test the most promising ones first.
        </p>
        <p className="text-slate-700">
          It works like deciding whether two people are likely to be friends: you read their profiles, you look at
          their mutual friends, and you combine both clues.
          {!showTech && ' Turn on "Technical details" at the top to see the model behind each step.'}
        </p>
      </header>

      <ol className="space-y-5">
        {STEPS.map((step, i) => {
          const Icon = step.icon;
          return (
            <li key={step.title} className="glass-card bg-white p-6 flex gap-5">
              <div className="shrink-0 flex flex-col items-center">
                <span className="w-10 h-10 rounded-full bg-emerald-700 text-white font-black flex items-center justify-center" aria-hidden="true">{i + 1}</span>
              </div>
              <div className="space-y-2">
                <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
                  <Icon size={20} className="text-emerald-700" aria-hidden="true" />
                  <span><span className="sr-only">Step {i + 1}: </span>{step.title}</span>
                </h2>
                <p className="text-slate-700 leading-relaxed">{step.plain}</p>
                <p className="text-sm text-slate-600 italic">{step.analogy}</p>
                <TechOnly>
                  <p className="text-sm text-slate-800 bg-slate-50 border border-slate-200 rounded-lg p-3">
                    <strong>Technical:</strong> {step.tech}
                  </p>
                </TechOnly>
              </div>
            </li>
          );
        })}
      </ol>

      <section aria-labelledby="results-heading" className="glass-card bg-white p-6 space-y-4">
        <h2 id="results-heading" className="text-2xl font-bold text-slate-900">What we found</h2>
        <ul className="space-y-3 text-slate-700">
          <li className="flex gap-3"><CheckCircle2 className="text-emerald-700 shrink-0 mt-0.5" size={20} aria-hidden="true" />
            <span>On 20,172 protein pairs the model never saw while training, <strong>93.10% of predictions were correct</strong>.</span></li>
          <li className="flex gap-3"><CheckCircle2 className="text-emerald-700 shrink-0 mt-0.5" size={20} aria-hidden="true" />
            <span>Combining both opinions beats either one alone (88.42% for the protein-only model, 91.62% for the network-only model).</span></li>
          <li className="flex gap-3"><CheckCircle2 className="text-emerald-700 shrink-0 mt-0.5" size={20} aria-hidden="true" />
            <span>We also built the graph-attention (GAT) version from our proposal. It reached 89.79%, so we chose the
              simpler network model (GraphSAGE). The GAT spread its attention almost evenly across neighbours,
              which is consistent with it not doing better.</span></li>
          <li className="flex gap-3"><AlertTriangle className="text-amber-600 shrink-0 mt-0.5" size={20} aria-hidden="true" />
            <span><strong>Limits:</strong> the test proteins also appear in the training data, and on an independent
              dataset measured with a different lab method (HuRI) the model is close to guessing. It works best on
              data similar to what it learned from.</span></li>
        </ul>
      </section>

      <section aria-labelledby="try-heading" className="space-y-4">
        <h2 id="try-heading" className="text-2xl font-bold text-slate-900">Try it</h2>
        <div className="grid sm:grid-cols-3 gap-4">
          <Link to="/predict?p1=ENSP00000269305&p2=ENSP00000258149" className="glass-card bg-white p-5 hover:border-emerald-400 transition-all">
            <p className="font-bold text-slate-900">Predict a famous pair</p>
            <p className="text-sm text-slate-700 mt-1">TP53 and MDM2, a key cancer-related interaction.</p>
            <span className="text-sm font-bold text-emerald-700 flex items-center gap-1 mt-3">Open <ArrowRight size={15} aria-hidden="true" /></span>
          </Link>
          <Link to="/drug-targets" className="glass-card bg-white p-5 hover:border-emerald-400 transition-all">
            <p className="font-bold text-slate-900 flex items-center gap-2"><Pill size={16} aria-hidden="true" className="text-emerald-700" /> See drug targets</p>
            <p className="text-sm text-slate-700 mt-1">The most central proteins that already have medicines.</p>
            <span className="text-sm font-bold text-emerald-700 flex items-center gap-1 mt-3">Open <ArrowRight size={15} aria-hidden="true" /></span>
          </Link>
          <Link to="/benchmark" className="glass-card bg-white p-5 hover:border-emerald-400 transition-all">
            <p className="font-bold text-slate-900">See all the results</p>
            <p className="text-sm text-slate-700 mt-1">Every model compared, with honest limitations.</p>
            <span className="text-sm font-bold text-emerald-700 flex items-center gap-1 mt-3">Open <ArrowRight size={15} aria-hidden="true" /></span>
          </Link>
        </div>
      </section>
    </div>
  );
};

export default HowItWorks;
