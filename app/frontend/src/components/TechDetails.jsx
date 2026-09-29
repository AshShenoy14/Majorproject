import { useState } from 'react';
import { TechDetailsContext, useTechDetails } from '../techDetails';

const STORAGE_KEY = 'transgraph_show_tech';

const readStored = () => {
  try {
    return localStorage.getItem(STORAGE_KEY) === 'true';
  } catch {
    return false;
  }
};

export const TechDetailsProvider = ({ children }) => {
  const [showTech, setShowTechState] = useState(readStored);
  const setShowTech = (value) => {
    setShowTechState(value);
    try {
      localStorage.setItem(STORAGE_KEY, String(value));
    } catch {
      // storage unavailable (private window): the switch still works for this visit
    }
  };
  return (
    <TechDetailsContext.Provider value={{ showTech, setShowTech }}>
      {children}
    </TechDetailsContext.Provider>
  );
};

// Renders its children only when "Show technical details" is on.
export const TechOnly = ({ children }) => {
  const { showTech } = useTechDetails();
  return showTech ? children : null;
};

// Renders its children only when technical details are off (the plain-language version).
export const PlainOnly = ({ children }) => {
  const { showTech } = useTechDetails();
  return showTech ? null : children;
};

// The switch itself, used in the top bar.
export const TechDetailsSwitch = () => {
  const { showTech, setShowTech } = useTechDetails();
  return (
    <button
      type="button"
      role="switch"
      aria-checked={showTech}
      onClick={() => setShowTech(!showTech)}
      className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold text-slate-700 hover:bg-slate-100 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-emerald-600"
      title="Show the model names, formulas and raw scores behind each result"
    >
      <span className={`relative inline-block w-8 h-4 rounded-full transition-colors ${showTech ? 'bg-emerald-600' : 'bg-slate-300'}`} aria-hidden="true">
        <span className={`absolute top-0.5 w-3 h-3 bg-white rounded-full shadow transition-all ${showTech ? 'left-4' : 'left-0.5'}`} />
      </span>
      <span>Technical details</span>
    </button>
  );
};
