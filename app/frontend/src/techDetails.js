import { createContext, useContext } from 'react';

// "Show technical details" switch: pages lead with plain English and reveal model names,
// formulas and raw scores only when this is on. Provided by <TechDetailsProvider>.
export const TechDetailsContext = createContext({ showTech: false, setShowTech: () => {} });

export const useTechDetails = () => useContext(TechDetailsContext);
