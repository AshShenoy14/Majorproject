import pandas as pd
from typing import List
from src.utils.paths import RAW_DATA_DIR

def prioritize_drug_targets(centrality_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """
    Maps top-k central proteins to ChEMBL drug targets.
    Args:
        centrality_df: DataFrame with 'Protein' column, sorted by centrality.
        top_k: Number of top proteins to query.
    """
    top_proteins = centrality_df.head(top_k)["Protein"].tolist()
    print(f"Prioritizing targets for top {top_k} proteins: {top_proteins}")
    
    # Load ChEMBL data if available locally
    chembl_path = RAW_DATA_DIR / "chembl_targets.csv"
    if chembl_path.exists():
        chembl_df = pd.read_csv(chembl_path)
    else:
        # If not local, we might use the collect_drug_targets script dynamically
        # But here we assume analysis runs on collected data.
        print("ChEMBL data not found locally. Please run data collection.")
        return pd.DataFrame()
        
    # Merge
    # chembl_df should have 'uniprot_id'
    prioritized = chembl_df[chembl_df["uniprot_id"].isin(top_proteins)]
    
    # Add centrality info
    prioritized = prioritized.merge(centrality_df[["Protein", "Degree", "Betweenness"]], 
                                    left_on="uniprot_id", right_on="Protein", how="left")
    
    return prioritized.sort_values("Degree", ascending=False)
