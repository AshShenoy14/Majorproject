import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.analysis.network_analysis import NetworkAnalyzer
from src.data.target_manager import TargetManager
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT

def generate_target_table():
    print("Generating Hub Verification Table...")
    # Load interactions
    train_path = PROCESSED_DATA_DIR / "train.csv"
    if not train_path.exists():
        print("Training data not found.")
        return
        
    df = pd.read_csv(train_path)
    df_pos = df[df['label'] == 1]
    
    analyzer = NetworkAnalyzer()
    analyzer.build_from_dataframe(df_pos)
    
    # Get Top Hubs
    hubs = analyzer.identify_hubs(top_k=20)
    hub_ids = [h['id'] for h in hubs]
    
    # Visualize top 10
    plot_path = str(PROJECT_ROOT / "data" / "processed" / "plots" / "top_hubs_subgraph.png")
    analyzer.visualize_top_hubs(top_k=10, output_path=plot_path)
    
    # Check ChEMBL for these hubs
    tm = TargetManager()
    target_df = tm.get_targets(hub_ids)
    
    chembl_proteins = set(target_df['protein_id'].unique())
    
    results = []
    overlap_count = 0
    
    for hub in hubs:
        pid = hub['id']
        in_chembl = pid in chembl_proteins
        
        if in_chembl:
            is_known = "Yes"
            overlap_count += 1
        else:
            is_known = "No"
            
        results.append({
            "Protein": pid,
            "Degree": round(hub['score'], 4),
            "In ChEMBL?": "Yes" if in_chembl else "No",
            "Known Drug Target?": is_known
        })
        
    res_df = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("Drug Target Verification Table")
    print("="*50)
    print(res_df.to_markdown(index=False))
    
    print(f"\nOverlap Count: {overlap_count} out of {len(hubs)} hubs are known drug targets in ChEMBL.")
    res_df.to_csv(PROCESSED_DATA_DIR / "hub_targets_verification.csv", index=False)

if __name__ == "__main__":
    generate_target_table()
