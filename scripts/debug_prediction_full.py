import torch
import sys
import os
import joblib
import numpy as np
import pandas as pd

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT

def debug_prediction(p1, p2):
    print(f"Debugging prediction for: {p1} - {p2}")
    
    # 1. Check Graph Mapping
    map_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    if map_path.exists():
        mapping = torch.load(map_path, weights_only=False)
        print(f"Graph Mapping Loaded. Total Nodes: {len(mapping)}")
        
        in_p1 = p1 in mapping
        in_p2 = p2 in mapping
        
        print(f"Protein 1 ({p1}) in Graph: {in_p1}")
        print(f"Protein 2 ({p2}) in Graph: {in_p2}")
        
        if in_p1: print(f"  Index 1: {mapping[p1]}")
        if in_p2: print(f"  Index 2: {mapping[p2]}")
    else:
        print("Graph mapping file not found.")

    # 2. Check Ensemble Model Behavior
    ensemble_path = PROJECT_ROOT / "models" / "ensemble_model.pkl"
    if ensemble_path.exists():
        try:
            ensemble = joblib.load(ensemble_path)
            print("\nEnsemble Model Loaded.")
            
            # Simulated Inputs from User Screenshot
            seq_prob = 0.5151
            graph_prob = 0.5052 # This is suspiciously close to 0.5 default, or a weak output
            
            # Test Scenario A: User Values
            feat = np.column_stack((np.array([seq_prob]), np.array([graph_prob])))
            pred = ensemble.predict_proba(feat)[:, 1][0]
            print(f"\nScenario A (User Inputs): Seq={seq_prob}, Graph={graph_prob}")
            print(f"  Ensemble Prediction: {pred:.4f}")
            
            # Test Scenario B: Default Graph Value
            seq_prob_b = 0.5151
            graph_prob_b = 0.5 # Exact default
            feat_b = np.column_stack((np.array([seq_prob_b]), np.array([graph_prob_b])))
            pred_b = ensemble.predict_proba(feat_b)[:, 1][0]
            print(f"Scenario B (Seq=0.5151, Graph=0.5 (Default)):")
            print(f"  Ensemble Prediction: {pred_b:.4f}")

            # Test Scenario C: High Graph Value
            seq_prob_c = 0.5151
            graph_prob_c = 0.9 
            feat_c = np.column_stack((np.array([seq_prob_c]), np.array([graph_prob_c])))
            pred_c = ensemble.predict_proba(feat_c)[:, 1][0]
            print(f"Scenario C (Seq=0.5151, Graph=0.9):")
            print(f"  Ensemble Prediction: {pred_c:.4f}")
            
            # Check Feature Importances
            if hasattr(ensemble, "feature_importances_"):
                print(f"\nFeature Importances: {ensemble.feature_importances_}")
                print("(Usually [Seq, Graph])")
                
        except Exception as e:
            print(f"Error loading/running ensemble: {e}")
    else:
        print("Ensemble model not found.")

if __name__ == "__main__":
    p1 = "ENSP00000269305"
    p2 = "ENSP00000361423"
    debug_prediction(p1, p2)
