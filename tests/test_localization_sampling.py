import pytest
import pandas as pd
from src.data.preprocess_data import generate_negative_samples, is_co_localized

def test_localization_constrained_sampling():
    # Setup simple mock positive interactions
    positive_df = pd.DataFrame([
        {"protein1": "PROT_A", "protein2": "PROT_B"},
        {"protein1": "PROT_B", "protein2": "PROT_C"},
    ])
    all_proteins = ["PROT_A", "PROT_B", "PROT_C", "PROT_D", "PROT_E"]
    
    # Mock localization dictionary
    # A and D are Cytoplasm (co-localized)
    # B, C, E are Nucleus (co-localized with each other, but not A or D)
    loc_dict = {
        "PROT_A": {"cytoplasm"},
        "PROT_B": {"nucleus"},
        "PROT_C": {"nucleus"},
        "PROT_D": {"cytoplasm"},
        "PROT_E": {"nucleus"}
    }
    
    # 1. Test helper directly
    assert is_co_localized("PROT_A", "PROT_D", loc_dict) is True
    assert is_co_localized("PROT_B", "PROT_E", loc_dict) is True
    assert is_co_localized("PROT_A", "PROT_E", loc_dict) is False
    
    # 2. Generate negatives and check constraints
    neg_df = generate_negative_samples(
        positive_df=positive_df,
        all_proteins=all_proteins,
        ratio=2.0,
        hard_ratio=0.5,
        loc_dict=loc_dict
    )
    
    assert len(neg_df) > 0
    # Every pair in neg_df must be co-localized according to our loc_dict,
    # or have one element missing from loc_dict (but here all are present)
    for _, row in neg_df.iterrows():
        p1, p2 = row["protein1"], row["protein2"]
        assert is_co_localized(p1, p2, loc_dict) is True
