import unittest
import torch
import sys
import os
from pathlib import Path

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.irlm_module import BiDirectionalCrossAttention, InteractionRegionLocalizationModule
from src.analysis.irlm_analyzer import IRLMAnalyzer
from fastapi.testclient import TestClient
from app.backend.main import app

class TestIRLM(unittest.TestCase):

    def test_irlm_pytorch_module(self):
        """Test PyTorch forward pass of BiDirectionalCrossAttention."""
        d_model = 480
        attn = BiDirectionalCrossAttention(embed_dim=d_model, num_heads=4)

        # Synthetic per-residue embeddings: L_A=20, L_B=30
        h_a = torch.randn(20, d_model)
        h_b = torch.randn(30, d_model)

        attn_ab, attn_ba, interaction_matrix = attn(h_a, h_b)

        self.assertEqual(attn_ab.shape, (20, 30))
        self.assertEqual(attn_ba.shape, (30, 20))
        self.assertEqual(interaction_matrix.shape, (20, 30))
        print("[PASS] PyTorch BiDirectionalCrossAttention unit test passed.")

    def test_irlm_wrapper_and_mutations(self):
        """Test analyzer localization & mutation annotation."""
        analyzer = IRLMAnalyzer(esm_extractor=None, graph_model=None, device="cpu")

        # Fake IRLM output data
        irlm_data = {
            "protein_A_region": [5, 15],
            "protein_B_region": [1, 10],
            "protein_A_key_residues": [7, 8, 9],
            "protein_B_key_residues": [3, 4],
        }

        mutation_results = [
            {
                "protein": 1,
                "pos": 8,
                "orig": "A",
                "mut": "V",
                "base_score": 0.80,
                "mutated_score": 0.65,
                "impact_delta": -0.15,
                "interpretation": "Disruptive"
            },
            {
                "protein": 1,
                "pos": 50,
                "orig": "G",
                "mut": "R",
                "base_score": 0.80,
                "mutated_score": 0.79,
                "impact_delta": -0.01,
                "interpretation": "Neutral"
            }
        ]

        annotated = analyzer.annotate_mutations_with_irlm(mutation_results, irlm_data)
        self.assertTrue(annotated[0]["is_in_interaction_region"])
        self.assertFalse(annotated[1]["is_in_interaction_region"])
        print("[PASS] IRLM Mutation Annotation unit test passed.")

    def test_localize_api_endpoint(self):
        """Test FastAPI /analysis/localize endpoint."""
        client = TestClient(app)
        payload = {
            "protein1_id": "ENSP00000327694",
            "protein2_id": "ENSP00000373627",
            "protein1_seq": "MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPF",
            "protein2_seq": "MSLFLRNAMVRKKIQVFEQEVMEKLLSKDEELQKAKELLAEKRAELEKELEAEAEKY",
            "base_probability": 0.85
        }
        response = client.post("/analysis/localize", json=payload)
        self.assertIn(response.status_code, [200, 503])
        if response.status_code == 200:
            data = response.json()
            self.assertIn("protein_A_region", data)
            self.assertIn("protein_B_region", data)
            print("[PASS] FastAPI /analysis/localize endpoint test passed.")
        else:
            print("[NOTICE] Service returned 503 (model loading state in unit test env).")

if __name__ == "__main__":
    unittest.main()

