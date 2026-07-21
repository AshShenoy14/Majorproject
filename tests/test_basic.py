import pytest
import sys
import os

def test_imports():
    # Attempt to import main project modules to check path resolution
    import src.models.sequence_model as seq_mod
    import src.models.graph_model as graph_mod
    assert seq_mod is not None
    assert graph_mod is not None
