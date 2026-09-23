from fastapi import APIRouter, HTTPException
from typing import List

from app.backend import state
from app.backend.schemas import BioMetaResponse, FeasibilityResponse

router = APIRouter(tags=["Biology"])


@router.get("/drug_targets",
            summary="Get Drug Targets",
            description="Retrieves known drug targets for a given list of proteins.")
async def get_drug_targets(proteins: str = None):
    """
    Returns drug target information from ChEMBL/UniProt mappings.
    """

    try:
        if proteins:
            p_list = proteins.split(",")
        else:
            p_list = ["ENSP00000327694", "ENSP00000373627"]

        df = state.managers["target"].get_targets(p_list)

        if df.empty:
            return []

        return df.to_dict(orient="records")

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bio/metadata",
            response_model=List[BioMetaResponse],
            summary="Get Biological Metadata",
            description="Fetches localization and pathway information for specific proteins.")
async def get_bio_metadata(proteins: str):
    """
    Returns subcellular localization and functional pathway data.
    """

    if "bio" not in state.managers:
        raise HTTPException(status_code=503, detail="Biological Manager not initialized")

    try:
        p_list = proteins.split(",")
        df = state.managers["bio"].get_bio_metadata(p_list)
        return df.to_dict(orient="records")
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bio/feasibility",
            response_model=FeasibilityResponse,
            summary="Check Interaction Feasibility",
            description="Checks if two proteins share compatible subcellular localizations.")
async def check_feasibility(p1: str, p2: str):
    """
    Determines if an interaction is physically possible based on biological context.
    """

    if "bio" not in state.managers:
        raise HTTPException(status_code=503, detail="Biological Manager not initialized")

    try:
        return state.managers["bio"].check_localization_compatibility(p1, p2)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/vulnerability",
            summary="Calculate Pathway Vulnerability",
            description="Assesses how fragile a biological pathway is to mutations in a specific protein pair.")
async def get_vulnerability(p1: str, p2: str, delta: float):
    """
    Calculates vulnerability scores for downstream pathways.
    """

    if "bio" not in state.managers:
        raise HTTPException(status_code=503, detail="Biological Manager not initialized")
    return state.managers["bio"].calculate_pathway_vulnerability(p1, p2, delta)
