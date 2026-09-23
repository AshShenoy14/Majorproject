import json

from fastapi import APIRouter, HTTPException

from src.utils.paths import PROJECT_ROOT

router = APIRouter(prefix="/evaluation", tags=["Analysis"])

FINAL_EVALUATION_PATH = PROJECT_ROOT / "assets" / "evaluation" / "final_test_metrics.json"


@router.get("/final",
            summary="Get Final Test Evaluation",
            description="Read-only. Returns the final held-out test evaluation written by src/analysis/compare_models.py.")
async def get_final_evaluation():
    """
    Serves assets/evaluation/final_test_metrics.json unchanged. No metrics are computed or stored here.
    """
    if not FINAL_EVALUATION_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Final evaluation results not found. Run `python src/analysis/compare_models.py` to generate them."
        )
    try:
        with open(FINAL_EVALUATION_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Could not read final evaluation results: {e}")


@router.get("/benchmarks",
            summary="Get All Evaluation Benchmarks",
            description="Returns bootstrap CIs, cold-start metrics, and external benchmarks (SHS27k, HuRI).")
async def get_all_benchmarks():
    """
    Returns bootstrap confidence intervals, cold-start simulation, and external benchmark artifacts.
    """
    eval_dir = PROJECT_ROOT / "assets" / "evaluation"
    res = {}
    for key, filename in [
        ("bootstrap_ci", "bootstrap_ci.json"),
        ("cold_start", "cold_start_eval.json"),
        ("shs27k", "external_benchmark_shs27k.json"),
        ("huri", "external_benchmark_huri.json"),
        ("calibration", "graph_calibration.json")
    ]:
        p = eval_dir / filename
        if p.exists():
            try:
                res[key] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return res
