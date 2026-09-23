import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager

# Add project root and configure environments BEFORE importing heavy ML libraries
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.backend import state
from app.backend.state import insert_novel_node_knn  # re-exported for tests/back-compat
from app.backend.routers import prediction, network, biology, mutation, evaluation, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    await state.load_system()
    yield

app = FastAPI(
    title="TransGraph-PPI API",
    description="Hybrid Ensemble PPI Prediction System with Real Data",
    lifespan=lifespan
)

# CORS configuration
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction.router)
app.include_router(network.router)
app.include_router(biology.router)
app.include_router(mutation.router)
app.include_router(evaluation.router)
app.include_router(chat.router)


if __name__ == "__main__":
    # Auto-reload trigger
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
