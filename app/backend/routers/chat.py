from fastapi import APIRouter, HTTPException

from app.backend import state
from app.backend.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["AI Assistant"])


@router.get("/greeting",
            summary="Get AI Assistant Greeting",
            description="Returns a welcome message and suggested questions for the protein assistant.")
async def get_chat_greeting():
    if "assistant" not in state.analyzers:
        raise HTTPException(status_code=503, detail="Protein Assistant not initialized")
    return state.analyzers["assistant"].get_greeting()


@router.post("",
             response_model=ChatResponse,
             summary="Chat with Protein AI Assistant",
             description="Ask questions about proteins, diseases, drug targets, and biology concepts.")
async def chat_with_assistant(request: ChatRequest):
    if "assistant" not in state.analyzers:
        raise HTTPException(status_code=503, detail="Protein Assistant not initialized")

    try:
        result = state.analyzers["assistant"].answer(request.message)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
