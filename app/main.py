# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import logfire
from contextlib import asynccontextmanager
from typing import Optional

from app.config import settings


logfire.configure(token=settings.LOGFIRE_TOKEN)

# Now safe to import app modules - logfire is already active
from fastapi import FastAPI, HTTPException, Response
from app.agent.graph import rag_agent
from app.guardrails import initialize_rails, guard

from pydantic import BaseModel, Field


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize application-wide services before accepting requests."""
    initialize_rails()
    yield


# Initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API", lifespan=lifespan)

class QueryRequest(BaseModel):
    q: str = Field(min_length=1, max_length=12_000)
    thread_id: Optional[str] = Field(default="default_user", max_length=128)
    
    
@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}
    
    
@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = request.q
    thread_id = request.thread_id

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph..."
    }
    
    # Configuration for Memory (Thread ID)
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
        rail_fired, rail_response = guard(q)
        if rail_fired:
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": []
            }

        # Gate 2: LangGraph RAG pipeline
        # Run the graph synchronously to preserve Logfire context variables
        final_output = rag_agent.invoke(initial_state, config=config)
        
        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", [])
        }
    except Exception as exc:
        # Returning a 200 here makes the UI and evaluation pipeline record a
        # failed request as a valid answer. Keep the internal detail in logs
        # and expose only a safe error message to callers.
        logfire.error(
            "Backend execution failed.",
            error_type=type(exc).__name__,
            error=str(exc),
            thread_id=thread_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Unable to process the request. Please try again later.",
        ) from exc
