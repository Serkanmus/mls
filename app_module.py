from fastapi import FastAPI
from request_queue import RequestQueue  # 기존 모듈 임포트
from batcher import Batcher
from rag_pipeline import rag_pipeline, init_doc_embeddings
from pydantic import BaseModel
import asyncio
import time
from concurrent.futures import Future

app = FastAPI()

# Global readiness flag; workers report readiness only after initialization.
is_ready = False

@app.on_event("startup")
async def startup_event():
    global is_ready
    # Perform all necessary initialization such as model loading and computing embeddings.
    init_doc_embeddings()
    is_ready = True
    print("Worker ready: Initialization complete.")

# Health endpoint for readiness check.
@app.get("/health")
async def health_check():
    if is_ready:
        return {"status": "ready"}
    return {"status": "initializing"}, 503

# Request schema definition.
class QueryRequest(BaseModel):
    query: str
    k: int = 2
    mode: str = "batch"  # "single" or "batch"
    batch_size: int | None = None

# Initialize the request queue and batcher.
request_queue = RequestQueue()
batcher = Batcher(request_queue)

@app.post("/rag")
async def predict(payload: QueryRequest):
    try:
        if payload.mode == "single":
            server_start = time.time()
            result_dict = await asyncio.to_thread(rag_pipeline, payload.query, payload.k, "single")
            server_end = time.time()
            return {
                "query": payload.query,
                "result": result_dict["result"],
                "metrics": result_dict["metrics"],
                "server_start_time": server_start,
                "server_end_time": server_end
            }
        future = Future()
        request_queue.enqueue((payload, future))
        result = await asyncio.get_running_loop().run_in_executor(None, future.result)
        return {
            "query": payload.query,
            "result": result["result"],
            "metrics": result["metrics"],
            "server_start_time": result["server_start_time"],
            "server_end_time": result["server_end_time"]
        }
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print("Exception in /rag endpoint:", error_msg)
        return {"error": str(e), "traceback": error_msg}, 500
