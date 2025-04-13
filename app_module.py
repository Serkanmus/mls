# app_module.py
from fastapi import FastAPI
from request_queue import RequestQueue  # 기존 모듈 임포트
from batcher import Batcher
from rag_pipeline import rag_pipeline, init_doc_embeddings
from pydantic import BaseModel
import asyncio
import time
from concurrent.futures import Future

# FastAPI 앱 객체 생성
app = FastAPI()

# 요청 스키마 정의
class QueryRequest(BaseModel):
    query: str
    k: int = 2
    mode: str = "batch"  # "single" 또는 "batch"
    batch_size: int | None = None

# 서버 초기화: 문서 임베딩 계산, request queue 및 batcher 설정
init_doc_embeddings()
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
        # 로그에 오류 상세 정보를 기록합니다.
        print("Exception in /rag endpoint:", error_msg)
        return {"error": str(e), "traceback": error_msg}, 500