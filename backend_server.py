#!/usr/bin/env python
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import threading
import queue
import time
from concurrent.futures import Future
from prometheus_client import Counter, Histogram, start_http_server
import logging
import argparse
import socket

# -------------------
# Logging Setup
# -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# -------------------
# Helper function to get the current IP address.
# -------------------
def get_my_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

# -------------------
# Metrics Setup
# -------------------
request_count = Counter('rag_requests_total', 'Total RAG requests received')
batch_duration = Histogram('rag_batch_processing_seconds', 'Time spent processing each batch')
batch_size_histogram = Histogram('rag_batch_size', 'Sizes of processed batches')
request_latency = Histogram('rag_request_total_latency_seconds', 'Overall request latency (from reception to response)')
embedding_duration = Histogram('rag_embedding_processing_seconds', 'Time spent during query embedding')
generation_duration = Histogram('rag_generation_processing_seconds', 'Time spent during text generation')

# Start Prometheus metrics server on port 9100
start_http_server(9100)

# -------------------
# Batcher Hyperparameters
# -------------------
NUM_WORKERS = 1
MAX_BATCH_SIZE = 32
MAX_WAITING_TIME = 0.2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------
# Documents and Model Setup
# -------------------
documents = [
    "Cats are small furry carnivores that are often kept as pets.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "Hummingbirds can hover in mid-air by rapidly flapping their wings."
]

EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME, use_fast=False)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(DEVICE)

CHAT_MODEL_NAME = "facebook/opt-125m"
chat_tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_NAME)
chat_model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL_NAME).to(DEVICE)

doc_embeddings = []

def get_embedding_batch(texts: list[str]) -> np.ndarray:
    start_embedding = time.time()
    logging.info(f"[EMBEDDING] Encoding {len(texts)} queries")
    inputs = embed_tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = embed_model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    duration = time.time() - start_embedding
    embedding_duration.observe(duration)
    logging.info(f"[EMBEDDING] Completed encoding in {duration:.3f} seconds")
    return emb

def retrieve_top_k(query_embs: np.ndarray, k: int = 2) -> list[list[str]]:
    global doc_embeddings
    sims = doc_embeddings @ query_embs.T
    results = []
    for i in range(sims.shape[1]):
        sim_vec = sims[:, i]
        top_k_indices = np.argsort(sim_vec.ravel())[::-1][:k]
        retrieved_docs = [documents[idx] for idx in top_k_indices]
        results.append(retrieved_docs)
    return results

def batch_generation(prompts: list[str], max_new_tokens=50):
    start_gen = time.time()
    logging.info(f"[GENERATION] Generating responses for {len(prompts)} prompts")
    inputs = chat_tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = chat_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    decoded = [chat_tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
    duration = time.time() - start_gen
    generation_duration.observe(duration)
    logging.info(f"[GENERATION] Completed generation in {duration:.3f} seconds")
    return decoded

def rag_pipeline_batch(queries: list[str], k: int = 2) -> list[str]:
    logging.info(f"[RAG PIPELINE] Processing batch of {len(queries)} queries")
    query_embs = get_embedding_batch(queries)
    retrieved_docs_lists = retrieve_top_k(query_embs, k)
    prompts = []
    for query, docs in zip(queries, retrieved_docs_lists):
        context_str = "\n".join(docs)
        prompt = f"Question: {query}\nContext:\n{context_str}\nAnswer:"
        prompts.append(prompt)
    generated_texts = batch_generation(prompts, max_new_tokens=50)
    return generated_texts

# -------------------
# FastAPI and Batching Implementation
# -------------------
app = FastAPI()
request_queue = queue.Queue()

def batch_worker():
    while True:
        batch_items = []
        start_time = time.time()
        # Block until at least one request is available.
        first_item = request_queue.get()
        batch_items.append(first_item)

        # Try to form a larger batch until MAX_BATCH_SIZE or timeout.
        while len(batch_items) < MAX_BATCH_SIZE:
            elapsed = time.time() - start_time
            if elapsed > MAX_WAITING_TIME:
                break
            try:
                item = request_queue.get(timeout=MAX_WAITING_TIME - elapsed)
                batch_items.append(item)
            except queue.Empty:
                break

        batch_size = len(batch_items)
        batch_size_histogram.observe(batch_size)
        logging.info(f"[BATCH WORKER] Formed batch of size {batch_size} (Queue size: {request_queue.qsize()})")
        logging.info(f"[BATCH WORKER] Time elapsed while batching: {time.time() - start_time:.3f}s")

        queries = [item[0].query for item in batch_items]
        ks = [item[0].k for item in batch_items]
        k_for_batch = ks[0]

        try:
            with batch_duration.time():
                results = rag_pipeline_batch(queries, k=k_for_batch)
        except Exception as e:
            logging.exception("Error during batch processing")
            for item in batch_items:
                future_obj = item[1]
                future_obj.set_exception(e)
            continue

        for item, output in zip(batch_items, results):
            future_obj = item[1]
            future_obj.set_result(output)

class QueryRequest(BaseModel):
    query: str
    k: int = 2

# Start the batch worker(s).
for i in range(NUM_WORKERS):
    threading.Thread(target=batch_worker, name=f"Worker-{i+1}", daemon=True).start()

def init_doc_embeddings():
    global doc_embeddings
    logging.info("[INIT] Computing static document embeddings")
    doc_embeddings = get_embedding_batch(documents)

init_doc_embeddings()

@app.post("/rag")
def predict(payload: QueryRequest):
    logging.info(f"[REQUEST] Received query: '{payload.query}'")
    start_time = time.time()
    request_count.inc()

    fut = Future()
    request_queue.put((payload, fut))
    try:
        result = fut.result()
    except Exception as e:
        logging.error(f"Error processing query '{payload.query}': {e}")
        return {"error": str(e)}

    total_latency = time.time() - start_time
    request_latency.observe(total_latency)
    logging.info(f"[RESPONSE] Completed query: '{payload.query}' with total latency: {total_latency:.3f}s")
    return {"query": payload.query, "result": result}

# -------------------
# Entry Point: Parse Port and Run Server
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backend RAG Server")
    parser.add_argument("--port", type=int, default=8147, help="Port number to run the server on")
    args = parser.parse_args()
    ip = get_my_ip()
    logging.info(f"[SERVER] Starting RAG server on {ip}:{args.port}")
    uvicorn.run(app, host=ip, port=args.port)
