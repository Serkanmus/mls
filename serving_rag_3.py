import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForCausalLM
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import threading
import queue
import time
from concurrent.futures import Future
from prometheus_client import Counter, Histogram, start_http_server
import logging

#########################################
# LOGGING SETUP
#########################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

#########################################
# METRICS SETUP
#########################################
request_count = Counter('rag_requests_total', 'Total RAG requests received')
batch_duration = Histogram('rag_batch_processing_seconds', 'Time spent processing each batch')
batch_size_histogram = Histogram('rag_batch_size', 'Sizes of processed batches')
request_latency = Histogram('rag_request_total_latency_seconds', 'Overall request latency (from reception to response)')
embedding_duration = Histogram('rag_embedding_processing_seconds', 'Time spent during query embedding')
generation_duration = Histogram('rag_generation_processing_seconds', 'Time spent during text generation')

# Start metrics server (separate from FastAPI)
start_http_server(9100)

#########################################
# HYPERPARAMETERS FOR BATCHER
#########################################
MAX_BATCH_SIZE = 8
MAX_WAITING_TIME = 0.5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#########################################
# Setup: Documents, Models, Tokenizers
#########################################
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

#########################################
# FASTAPI + BATCHING IMPLEMENTATION
#########################################
app = FastAPI()
request_queue = queue.Queue()

def batch_worker():
    while True:
        batch_items = []
        start_time = time.time()
        # Block until at least one item is available
        first_item = request_queue.get()
        batch_items.append(first_item)

        # Attempt to collect more items for the batch until timeout or limit reached
        while len(batch_items) < MAX_BATCH_SIZE:
            elapsed = time.time() - start_time
            if elapsed > MAX_WAITING_TIME:
                break
            try:
                item = request_queue.get(timeout=MAX_WAITING_TIME - elapsed)
                batch_items.append(item)
            except queue.Empty:
                break

        batch_size_histogram.observe(len(batch_items))
        logging.info(f"[BATCH WORKER] Formed batch of size {len(batch_items)} (Queue size: {request_queue.qsize()})")

        queries = [item[0].query for item in batch_items]
        # Use the k from the first request in the batch; ideally, requests should be uniform in parameters
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

        # Deliver results to the waiting futures
        for item, output in zip(batch_items, results):
            future_obj = item[1]
            future_obj.set_result(output)

class QueryRequest(BaseModel):
    query: str
    k: int = 2

# Start batch worker thread (daemon so it won't block shutdown)
threading.Thread(target=batch_worker, daemon=True).start()

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
        result = fut.result()  # Wait for the batched processing to complete
    except Exception as e:
        logging.error(f"Error processing query '{payload.query}': {e}")
        return {"error": str(e)}

    total_latency = time.time() - start_time
    request_latency.observe(total_latency)
    logging.info(f"[RESPONSE] Completed query: '{payload.query}' with total latency: {total_latency:.3f} seconds")
    return {"query": payload.query, "result": result}

if __name__ == "__main__":
    logging.info("[SERVER] Starting RAG server on port 8147")
    uvicorn.run(app, host="0.0.0.0", port=8147)
