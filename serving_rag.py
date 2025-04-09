import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import time

print(torch.cuda.is_available())
print(torch.version.cuda)
devicee = "cuda"
app = FastAPI()

# Example documents in memory
documents = [
    "Cats are small furry carnivores that are often kept as pets.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "Hummingbirds can hover in mid-air by rapidly flapping their wings."
]

# 1. Load embedding model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
embed_model.to(devicee)

# Basic Chat LLM
#chat_pipeline = pipeline("text-generation", model="facebook/opt-125m", device=0)
# Note: try this 1.5B model if you got enough GPU memory
chat_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct", device=devicee)



## Hints:

### Step 3.1:
# 1. Initialize a request queue
# 2. Initialize a background thread to process the request (via calling the rag_pipeline function)
# 3. Modify the predict function to put the request in the queue, instead of processing it immediately

### Step 3.2:
# 1. Take up to MAX_BATCH_SIZE requests from the queue or wait until MAX_WAITING_TIME
# 2. Process the batched requests

def get_embedding(text: str) -> np.ndarray:
    """Compute a simple average-pool embedding."""
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(devicee) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Precompute document embeddings
doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])

### You may want to use your own top-k retrieval method (task 1)
def retrieve_top_k(query_emb: np.ndarray, k: int = 2) -> list:
    """Retrieve top-k docs via dot-product similarity."""
    sims = doc_embeddings @ query_emb.T
    top_k_indices = np.argsort(sims.ravel())[::-1][:k]
    return [documents[i] for i in top_k_indices]


def rag_pipeline(query: str, k: int = 2) -> str:
    t0 = time.time()
    query_emb = get_embedding(query)
    t1 = time.time()

    retrieved_docs = retrieve_top_k(query_emb, k)
    t2 = time.time()

    context = "\n".join(retrieved_docs)
    prompt = f"Question: {query}\nContext:\n{context}\nAnswer:"
    
    generated = chat_pipeline(prompt, max_length=50, do_sample=True)[0]["generated_text"]
    t3 = time.time()

    # Print out durations
    print(f"Embedding took {t1 - t0:.3f}s, Retrieval took {t2 - t1:.3f}s, Generation took {t3 - t2:.3f}s")

    return generated


# Define request model
class QueryRequest(BaseModel):
    query: str
    k: int = 2

@app.post("/rag")
def predict(payload: QueryRequest):
    result = rag_pipeline(payload.query, payload.k)
    
    return {
        "query": payload.query,
        "result": result,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
