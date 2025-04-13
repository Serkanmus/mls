import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import logging
from utils import device, logger, get_hardware_metrics

# ------------------------------------------------------------------------------
# Load documents and models
# ------------------------------------------------------------------------------
documents = [
    "Cats are small furry carnivores that are often kept as pets.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "Hummingbirds can hover in mid-air by rapidly flapping their wings."
]

EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
CHAT_MODEL_NAME = "facebook/opt-125m"

embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device)
chat_tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_NAME)
chat_model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL_NAME).to(device)

# ------------------------------------------------------------------------------
# Global document embeddings (precomputed)
# ------------------------------------------------------------------------------
doc_embeddings = None

def init_doc_embeddings():
    """Compute and cache document embeddings once at startup."""
    global doc_embeddings
    logger.info("Initializing document embeddings...")
    doc_embeddings = get_embedding(documents, mode="batch")
    logger.info("Initialized document embeddings.")

# ------------------------------------------------------------------------------
# Embedding, Retrieval, Generation utilities
# ------------------------------------------------------------------------------
def get_embedding(text, mode="single"):
    if mode == "batch":
        if not isinstance(text, list):
            raise ValueError("For batch mode, text must be a list of strings.")
    else:
        if isinstance(text, list):
            text = text[0]

    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = embed_model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    logger.info(f"get_embedding ({mode}): {len(text) if isinstance(text, list) else 1} items")
    return embeddings

def retrieve_top_k(query_emb, k=2, mode="single"):
    global doc_embeddings
    sims = doc_embeddings @ query_emb.T

    if mode == "batch":
        return [
            [documents[i] for i in np.argsort(sims[:, idx].ravel())[::-1][:k]]
            for idx in range(sims.shape[1])
        ]
    else:
        top_k_indices = np.argsort(sims.ravel())[::-1][:k]
        return [documents[i] for i in top_k_indices]

def generate(prompt, mode="single", max_new_tokens=50):
    if mode == "batch":
        if not isinstance(prompt, list):
            raise ValueError("Prompt must be a list in batch mode.")
        inputs = chat_tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = chat_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
        return chat_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    if isinstance(prompt, list):
        prompt = prompt[0]
    inputs = chat_tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = chat_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
    return chat_tokenizer.decode(outputs[0], skip_special_tokens=True)

# ------------------------------------------------------------------------------
# Main RAG pipeline
# ------------------------------------------------------------------------------
def rag_pipeline(query, k=2, mode="single"):
    total_start = time.time()

    if mode == "batch":
        if not isinstance(query, list):
            query = [query]

        embed_start = time.time()
        query_embs = get_embedding(query, mode="batch")
        embedding_time = time.time() - embed_start

        retrieval_start = time.time()
        docs_per_query = retrieve_top_k(query_embs, k=k, mode="batch")
        retrieval_time = time.time() - retrieval_start

        prompts = [f"Question: {q}\nContext:\n" + "\n".join(ctx) + "\nAnswer:" for q, ctx in zip(query, docs_per_query)]

        generation_start = time.time()
        generated = generate(prompts, mode="batch")
        generation_time = time.time() - generation_start

        total_time = time.time() - total_start

        return [
            {
                "result": g,
                "metrics": {
                    "embedding_time": embedding_time,
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "total_time": total_time,
                    "batch_size": len(query),
                    "hardware": get_hardware_metrics()
                }
            } for g in generated
        ]

    if isinstance(query, list):
        query = query[0]

    embed_start = time.time()
    query_emb = get_embedding(query, mode="single")
    embedding_time = time.time() - embed_start

    retrieval_start = time.time()
    docs = retrieve_top_k(query_emb, k=k, mode="single")
    retrieval_time = time.time() - retrieval_start

    prompt = f"Question: {query}\nContext:\n" + "\n".join(docs) + "\nAnswer:"

    generation_start = time.time()
    generated = generate(prompt, mode="single")
    generation_time = time.time() - generation_start

    total_time = time.time() - total_start

    return {
        "result": generated,
        "metrics": {
            "embedding_time": embedding_time,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "hardware": get_hardware_metrics()
        }
    }
