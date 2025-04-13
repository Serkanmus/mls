from fastapi import FastAPI, Request
import httpx
import itertools
from fastapi.responses import JSONResponse
from threading import Lock

app = FastAPI()

# Global request counter for autoscaling metrics (total count across all requests)
request_count = 0
request_count_lock = Lock()

def increment_request_count():
    global request_count
    with request_count_lock:
        request_count += 1

def get_and_reset_request_count():
    global request_count
    with request_count_lock:
        count = request_count
        request_count = 0
    return count

class LoadBalancer:
    def __init__(self, worker_urls):
        """
        :param worker_urls: List of base URLs for worker servers (e.g., ["http://127.0.0.1:8777", ...])
        """
        self.worker_urls = worker_urls
        self.worker_iter = itertools.cycle(worker_urls)
        self.client = httpx.AsyncClient()
        # Initialize a dictionary to keep track of per-worker request counts.
        self.worker_request_counts = {url: 0 for url in worker_urls}
        self.lock = Lock()  # Protect dictionary updates.

    async def forward(self, request: Request):
        # Increment global request counter.
        increment_request_count()
        # Select the next worker URL using round-robin.
        target_base = next(self.worker_iter)
        # Increment per-worker counter.
        with self.lock:
            if target_base not in self.worker_request_counts:
                self.worker_request_counts[target_base] = 0
            self.worker_request_counts[target_base] += 1
            # Optionally log the current count for this worker.
            print(f"[LoadBalancer] Forwarding to {target_base}; count={self.worker_request_counts[target_base]}")

        # Construct the target URL.
        target_url = f"{target_base}{request.url.path}"
        if request.url.query:
            target_url = f"{target_url}?{request.url.query}"
        body = await request.body()
        try:
            response = await self.client.request(
                method=request.method,
                url=target_url,
                headers=request.headers,
                content=body,
                timeout=10.0
            )
            return response
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    def update_worker_urls(self, new_urls):
        with self.lock:
            self.worker_urls = new_urls
            self.worker_iter = itertools.cycle(new_urls)
            # Reset and reinitialize the per-worker counters.
            self.worker_request_counts = {url: 0 for url in new_urls}

# By default, these worker URLs will be updated in combined mode.
WORKER_URLS = [
    "http://127.0.0.1:8770",
    "http://127.0.0.1:8778",
    "http://127.0.0.1:8779",
    "http://127.0.0.1:8780"
]

lb = LoadBalancer(WORKER_URLS)

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_all(request: Request):
    response = await lb.forward(request)
    if response.__class__.__name__ == "JSONResponse":
        return response
    return JSONResponse(content=response.json(), status_code=response.status_code)

# Optionally, add an endpoint to inspect per-worker metrics.
@app.get("/metrics")
async def get_metrics():
    return {
        "worker_request_counts": lb.worker_request_counts
    }
