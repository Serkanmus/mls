from fastapi import FastAPI, Request
import httpx
import itertools
from fastapi.responses import JSONResponse
from threading import Lock

app = FastAPI()

# Global request counter for autoscaling metrics.
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
        self.worker_urls = worker_urls
        self.worker_iter = itertools.cycle(worker_urls)
        self.client = httpx.AsyncClient()
        # Dictionary to count per-worker requests.
        self.worker_request_counts = {url: 0 for url in worker_urls}
        self.lock = Lock()

    async def _get_healthy_worker(self):
        """
        Try each worker URL once and return the first one that passes a health check.
        If none are healthy, return None.
        """
        num_urls = len(self.worker_urls)
        for _ in range(num_urls):
            target_base = next(self.worker_iter)
            health_url = f"{target_base}/health"
            try:
                resp = await self.client.get(health_url, timeout=2.0)
                if resp.status_code == 200 and resp.json().get("status") == "ready":
                    return target_base
            except Exception:
                continue
        return None

    async def forward(self, request: Request):
        # Increment global request counter.
        increment_request_count()
        # Try to get a healthy worker; if none, return a 503 error.
        target_base = await self._get_healthy_worker()
        if not target_base:
            return JSONResponse({"error": "No healthy worker available"}, status_code=503)
        # Update the per-worker counter.
        with self.lock:
            if target_base not in self.worker_request_counts:
                self.worker_request_counts[target_base] = 0
            self.worker_request_counts[target_base] += 1
            print(f"[LoadBalancer] Forwarding to {target_base}; count={self.worker_request_counts[target_base]}")

        # Construct the full target URL.
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
            self.worker_request_counts = {url: 0 for url in new_urls}

# Default worker URLs (to be updated by the autoscaler).
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

# Optional: a metrics endpoint to query per-worker request counts.
@app.get("/metrics")
async def get_metrics():
    return {"worker_request_counts": lb.worker_request_counts}
