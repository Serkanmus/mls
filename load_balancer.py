from fastapi import FastAPI, Request
import httpx
import itertools
from fastapi.responses import JSONResponse

app = FastAPI()

class LoadBalancer:
    def __init__(self, worker_urls):
        """
        :param worker_urls: List of base URLs for worker servers (e.g., ["http://127.0.0.1:8777", ...])
        """
        self.worker_urls = worker_urls
        self.worker_iter = itertools.cycle(worker_urls)
        self.client = httpx.AsyncClient()

    async def forward(self, request: Request):
        # Select the next worker URL using round-robin
        target_base = next(self.worker_iter)
        # Pass the request path and query string as-is
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

# By default, 4 worker URLs are used (can be adjusted for combined or loadbalancer modes)
WORKER_URLS = [
    "http://127.0.0.1:8770",
    "http://127.0.0.1:8778",
    "http://127.0.0.1:8779",
    "http://127.0.0.1:8780"
]

lb = LoadBalancer(WORKER_URLS)

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_all(request: Request):
    """Forward requests to the workers via the load balancer for all routes"""
    response = await lb.forward(request)
    # Return a JSONResponse in case of an error
    if response.__class__.__name__ == "JSONResponse":
        return response
    return JSONResponse(content=response.json(), status_code=response.status_code)