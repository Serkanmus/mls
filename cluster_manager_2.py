#!/usr/bin/env python
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
import httpx
import threading
import time
import subprocess
import logging
import socket
import torch  # Added to determine the number of GPUs

# -------------------
# Logging Setup
# -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

app = FastAPI()

def get_my_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # This does not actually send data.
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

# -------------------
# Global Async HTTP Client
# -------------------
async_client = httpx.AsyncClient(
    timeout=10,
    limits=httpx.Limits(max_connections=50, max_keepalive_connections=20)
)

# -------------------
# Determine available GPUs
# -------------------
if torch.cuda.is_available():
    available_gpus = list(range(torch.cuda.device_count()))
    logging.info(f"Available GPUs: {available_gpus}")
else:
    available_gpus = [None]
    logging.info("No GPUs available, running on CPU.")

# -------------------
# Global list of backend servers.
# -------------------
backend_servers = []
backend_lock = threading.Lock()

# -------------------
# Autoscaler metrics: track number of requests forwarded during each interval.
# -------------------
request_count = 0
request_count_lock = threading.Lock()

# -------------------
# Autoscaler parameters
# -------------------
MIN_BACKENDS = 1
MAX_BACKENDS = 10
SCALE_UP_THRESHOLD = 30   # if more than 30 requests per interval, scale up
SCALE_DOWN_THRESHOLD = 10 # if fewer than 10 requests per interval, scale down
SCALER_INTERVAL = 10      # check every 10 seconds

# -------------------
# For round-robin load balancing.
# -------------------
rr_index = 0
rr_lock = threading.Lock()

# -------------------
# Request Model
# -------------------
class QueryRequest(BaseModel):
    query: str
    k: int = 2

def get_next_backend():
    global rr_index
    with rr_lock, backend_lock:
        if not backend_servers:
            return None
        backend = backend_servers[rr_index % len(backend_servers)]
        rr_index = (rr_index + 1) % len(backend_servers)
        return backend["url"]

# -------------------
# Asynchronous Forwarding using the global async client
# -------------------
@app.post("/rag")
async def route_request(payload: QueryRequest, req: Request):
    global request_count
    with request_count_lock:
        request_count += 1

    backend_url = get_next_backend()
    if backend_url is None:
        return {"error": "No backend available"}
    try:
        response = await async_client.post(backend_url + "/rag", json=payload.dict())
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def add_backend():
    """Start a new backend server (runs backend_server.py) on a new port with a dedicated GPU."""
    with backend_lock:
        next_port = max([b["port"] for b in backend_servers]) + 1 if backend_servers else 8147
        # Determine GPU assignment using round-robin over available_gpus.
        if available_gpus[0] is not None:
            next_gpu = available_gpus[len(backend_servers) % len(available_gpus)]
            cmd = ["python", "backend_server.py", "--port", str(next_port), "--gpu", str(next_gpu)]
            logging.info(f"Assigning GPU {next_gpu} to backend on port {next_port}")
        else:
            cmd = ["python", "backend_server.py", "--port", str(next_port)]
            logging.info(f"No GPU assigned for backend on port {next_port} (running on CPU)")
        process = subprocess.Popen(cmd)
        time.sleep(1)  # Give the new backend time to start up.
        ip = get_my_ip()
        backend = {"url": f"http://{ip}:{next_port}", "port": next_port, "process": process}
        backend_servers.append(backend)
        logging.info(f"[Autoscaler] Added backend on port {next_port}. Total backends: {len(backend_servers)}")

def remove_backend():
    """Terminate the most recently added backend server."""
    with backend_lock:
        if len(backend_servers) <= MIN_BACKENDS:
            return
        backend = backend_servers.pop()
        process = backend["process"]
        process.terminate()
        logging.info(f"[Autoscaler] Removed backend on port {backend['port']}. Total backends: {len(backend_servers)}")

def autoscaler():
    """Periodically adjusts the number of backend servers based on request rate."""
    global request_count
    while True:
        time.sleep(SCALER_INTERVAL)
        with request_count_lock:
            rps = request_count / SCALER_INTERVAL
            request_count = 0
        logging.info(f"[Autoscaler] Average requests/sec: {rps:.2f}")
        with backend_lock:
            current_backends = len(backend_servers)
        if rps > SCALE_UP_THRESHOLD and current_backends < MAX_BACKENDS:
            logging.info("[Autoscaler] Scaling up!")
            add_backend()
        elif rps < SCALE_DOWN_THRESHOLD and current_backends > MIN_BACKENDS:
            logging.info("[Autoscaler] Scaling down!")
            remove_backend()

def initial_setup():
    """Start the cluster with the minimum number of backend servers."""
    for _ in range(MIN_BACKENDS):
        add_backend()

if __name__ == "__main__":
    initial_setup()
    scaler_thread = threading.Thread(target=autoscaler, daemon=True)
    scaler_thread.start()
    ip = get_my_ip()
    logging.info(f"[Cluster Manager] Starting load balancer on {ip}:8100")
    uvicorn.run(app, host=ip, port=8100)
