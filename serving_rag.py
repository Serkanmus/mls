#!/usr/bin/env python
import os
import time
import argparse
import multiprocessing
import threading
import uvicorn
import itertools
import asyncio
from fastapi import Request
from utils import get_ip, update_env_variable

# Use the existing autoscaler and load_balancer modules as is.
from autoscaler import AutoScaler
from load_balancer import lb as load_balancer_instance, app as lb_app

# Import the FastAPI application object from a separate module.
from app_module import app

def run_worker(host: str, port: int, gpu_id: str, dev: bool):
    # Set the GPU environment variable for this worker process.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"Worker starting on {host}:{port} using GPU {gpu_id}")
    uvicorn.run(app, host=host, port=port, reload=dev)

def main():
    parser = argparse.ArgumentParser(
        description="Serving_rag launcher: supports worker, autoscale, loadbalancer, and combined modes with GPU assignment"
    )
    parser.add_argument("--dev", action="store_true", help="Development mode (enable reload)")
    parser.add_argument("--mode", type=str, choices=["worker", "autoscale", "loadbalancer", "combined"],
                        default="worker", help="Select execution mode")
    parser.add_argument("--host", type=str, default=None, help="IP address to bind the server (auto-detected if not provided)")
    parser.add_argument("--base_port", type=int, default=8777, help="Starting port number for worker servers")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes (used in autoscale/combined modes)")
    parser.add_argument("--lb_port", type=int, default=8888, help="Load Balancer port number")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", 
                        help="Comma-separated list of GPU IDs to assign to worker processes")
    args = parser.parse_args()

    # Parse the provided GPU IDs.
    gpu_ids = [gpu.strip() for gpu in args.gpu_ids.split(",") if gpu.strip()]
    num_gpus = len(gpu_ids)
    if num_gpus == 0:
        raise ValueError("No GPU IDs provided via --gpu_ids")

    if args.host is None:
        args.host = get_ip(args.dev)
    update_env_variable("HOST_IP", args.host)
    print(f"Server Host: {args.host}")

    if args.mode == "worker":
        # In worker mode, run a single worker using the first GPU in the list.
        gpu_id = gpu_ids[0]
        print(f"[Worker mode] Running a single worker on {args.host}:{args.base_port} with GPU {gpu_id}")
        run_worker(args.host, args.base_port, gpu_id, args.dev)
    elif args.mode == "autoscale":
        print(f"[Autoscale mode] Starting {args.workers} worker(s) from {args.host}:{args.base_port} with round-robin GPU assignment")
        processes = []
        for i in range(args.workers):
            port = args.base_port + i
            gpu_id = gpu_ids[i % num_gpus]  # Round-robin assignment of GPU IDs.
            p = multiprocessing.Process(target=run_worker, args=(args.host, port, gpu_id, args.dev))
            p.start()
            processes.append(p)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("KeyboardInterrupt: shutting down autoscale mode...")
            for p in processes:
                p.terminate()
    elif args.mode == "loadbalancer":
        print(f"[Load Balancer mode] Running load balancer on {args.host}:{args.lb_port}")
        # In loadbalancer mode, it is assumed that worker processes are launched separately with proper GPU assignments.
        worker_urls = [f"http://{args.host}:{args.base_port + i}" for i in range(args.workers)]
        load_balancer_instance.worker_urls = worker_urls
        load_balancer_instance.worker_iter = itertools.cycle(worker_urls)
        uvicorn.run(lb_app, host=args.host, port=args.lb_port, reload=args.dev)
    elif args.mode == "combined":
        print("[Combined mode] Running Autoscaler and Load Balancer concurrently with GPU assignment")
        processes = []
        for i in range(args.workers):
            port = args.base_port + i
            gpu_id = gpu_ids[i % num_gpus]  # Round-robin assignment.
            p = multiprocessing.Process(target=run_worker, args=(args.host, port, gpu_id, args.dev))
            p.start()
            processes.append(p)
        worker_urls = [f"http://{args.host}:{args.base_port + i}" for i in range(args.workers)]
        load_balancer_instance.worker_urls = worker_urls
        load_balancer_instance.worker_iter = itertools.cycle(worker_urls)
        lb_thread = threading.Thread(
            target=lambda: uvicorn.run(lb_app, host=args.host, port=args.lb_port, reload=args.dev),
            daemon=True
        )
        lb_thread.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("KeyboardInterrupt: shutting down combined mode...")
            for p in processes:
                p.terminate()

if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # Ignore if already set.
        pass
    main()