#!/usr/bin/env python
import os
import time
import argparse
import multiprocessing
import threading
import uvicorn
import itertools
from fastapi import Request
from utils import get_ip, update_env_variable
# Import load_balancer components (including the lb instance and metric functions)
from load_balancer import lb as load_balancer_instance, app as lb_app, get_and_reset_request_count
from app_module import app

def run_worker(host: str, port: int, gpu_id: str, dev: bool):
    # Set the GPU environment variable before any other imports occur.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"Worker starting on {host}:{port} using GPU {gpu_id}")
    uvicorn.run(app, host=host, port=port, reload=dev)

def update_load_balancer_urls(worker_urls):
    load_balancer_instance.worker_urls = worker_urls
    load_balancer_instance.worker_iter = itertools.cycle(worker_urls)

def dynamic_autoscaler(args, gpu_ids):
    """
    Dynamic autoscaler thread function that monitors workload demand 
    (via request count) and adjusts the number of active inference workers.
    """
    # Configuration parameters for dynamic scaling.
    min_workers = 1
    max_workers = len(gpu_ids)  # Maximum workers limited to available GPUs.
    scale_out_threshold = 10    # If more than 10 requests arrive in the monitoring interval, scale out.
    scale_in_threshold = 2      # If fewer than 2 requests in the interval, scale in.
    monitoring_interval = 5     # Interval in seconds for checking workload.

    worker_processes = []       # List of tuples: (process, port)
    next_port = args.base_port  # Start assigning ports from base_port.

    def spawn_worker():
        nonlocal next_port
        gpu_id = gpu_ids[len(worker_processes) % len(gpu_ids)]
        port = next_port
        next_port += 1
        p = multiprocessing.Process(target=run_worker, args=(args.host, port, gpu_id, args.dev))
        p.start()
        worker_processes.append((p, port))
        print(f"Spawned worker on port {port} with GPU {gpu_id}")
        # Update the load balancer's URL list.
        urls = [f"http://{args.host}:{p_port}" for (_, p_port) in worker_processes]
        update_load_balancer_urls(urls)

    def kill_worker():
        if worker_processes:
            p, port = worker_processes.pop()
            print(f"Terminating worker on port {port}")
            p.terminate()
            p.join()
            # Update the load balancer's URL list.
            urls = [f"http://{args.host}:{p_port}" for (_, p_port) in worker_processes]
            update_load_balancer_urls(urls)

    # Start with the minimum number of workers.
    while len(worker_processes) < min_workers:
        spawn_worker()

    while True:
        time.sleep(monitoring_interval)
        req_count = get_and_reset_request_count()
        print(f"[Autoscaler] In the last {monitoring_interval} seconds, received {req_count} requests.")
        if req_count > scale_out_threshold and len(worker_processes) < max_workers:
            print("[Autoscaler] High load detected; scaling out.")
            spawn_worker()
        elif req_count < scale_in_threshold and len(worker_processes) > min_workers:
            print("[Autoscaler] Low load detected; scaling in.")
            kill_worker()
        else:
            print("[Autoscaler] No scaling action required.")

def main():
    parser = argparse.ArgumentParser(
        description="Serving_rag launcher with dynamic autoscaling and GPU assignment"
    )
    parser.add_argument("--dev", action="store_true", help="Development mode (enable reload)")
    parser.add_argument("--mode", type=str, choices=["worker", "autoscale", "loadbalancer", "combined"],
                        default="worker", help="Select execution mode")
    parser.add_argument("--host", type=str, default=None, help="IP address to bind the server (auto-detected if not provided)")
    parser.add_argument("--base_port", type=int, default=8777, help="Starting port number for worker servers")
    # In dynamic autoscale mode, start with the minimum number of workers.
    parser.add_argument("--workers", type=int, default=1, help="Initial number of worker processes in dynamic autoscale mode")
    parser.add_argument("--lb_port", type=int, default=8888, help="Load Balancer port number")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", 
                        help="Comma-separated list of GPU IDs to assign to worker processes")
    args = parser.parse_args()

    gpu_ids = [gpu.strip() for gpu in args.gpu_ids.split(",") if gpu.strip()]
    if len(gpu_ids) == 0:
        raise ValueError("No GPU IDs provided via --gpu_ids")

    if args.host is None:
        args.host = get_ip(args.dev)
    update_env_variable("HOST_IP", args.host)
    print(f"Server Host: {args.host}")

    if args.mode == "worker":
        gpu_id = gpu_ids[0]
        print(f"[Worker mode] Running a single worker on {args.host}:{args.base_port} with GPU {gpu_id}")
        run_worker(args.host, args.base_port, gpu_id, args.dev)
    elif args.mode == "autoscale":
        # Legacy static autoscale: spawn a fixed number of workers.
        print(f"[Autoscale mode] Starting {args.workers} worker(s) on ports {args.base_port} to {args.base_port + args.workers - 1}")
        processes = []
        for i in range(args.workers):
            port = args.base_port + i
            gpu_id = gpu_ids[i % len(gpu_ids)]
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
        # In loadbalancer mode, assume workers are launched separately.
        worker_urls = [f"http://{args.host}:{args.base_port + i}" for i in range(args.workers)]
        update_load_balancer_urls(worker_urls)
        uvicorn.run(lb_app, host=args.host, port=args.lb_port, reload=args.dev)
    elif args.mode == "combined":
        print("[Combined mode] Running dynamic autoscaler with load balancer concurrently")
        # Launch the dynamic autoscaler in a daemon thread.
        autoscaler_thread = threading.Thread(target=dynamic_autoscaler, args=(args, gpu_ids), daemon=True)
        autoscaler_thread.start()
        # Start the load balancer (which is responsible for routing requests to current workers).
        uvicorn.run(lb_app, host=args.host, port=args.lb_port, reload=args.dev)

if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # Ignore if already set.
        pass
    main()
