def dynamic_autoscaler(args, gpu_ids):
    """
    Dynamic autoscaler thread function that monitors workload demand 
    and adjusts the number of active workers accordingly. It now uses a dedicated
    counter for GPU assignment and waits longer for new workers to become healthy.
    """
    # Configuration parameters for dynamic scaling.
    min_workers = 1
    max_workers = len(gpu_ids)  # Maximum workers limited to available GPUs.
    scale_out_threshold = 10    # If more than 10 requests arrive within the monitoring interval, scale out.
    scale_in_threshold = 2      # If fewer than 2 requests in the interval, scale in.
    monitoring_interval = 5     # Interval (in seconds) for checking workload.
    
    worker_processes = []       # List of tuples: (process, port)
    next_port = args.base_port  # Start assigning ports from base_port.
    next_gpu_index = 0          # Counter for GPU assignment.
    
    def spawn_worker():
        nonlocal next_port, next_gpu_index
        # Use next_gpu_index to pick the GPU (round-robin) regardless of successful health checks.
        gpu_id = gpu_ids[next_gpu_index % len(gpu_ids)]
        next_gpu_index += 1  # Increment for the next spawn.
        port = next_port
        next_port += 1
        p = multiprocessing.Process(target=run_worker, args=(args.host, port, gpu_id, args.dev))
        p.start()
        new_worker_url = f"http://{args.host}:{port}"
        # Wait up to 60 seconds for the worker to report readiness.
        if wait_for_worker_ready(new_worker_url, timeout=60):
            worker_processes.append((p, port))
            print(f"Spawned worker on port {port} with GPU {gpu_id}")
            # Update the load balancer's URL list.
            urls = [f"http://{args.host}:{p_port}" for (_, p_port) in worker_processes]
            update_load_balancer_urls(urls)
        else:
            print(f"Worker on port {port} failed to become healthy; terminating.")
            p.terminate()
            p.join()
    
    def kill_worker():
        if worker_processes:
            p, port = worker_processes.pop()
            print(f"Terminating worker on port {port}")
            p.terminate()
            p.join()
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
