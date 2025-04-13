import time
import requests
import json
import random
from statistics import mean
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Experiment configuration
# ----------------------------
NUM_REQUESTS = 1000
INTERVAL = 0
MAX_WORKERS = None
BATCH_SIZES = [2, 4, 8, 16, 32, 64]
SEND_PATTERNS = [ "spread", "random"]
DURATION = 20  # seconds to spread over for spread/random
HOST_IP = os.environ.get("HOST_IP")

URL = f"http://{HOST_IP}:8777/rag"
# URL = f"http://10.124.53.125:8777/rag"

BASE_PAYLOAD = {
    "query": "What is the capital of France?",
    "k": 3
}

# ----------------------------
# Request sender with timing capture
# ----------------------------
def send_request_with_timings(url, payload, results, index):
    client_send = time.time()
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        res = response.json()
    except Exception as e:
        print(f"Request {index} failed: {e}")
        results[index] = None
        return

    client_recv = time.time()
    server_start = res.get("server_start_time")
    server_end = res.get("server_end_time")
    metrics = res.get("metrics", {})

    results[index] = {
        "total_latency": client_recv - client_send,
        "queue_time": server_start - client_send if server_start else None,
        "processing_time": server_end - server_start if server_start and server_end else None,
        "send_time": client_send,
        "receive_time": client_recv,
        "metrics": metrics
    }

# ----------------------------
# Simulation runner for a batch of requests
# ----------------------------
def simulate_requests(payload, num_requests=NUM_REQUESTS, interval=INTERVAL, max_workers=MAX_WORKERS,
                      send_pattern="immediate", duration=DURATION):
    results = [None] * num_requests
    print(f"\n>>> Simulating {num_requests} requests to /rag (mode={payload.get('mode')}, batch_size={payload.get('batch_size')}), pattern={send_pattern}")

    def schedule_send(i, delay):
        time.sleep(delay)
        send_request_with_timings(URL, payload, results, i)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if send_pattern == "immediate":
            futures = [executor.submit(send_request_with_timings, URL, payload, results, i) for i in range(num_requests)]
        elif send_pattern == "spread":
            step = duration / num_requests
            futures = [executor.submit(schedule_send, i, i * step) for i in range(num_requests)]
        elif send_pattern == "random":
            futures = [executor.submit(schedule_send, i, random.uniform(0, duration)) for i in range(num_requests)]
        else:
            raise ValueError(f"Unknown send_pattern: {send_pattern}")

        for future in as_completed(futures):
            future.result()

    valid = [r for r in results if r is not None]
    latencies = [r["total_latency"] for r in valid]

    if valid:
        first = min(r["send_time"] for r in valid)
        last = max(r["receive_time"] for r in valid)
        real_duration = last - first
    else:
        real_duration = 0.0

    embed_times = [r["metrics"].get("embedding_time") for r in valid if r["metrics"].get("embedding_time") is not None]
    retrieval_times = [r["metrics"].get("retrieval_time") for r in valid if r["metrics"].get("retrieval_time") is not None]
    gen_times = [r["metrics"].get("generation_time") for r in valid if r["metrics"].get("generation_time") is not None]
    pipeline_times = [r["metrics"].get("total_time") for r in valid if r["metrics"].get("total_time") is not None]
    cpu_usages = [r["metrics"]["hardware"]["cpu_usage_percent"] for r in valid if "hardware" in r["metrics"]]
    mem_usages = [r["metrics"]["hardware"]["memory_usage_percent"] for r in valid if "hardware" in r["metrics"]]
    gpu_usages = [r["metrics"]["hardware"]["gpu_usage"]["gpu"] for r in valid
                  if "hardware" in r["metrics"] and r["metrics"]["hardware"]["gpu_usage"] is not None]
    net_sent = [r["metrics"]["hardware"]["network_sent_kBps"] for r in valid if "hardware" in r["metrics"]]
    net_recv = [r["metrics"]["hardware"]["network_recv_kBps"] for r in valid if "hardware" in r["metrics"]]

    print("\n--- Results ---")
    print(f"Success: {len(valid)}, Fail: {num_requests - len(valid)}")
    print(f"Real Total Time: {real_duration:.3f} sec")
    print(f"Average Latency: {mean(latencies):.3f} sec" if latencies else "Latency: N/A")
    print(f"Throughput: {len(valid) / real_duration:.2f} req/sec" if real_duration > 0 else "Throughput: N/A")
    if embed_times:
        print(f"Average Embedding Time: {mean(embed_times):.3f} sec")
    if retrieval_times:
        print(f"Average Retrieval Time: {mean(retrieval_times):.3f} sec")
    if gen_times:
        print(f"Average Generation Time: {mean(gen_times):.3f} sec")
    if pipeline_times:
        print(f"Average Pipeline Time: {mean(pipeline_times):.3f} sec")
    if cpu_usages:
        print(f"Avg CPU Usage: {mean(cpu_usages):.2f}%")
    if mem_usages:
        print(f"Avg Memory Usage: {mean(mem_usages):.2f}%")
    if gpu_usages:
        print(f"Avg GPU Utilization: {mean(gpu_usages):.2f}%")
    if net_sent:
        print(f"Avg Network Sent: {mean(net_sent):.2f} kB/s")
    if net_recv:
        print(f"Avg Network Recv: {mean(net_recv):.2f} kB/s")


    return {
        "avg_total_latency": mean(latencies) if latencies else None,
        "throughput": len(valid) / real_duration if real_duration > 0 else None,
        "success": len(valid),
        "fail": num_requests - len(valid),
        "real_total_time": real_duration,
        "avg_embedding_time": mean(embed_times) if embed_times else None,
        "avg_retrieval_time": mean(retrieval_times) if retrieval_times else None,
        "avg_generation_time": mean(gen_times) if gen_times else None,
        "avg_pipeline_time": mean(pipeline_times) if pipeline_times else None,
        "cpu_usages": f"{mean(cpu_usages):.2f}%" if cpu_usages else None,
        "mem_usages": f"{mean(mem_usages):.2f}%" if mem_usages else None,
        "gpu_usages": f"{mean(gpu_usages):.2f}%" if gpu_usages else None,
        "net_sent": f"{mean(net_sent):.2f} kB/s" if net_sent else None,
        "net_recv": f"{mean(net_recv):.2f} kB/s" if net_recv else None, 
        "batch_size": payload.get("batch_size"),
        "mode": payload.get("mode"),
        "send_pattern": send_pattern
    }

# ----------------------------
# Main experimental entrypoint
# ----------------------------
def run_experiments():
    summary = {}

    for pattern in SEND_PATTERNS:
        for batch_size in BATCH_SIZES:
            print("\n===================================")
            print(f"Running experiment: mode='batch', batch_size={batch_size}, pattern={pattern}")
            payload = BASE_PAYLOAD.copy()
            payload["mode"] = "batch"
            payload["batch_size"] = batch_size
            key = f"rag_batch_{batch_size}_{pattern}"
            summary[key] = simulate_requests(payload, send_pattern=pattern)

        print("\n===================================")
        print(f"Running experiment: mode='single', pattern={pattern}")
        payload = BASE_PAYLOAD.copy()
        payload["mode"] = "single"
        key = f"rag_single_{pattern}"
        summary[key] = simulate_requests(payload, send_pattern=pattern)

    print("\n=== Final Summary ===")
    for config, stats in summary.items():
        print(f"Config: {config}")
        for key, val in stats.items():
            print(f"  {key}: {val}")

    return summary

if __name__ == "__main__":
    results = run_experiments()
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)