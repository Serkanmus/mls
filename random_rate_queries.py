import requests
import threading
import time
import random
import statistics
import numpy as np

# Test parameters
TOTAL_REQUESTS = 1000         # Total number of requests to send
TOTAL_DURATION = 20           # Total duration (in seconds) over which to spread the requests

# Use a fixed seed to always send the same request schedule.
random.seed(42)

# Generate 1000 random arrival times uniformly in [0, TOTAL_DURATION]
arrival_times = [random.uniform(0, TOTAL_DURATION) for _ in range(TOTAL_REQUESTS)]
# Sort the arrival times so that they are in increasing order.
arrival_times.sort()

# Request payload (modify as needed)
REQUEST_PAYLOAD = {"query": "Which animals can hover in the air?", "k": 2}

# Shared variables for collecting metrics
response_times = []
error_count = 0
lock = threading.Lock()  # To protect shared state
threads = []  # To keep track of request threads

def send_request(payload):
    global error_count
    start_time = time.time()
    try:
        # response = requests.post("http://localhost:8100/rag", json=payload)
        response = requests.post("http://192.168.47.132:8100/rag", json=payload)

    except Exception as e:
        elapsed = time.time() - start_time
        with lock:
            response_times.append(elapsed)
            error_count += 1
        print(f"Request exception: {e}, took {elapsed:.3f}s")
        return

    elapsed = time.time() - start_time
    with lock:
        response_times.append(elapsed)
    if response.status_code == 200:
        # Print just a snippet of the result for brevity.
        print(f"Finished request in {elapsed:.3f}s. Result: {response.json()['result'][:50]}...")
    else:
        with lock:
            error_count += 1
        print(f"Error: {response.status_code}, took {elapsed:.3f}s")

def schedule_request(scheduled_time, payload):
    """Sleep until the scheduled time (relative to global start), then send request."""
    # Calculate how long to sleep from now until the desired start time.
    now = time.time()
    sleep_time = scheduled_time - (now - global_start_time)
    if sleep_time > 0:
        time.sleep(sleep_time)
    send_request(payload)

# Record the global start time.
global_start_time = time.time()

# Spawn a thread for each request based on its scheduled arrival time.
for scheduled_offset in arrival_times:
    t = threading.Thread(target=schedule_request, args=(scheduled_offset, REQUEST_PAYLOAD))
    t.start()
    threads.append(t)

# Wait for all threads to complete.
for t in threads:
    t.join()

total_time = time.time() - global_start_time

# Calculate statistics from response times.
if response_times:
    avg_response_time = sum(response_times) / len(response_times)
    min_response_time = min(response_times)
    max_response_time = max(response_times)
    median_response_time = statistics.median(response_times)
    std_response_time = statistics.stdev(response_times) if len(response_times) > 1 else 0
    p95_response_time = np.percentile(response_times, 95)
else:
    avg_response_time = min_response_time = max_response_time = median_response_time = std_response_time = p95_response_time = 0

# Display performance metrics.
print(f"\nSent {TOTAL_REQUESTS} requests over {TOTAL_DURATION} seconds. Total wall time: {total_time:.3f}s")
print(f"Average response time: {avg_response_time:.3f}s")
print(f"Minimum response time: {min_response_time:.3f}s")
print(f"Median response time: {median_response_time:.3f}s")
print(f"95th percentile response time: {p95_response_time:.3f}s")
print(f"Standard deviation of response times: {std_response_time:.3f}s")
print(f"Maximum response time: {max_response_time:.3f}s")
print(f"Error count: {error_count}")
