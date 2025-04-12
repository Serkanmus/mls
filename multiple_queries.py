import requests
import threading
import time
import statistics
import numpy as np

# Adjust these values to your liking:
NUM_PARALLEL_REQUESTS = 50
REQUEST_PAYLOAD = {"query": "Which animals can hover in the air?", "k": 2}

# Shared lists/variables for recording metrics.
response_times = []
error_count = 0
lock = threading.Lock()

def send_request(payload):
    global error_count
    start_time = time.time()
    try:
        # Modify the URL as needed; here we use port 8147 as an example.
        # response = requests.post("http://localhost:8147/rag", json=payload)
        response = requests.post("http://129.215.18.53:8100/rag", json=payload)

    except Exception as e:
        elapsed = time.time() - start_time
        with lock:
            response_times.append(elapsed)
            error_count += 1
        print(f"Request exception: {e}, took {elapsed:.3f}s")
        return

    elapsed = time.time() - start_time
    try:
        res_json = response.json()
    except Exception as e:
        with lock:
            response_times.append(elapsed)
        print(f"Failed to parse JSON response: {response.text}")
        return

    # Print the response, then record the elapsed time.
    if "result" in res_json:
        print(f"Finished request in {elapsed:.3f}s. Result: {res_json['result'][:50]}...")
    elif "error" in res_json:
        print(f"Finished request in {elapsed:.3f}s. Error: {res_json['error']}")
    else:
        print(f"Finished request in {elapsed:.3f}s. Unexpected response: {res_json}")
    
    with lock:
        response_times.append(elapsed)

threads = []
for i in range(NUM_PARALLEL_REQUESTS):
    t = threading.Thread(target=send_request, args=(REQUEST_PAYLOAD,))
    threads.append(t)

start_all = time.time()
for t in threads:
    t.start()

for t in threads:
    t.join()
end_all = time.time()

total_time = end_all - start_all

# Calculate additional statistics from the collected response times.
if response_times:
    average_response_time = sum(response_times) / len(response_times)
    min_response_time = min(response_times)
    max_response_time = max(response_times)
    median_response_time = statistics.median(response_times)
    std_response_time = statistics.stdev(response_times) if len(response_times) > 1 else 0
    p95_response_time = np.percentile(response_times, 95)
else:
    average_response_time = min_response_time = max_response_time = median_response_time = std_response_time = p95_response_time = 0

# Display the gathered metrics.
print(f"\nSent {NUM_PARALLEL_REQUESTS} requests in parallel. Total wall time: {total_time:.3f}s.")
print(f"Average response time per request: {average_response_time:.3f}s")
print(f"Minimum response time: {min_response_time:.3f}s")
print(f"Median response time: {median_response_time:.3f}s")
print(f"Standard deviation of response times: {std_response_time:.3f}s")
print(f"95th percentile response time: {p95_response_time:.3f}s")
print(f"Maximum response time: {max_response_time:.3f}s")
print(f"Number of errors: {error_count}")
