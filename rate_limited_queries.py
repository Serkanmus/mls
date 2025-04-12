import requests
import threading
import time
import statistics
import numpy as np
import socket

# Parameters
REQUESTS_PER_SECOND = 50       # Send 50 requests each second
DURATION_SECONDS = 20          # Total duration (in seconds) to send requests
TOTAL_REQUESTS = REQUESTS_PER_SECOND * DURATION_SECONDS  # 50 * 20 = 1000 requests

# Request payload (modify as needed)
REQUEST_PAYLOAD = {"query": "Which animals can hover in the air?", "k": 2}

# Shared variables for metrics
response_times = []  
error_count = 0
lock = threading.Lock()  # To protect updates to shared variables
threads = []  # List to store thread references

def get_my_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

# Use the dynamically determined IP in your URL:
SERVER_IP = get_my_ip()
# SERVER_URL = f"http://{SERVER_IP}:8100/rag"
SERVER_URL = f"http://{SERVER_IP}:8147/rag"

def send_request(payload):
    global error_count
    start_time = time.time()
    try:
        # Modify the URL as needed; here we use port 8147 as an example.
        # response = requests.post("http://localhost:8147/rag", json=payload)
        response = requests.post(SERVER_URL, json=payload)

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

start_all = time.time()

# Send requests at a constant rate: a new thread every (1/REQUESTS_PER_SECOND) seconds.
for i in range(TOTAL_REQUESTS):
    t = threading.Thread(target=send_request, args=(REQUEST_PAYLOAD,))
    t.start()
    threads.append(t)
    time.sleep(1.0 / REQUESTS_PER_SECOND)

# Wait for all threads to complete.
for t in threads:
    t.join()

total_time = time.time() - start_all

# Calculate statistics from response times
if response_times:
    avg_response_time = sum(response_times) / len(response_times)
    min_response_time = min(response_times)
    max_response_time = max(response_times)
    median_response_time = statistics.median(response_times)
    std_response_time = statistics.stdev(response_times) if len(response_times) > 1 else 0
else:
    avg_response_time = min_response_time = max_response_time = median_response_time = std_response_time = 0

# Display collected metrics.
print(f"\nSent {TOTAL_REQUESTS} requests over {DURATION_SECONDS} seconds. Total wall time: {total_time:.3f}s")
print(f"Average response time: {avg_response_time:.3f}s")
print(f"Minimum response time: {min_response_time:.3f}s")
print(f"Median response time: {median_response_time:.3f}s")
print(f"Standard deviation of response times: {std_response_time:.3f}s")
print(f"Maximum response time: {max_response_time:.3f}s")
print(f"Error count: {error_count}")
