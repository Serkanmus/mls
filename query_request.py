import requests
import time

# A list of example queries (you can adjust this as you like)
queries = [
    "What is a cat?",
    "Which animals can hover in the air?",
    "Tell me about dogs."
]

num_requests = 30  # total requests to send
times = []

for i in range(num_requests):
    # Pick one query (could alternate using modulo)
    query = queries[i % len(queries)]
    payload = {"query": query, "k": 2}
    
    start_time = time.time()
    response = requests.post("http://localhost:8000/rag", json=payload)
    end_time = time.time()
    
    spent = end_time - start_time
    times.append(spent)
    
    # Print the time spent on this single request
    print(f"Request {i} took {spent:.4f} seconds. Status Code: {response.status_code}")

# Compute and print the average
average_time = sum(times) / len(times)
print(f"\nAverage time for {num_requests} requests: {average_time:.4f} seconds.")
