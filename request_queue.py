# src/request_queue.py
import queue

class RequestQueue:
    """
    Thread-safe FIFO queue wrapper for batching RAG requests.
    """
    def __init__(self):
        self._queue = queue.Queue()

    def enqueue(self, item):
        """Add a new item to the queue."""
        self._queue.put(item)

    def dequeue(self, timeout=None):
        """Remove and return an item from the queue, blocking with optional timeout."""
        return self._queue.get(timeout=timeout)

    def get_nowait(self):
        """Attempt to get an item from the queue without blocking."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def empty(self):
        """Check if the queue is empty."""
        return self._queue.empty()
