# src/batcher.py
import time
import threading
from concurrent.futures import Future
from rag_pipeline import rag_pipeline
from utils import MAX_BATCH_SIZE, MAX_WAITING_TIME, logger

class Batcher:
    def __init__(self, request_queue):
        self.request_queue = request_queue
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while True:
            batch_items = self._collect_batch()
            if not batch_items:
                continue

            self._process_batch(batch_items)

    def _collect_batch(self):
        batch_items = []
        start_time = time.time()

        try:
            first_item = self.request_queue.dequeue(timeout=MAX_WAITING_TIME)
            batch_items.append(first_item)
        except Exception:
            return []

        request_payload = first_item[0]
        batch_size = request_payload.batch_size or MAX_BATCH_SIZE

        while (time.time() - start_time) < MAX_WAITING_TIME and len(batch_items) < batch_size:
            try:
                timeout = MAX_WAITING_TIME - (time.time() - start_time)
                item = self.request_queue.dequeue(timeout=timeout)
                batch_items.append(item)
            except Exception:
                break

        return batch_items

    def _process_batch(self, batch_items):
        batch_start = time.time()

        queries = [req.query for req, _ in batch_items]
        k = batch_items[0][0].k
        mode = batch_items[0][0].mode

        logger.info(f"Processing batch of {len(batch_items)} requests in mode: {mode}")

        if mode == "batch":
            results = rag_pipeline(queries, k=k, mode="batch")
        else:
            results = [rag_pipeline(q, k=k, mode="single") for q in queries]

        batch_end = time.time()

        for (request, future), result in zip(batch_items, results):
            if not isinstance(result, dict):
                result = {"result": result, "metrics": {}}

            future.set_result({
                "result": result.get("result"),
                "metrics": result.get("metrics"),
                "server_start_time": batch_start,
                "server_end_time": batch_end
            })