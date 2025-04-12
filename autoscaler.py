# autoscaler.py
import multiprocessing
import uvicorn
from app_module import app as worker_app  # app_module.py로부터 app 가져오기

def run_worker(host: str, port: int):
    """A function for single worker server."""
    uvicorn.run(worker_app, host=host, port=port)

class AutoScaler:
    def __init__(self, host: str, base_port: int, num_workers: int):
        self.host = host
        self.base_port = base_port
        self.num_workers = num_workers
        self.processes = []

    def start_workers(self):
        print(f"Starting {self.num_workers} worker(s) on ports {self.base_port} to {self.base_port + self.num_workers - 1}")
        for i in range(self.num_workers):
            port = self.base_port + i
            p = multiprocessing.Process(target=run_worker, args=(self.host, port))
            p.start()
            self.processes.append(p)

    def stop_workers(self):
        for p in self.processes:
            p.terminate()
            p.join()