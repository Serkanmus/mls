import os
import torch
import logging
import socket

# ----------------------------
# IP configuration
# ----------------------------
def get_ip(dev_mode):
    if not dev_mode:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception as e:
            print("Failed to get local IP:", e)
            return None
    return "0.0.0.0"

def update_env_variable(key: str, value: str, env_path: str = ".env"):
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip().strip('"')
    
    env_vars[key] = value

    with open(env_path, "w", encoding="utf-8") as f:
        for k, v in env_vars.items():
            f.write(f'{k}="{v}"\n')

# ----------------------------
# Device configuration
# ----------------------------
def get_device():
    # Evaluate the available device at the time of the call.
    if torch.cuda.is_available():
         return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
         return torch.device("mps")
    else:
         return torch.device("cpu")

print(f"Using device: {get_device().type}")

# ----------------------------
# Logging configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------------------
# Batching hyperparameters
# ----------------------------
MAX_BATCH_SIZE = 10        # Maximum number of requests per batch
MAX_WAITING_TIME = 0.5     # Maximum wait time (in seconds) before processing a batch
