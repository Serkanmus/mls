# src/utils.py
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
    # 기존 .env 파일이 존재하면 읽어서 딕셔너리로 변환
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            # 빈 줄이나 주석 무시
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip().strip('"')
    
    # key 값 업데이트 또는 추가
    env_vars[key] = value

    # 업데이트된 딕셔너리를 .env 파일로 기록 (덮어쓰기)
    with open(env_path, "w", encoding="utf-8") as f:
        for k, v in env_vars.items():
            f.write(f'{k}="{v}"\n')

# ----------------------------
# Device configuration
# ----------------------------
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else torch.device("cpu")
)

print(f"Using device: {device.type}")

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