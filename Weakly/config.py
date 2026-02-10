# config.py
import random
from pathlib import Path
import numpy as np
import torch

# ---- Repro / device ----
SEED = 1337

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # reproducible training (slower sometimes)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Paths ----
ROOT_DIR = Path("/home/g202417400/lidc-kaggle/kaggle/archive/LIDC-IDRI-slices")
OUT_DIR  = Path("/home/g202417400/Msproject/MSPO-Net/semisup_runs_dw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Training knobs ----
BATCH_SIZE   = 16
LR           = 1e-5
MAX_EPOCHS   = 100
PATIENCE     = 10
VAL_FRAC_T   = 0.10
VAL_FRAC_S   = 0.10
SPLIT_5_3_2  = (5, 3, 2)
NUM_WORKERS  = 4
IMAGE_SIZE   = (256, 256)   # (H, W) or None


