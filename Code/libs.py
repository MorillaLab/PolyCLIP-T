from __future__ import annotations


import os
import sys
import gc
import csv
import math
import time
import json
import socket
import random
import datetime
import traceback
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Any


import numpy as np
import pandas as pd
import scipy.sparse as sp

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler, Subset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# ML / sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Transformers (DNABERT2)
from transformers import AutoTokenizer, AutoModel, BertConfig

# FASTA
from pyfaidx import Fasta

# Giotto-TDA
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceImage


# DDP helpers
def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def is_rank0() -> bool:
    return get_rank() == 0


def rprint(*args, **kwargs):
    """Print only on rank 0 (DDP-safe)."""
    if is_rank0():
        print(*args, **kwargs)


# Reproducibility
def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
