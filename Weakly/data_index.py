# data_index.py
import re
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

IMG_EXTS = {".png", ".jpg", ".jpeg"}
MSK_EXTS = IMG_EXTS

_num_pat = re.compile(r"(?:slice[-_]?|mask[-_]?|^)(\d+)(?=\.[^.]+$)")

def _idx_from_name(name: str) -> Optional[int]:
    m = _num_pat.search(name.lower())
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    digs = re.findall(r"(\d+)", name)
    return int(digs[-1]) if digs else None

def find_image_dir(nodule_dir: Path) -> Optional[Path]:
    for d in nodule_dir.iterdir():
        if d.is_dir() and ("image" in d.name.lower() or d.name.lower() in {"imgs","img","images"}):
            return d
    return None

def find_mask_dirs(nodule_dir: Path) -> List[Path]:
    out = []
    for d in nodule_dir.iterdir():
        if d.is_dir() and ("mask" in d.name.lower() or d.name.lower() in {"labels","label","annotation","annotations","masks"}):
            out.append(d)
    return out

def collect_pairs(root: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for patient in sorted(root.iterdir()):
        if not patient.is_dir() or not patient.name.upper().startswith("LIDC-IDRI-"):
            continue

        nodules = [d for d in patient.iterdir() if d.is_dir() and d.name.lower().startswith("nodule")]
        for nodule in sorted(nodules):
            img_dir = find_image_dir(nodule)
            if img_dir is None:
                continue

            mask_dirs = find_mask_dirs(nodule)
            mask_files = []
            for md in mask_dirs:
                mask_files += [p for p in md.rglob("*") if p.is_file() and p.suffix.lower() in MSK_EXTS]
            if not mask_files:
                continue

            masks_by_idx: Dict[int, Path] = {}
            for mp in mask_files:
                idx = _idx_from_name(mp.name)
                if idx is not None:
                    masks_by_idx.setdefault(idx, mp)

            img_files = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
            for ip in sorted(img_files):
                idx = _idx_from_name(ip.name)
                if idx is None:
                    continue
                if idx in masks_by_idx:
                    pairs.append((ip, masks_by_idx[idx]))

    if not pairs:
        raise RuntimeError("No (image, mask) pairs found under ROOT_DIR. Check your folders.")
    return pairs

def split_5_3_2(pairs: List[Tuple[Path, Path]], split=(5,3,2), seed: int = 1337):
    rng = random.Random(seed)
    pairs = pairs[:]
    rng.shuffle(pairs)

    n = len(pairs)
    a, b, c = split
    w = a + b + c

    nA = int(round(n * a / w))
    nB = int(round(n * b / w))

    A = pairs[:nA]            # Teacher train (coarse labels)
    B = pairs[nA:nA+nB]       # Student images (teacher pseudo-labels)
    C = pairs[nA+nB:]         # Test (fine labels)
    return A, B, C

def split_train_val(items, val_frac: float):
    n = len(items)
    n_val = max(1, int(round(n * val_frac))) if n > 5 else max(0, int(n * val_frac))
    if n_val <= 0:
        return items, []
    return items[n_val:], items[:n_val]
