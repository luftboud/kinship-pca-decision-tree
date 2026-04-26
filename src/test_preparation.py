import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

from preprocess import _get_dlib, _get_cache, _save_cache, _compute_embedding
from constants import *


def l2_normalize_rows(matrix):
    norms = np.sqrt(np.sum(matrix ** 2, axis=1, keepdims=True))
    norms[norms == 0] = 1.0
    return matrix / norms


def _batch_embed_paths(paths):
    """Compute dlib embeddings for a list of paths using cache + 6 threads."""
    cache = _get_cache()
    to_compute = [p for p in paths if p not in cache]

    if to_compute:
        total = len(to_compute)
        done = [0]
        lock = threading.Lock()
        step = max(1, total // 100)

        def compute_and_track(p):
            emb = _compute_embedding(p)
            with lock:
                done[0] += 1
                if done[0] % step == 0 or done[0] == total:
                    pct = done[0] * 100 // total
                    print(f"\rprogress: {pct}%", end="", flush=True)
            return p, emb

        with ThreadPoolExecutor(max_workers=12) as ex:
            for path, emb in ex.map(compute_and_track, to_compute):
                cache[path] = emb

        print()
        _save_cache(cache)
    else:
        print("progress: 100% (all cached)")

    return {p: cache[p] for p in paths}


def create_test_relations():
    pair_entries = []

    for root, dirs, files in os.walk(SPLIT_TEST_RELATIONS):
        for file in sorted(files):
            with open(Path(root) / file, "r") as inner_relations, \
                 open(TEST_FACES_LABELS / file) as inner_label:
                for rel, lbl in zip(inner_relations, inner_label):
                    if rel == "p1,p2\n":
                        continue
                    f1, f2 = rel.split(',')
                    f1 = TEST_FACES_ROOT / f1.strip()
                    f2 = TEST_FACES_ROOT / f2.strip()
                    label = int(lbl.strip())
                    pair_entries.append((f1, f2, label))

    unique_paths = sorted({str(p) for entry in pair_entries for p in entry[:2]})
    print(f"Test: {len(unique_paths)} unique images, computing embeddings...")
    embeddings_map = _batch_embed_paths(unique_paths)

    imgs = {Path(p): embeddings_map[p] for p in unique_paths}

    positive_pairs = set()
    negative_pairs = set()
    for f1, f2, label in pair_entries:
        if f1 not in imgs or f2 not in imgs:
            continue
        if label == 1:
            positive_pairs.add((f1, f2))
        else:
            negative_pairs.add((f1, f2))

    return imgs, positive_pairs, negative_pairs


def get_prepared_test_data(wk, centering):
    imgs, positive_pairs, negative_pairs = create_test_relations()

    person_dirs = sorted(imgs.keys())
    image_matrix = np.array([imgs[p] for p in person_dirs], dtype=np.float32)
    image_matrix = image_matrix - centering
    test_embeddings = image_matrix @ wk
    test_embeddings = l2_normalize_rows(test_embeddings)

    image_to_idx = {p: i for i, p in enumerate(person_dirs)}
    pos_idx_pairs = [(image_to_idx[a], image_to_idx[b]) for a, b in positive_pairs]
    neg_idx_pairs = [(image_to_idx[a], image_to_idx[b]) for a, b in negative_pairs]

    pairs = pos_idx_pairs + neg_idx_pairs
    labels = [1] * len(pos_idx_pairs) + [0] * len(neg_idx_pairs)

    print(f"Test: {len(pos_idx_pairs)} positive, {len(neg_idx_pairs)} negative pairs")
    return pairs, labels, test_embeddings
