import os
import numpy as np

from preprocess import preprocess_file
from constants import *


def l2_normalize_rows(matrix):
    norms = np.sqrt(np.sum(matrix ** 2, axis=1, keepdims=True))
    norms[norms == 0] = 1.0
    return matrix / norms

def create_test_relations():
    imgs = {}
    positive_pairs = set()
    negative_pairs = set()

    for root, dirs, files in os.walk(SPLIT_TEST_RELATIONS):
        for file in files:
            with open(Path(root) / file, "r") as inner_relations, open(TEST_FACES_LABELS / file) as inner_label:
                for (rel, lbl) in zip(inner_relations, inner_label):
                    if rel == "p1,p2\n": continue

                    f1, f2 = rel.split(',')
                    f1, f2 = TEST_FACES_ROOT / f1.strip(), TEST_FACES_ROOT / f2.strip()

                    if f1 not in imgs:
                        imgs[f1] = preprocess_file(f1, target_size=IMG_SIZE)
                    if f2 not in imgs:
                        imgs[f2] = preprocess_file(f2, target_size=IMG_SIZE)

                    if imgs[f1] is None or imgs[f2] is None:
                        continue

                    if int(lbl.strip()) == 1:
                        positive_pairs.add((f1, f2))
                        continue

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

    return pairs, labels, test_embeddings
