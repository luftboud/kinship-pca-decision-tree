import csv
import numpy as np

from pca import compute_pca_embeddings
from preprocess import preprocess_dir
from constants import *

def l2_normalize_rows(matrix):
    norms = np.sqrt(np.sum(matrix ** 2, axis=1, keepdims=True))
    norms[norms == 0] = 1.0
    return matrix / norms

def preprocess_img_dir(person_dir, target_size=IMG_SIZE):
    if not person_dir.exists() or not person_dir.is_dir():
        return []

    vectors = preprocess_dir(str(person_dir), target_size=target_size)
    if len(vectors) == 0:
        return []

    return vectors


def load_relationship_pairs(relationships_file, faces_root):
    positive_pairs = []
    person_dirs_set = set()

    with relationships_file.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p1_dir = faces_root / row["p1"].strip("/")
            p2_dir = faces_root / row["p2"].strip("/")

            if p1_dir.exists() and p2_dir.exists():
                positive_pairs.append((p1_dir, p2_dir))
                person_dirs_set.add(p1_dir)
                person_dirs_set.add(p2_dir)

    return positive_pairs, sorted(person_dirs_set)


def build_positive_photo_pairs(positive_person_pairs, person_to_photo_indices, rng, max_pairs_per_relation=10):
    positive_photo_pairs = []

    for p1, p2 in positive_person_pairs:
        imgs1 = person_to_photo_indices[p1]
        imgs2 = person_to_photo_indices[p2]

        all_pairs = [(i, j) for i in imgs1 for j in imgs2]

        if len(all_pairs) > max_pairs_per_relation:
            chosen = rng.choice(len(all_pairs), size=max_pairs_per_relation, replace=False)
            sampled_pairs = [all_pairs[k] for k in chosen]
        else:
            sampled_pairs = all_pairs

        positive_photo_pairs.extend(sampled_pairs)

    return positive_photo_pairs


def build_negative_photo_pairs(person_to_photo_indices, amount, rng, faces_root, embeddings,
                               candidate_multiplier=20, hard_fraction=0.1):
    family_groups = {}

    for person_path, photo_indices in person_to_photo_indices.items():
        rel = person_path.relative_to(faces_root)
        family_id = rel.parts[0]
        family_groups.setdefault(family_id, []).append((person_path, photo_indices))

    family_ids = sorted(family_groups.keys())

    def sample_random_negative():
        f1, f2 = rng.choice(family_ids, size=2, replace=False)

        _, imgs1 = family_groups[f1][int(rng.integers(0, len(family_groups[f1])))]
        _, imgs2 = family_groups[f2][int(rng.integers(0, len(family_groups[f2])))]

        i = imgs1[int(rng.integers(0, len(imgs1)))]
        j = imgs2[int(rng.integers(0, len(imgs2)))]

        return (i, j) if i < j else (j, i)

    num_hard = int(amount * hard_fraction)
    num_random = amount - num_hard

    negatives = set()
    while len(negatives) < num_random:
        negatives.add(sample_random_negative())

    candidate_target = max(num_hard * candidate_multiplier, num_hard)
    candidate_pairs = set()

    while len(candidate_pairs) < candidate_target:
        pair = sample_random_negative()
        if pair not in negatives:
            candidate_pairs.add(pair)

    scored = []
    for i, j in candidate_pairs:
        sim = float(np.dot(embeddings[i], embeddings[j]))
        scored.append((sim, (i, j)))

    scored.sort(key=lambda x: x[0], reverse=True)

    if len(scored) > 0 and num_hard > 0:
        top_k = max(len(scored) // 3, num_hard)
        pool = scored[:top_k]

        chosen_idx = rng.choice(len(pool), size=min(num_hard, len(pool)), replace=False)
        for k in chosen_idx:
            negatives.add(pool[k][1])

    while len(negatives) < amount:
        negatives.add(sample_random_negative())

    return list(negatives)


def get_prepared_train_data(relationships_file, faces_img_root):
    rng = np.random.default_rng(SEED)

    positive_person_pairs, candidate_person_dirs = load_relationship_pairs(relationships_file, faces_img_root)

    person_to_photo_indices = {}
    all_images = []

    for person_dir in candidate_person_dirs:
        photo_vectors = preprocess_img_dir(person_dir)

        if len(photo_vectors) == 0:
            continue

        indices = []
        for vec in photo_vectors:
            idx = len(all_images)
            all_images.append(vec)
            indices.append(idx)

        person_to_photo_indices[person_dir] = indices

    positive_person_pairs = [(a, b) for a, b in positive_person_pairs
                             if a in person_to_photo_indices and b in person_to_photo_indices]

    if len(all_images) == 0:
        raise ValueError("No training images were loaded.")

    image_matrix = np.asarray(all_images, dtype=np.float32)

    centering = np.mean(image_matrix, axis=0, keepdims=True)
    image_matrix = image_matrix - centering

    wk = compute_pca_embeddings(image_matrix, max_features_amount=FEATURES_AMOUNT)
    embeddings = image_matrix @ wk
    embeddings = l2_normalize_rows(embeddings)

    pos_idx_pairs = build_positive_photo_pairs(
        positive_person_pairs,
        person_to_photo_indices,
        rng,
        max_pairs_per_relation=10
    )

    neg_idx_pairs = build_negative_photo_pairs(
        person_to_photo_indices,
        amount=len(pos_idx_pairs),
        rng=rng,
        faces_root=faces_img_root,
        embeddings=embeddings
    )

    pairs = pos_idx_pairs + neg_idx_pairs
    labels = [1] * len(pos_idx_pairs) + [0] * len(neg_idx_pairs)

    return pairs, labels, embeddings, wk, centering