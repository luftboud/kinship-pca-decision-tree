import csv
from pathlib import Path

import numpy as np

from decision_tree_classifier import train_decision_tree_from_pairs
from pca import compute_pca_embeddings
from preprocess import preprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FACES_ROOT = PROJECT_ROOT / "data/test-public-faces/test-public-faces"
TRAIN_RELATIONSHIPS = PROJECT_ROOT / "data/train_relationships.csv"


def preprocess_person_dir(person_dir, target_size=(64, 64)):
    if not person_dir.exists() or not person_dir.is_dir():
        return None
    vectors = preprocess(str(person_dir), target_size=target_size)
    if len(vectors) == 0:
        return None

    return np.mean(vectors, axis=0)


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


def build_negative_pairs(person_dirs, amount, rng):
    family_groups = {}
    for person_dir in person_dirs:
        rel = person_dir.relative_to(FACES_ROOT)
        family_id = rel.parts[0]
        family_groups.setdefault(family_id, []).append(person_dir)

    family_ids = sorted(family_groups.keys())
    negatives = []
    while len(negatives) < amount:
        f1, f2 = rng.choice(family_ids, size=2, replace=False)
        p1 = family_groups[f1][int(rng.integers(0, len(family_groups[f1])))]
        p2 = family_groups[f2][int(rng.integers(0, len(family_groups[f2])))]
        negatives.append((p1, p2))

    return negatives


def main():
    rng = np.random.default_rng(42)

    positive_pairs, candidate_person_dirs = load_relationship_pairs(TRAIN_RELATIONSHIPS, FACES_ROOT)

    vectors = {}
    for person_dir in candidate_person_dirs:
        vec = preprocess_person_dir(person_dir)
        if vec is not None:
            vectors[person_dir] = vec

    positive_pairs = [(a, b) for a, b in positive_pairs if a in vectors and b in vectors]
    person_dirs = sorted(vectors.keys())
    image_matrix = np.array([vectors[p] for p in person_dirs], dtype=np.float32)
    embeddings = compute_pca_embeddings(image_matrix, max_features_amount=20)

    image_to_idx = {p: i for i, p in enumerate(person_dirs)}
    pos_idx_pairs = [(image_to_idx[a], image_to_idx[b]) for a, b in positive_pairs]
    neg_path_pairs = build_negative_pairs(person_dirs, amount=len(pos_idx_pairs), rng=rng)
    neg_idx_pairs = [(image_to_idx[a], image_to_idx[b]) for a, b in neg_path_pairs]

    pairs = pos_idx_pairs + neg_idx_pairs
    labels = [1] * len(pos_idx_pairs) + [0] * len(neg_idx_pairs)

    _, accuracy = train_decision_tree_from_pairs(embeddings, pairs, labels)

    print(f"Usable preprocessed identities: {len(person_dirs)}")
    print(f"Positive pairs: {len(pos_idx_pairs)}")
    print(f"Negative pairs: {len(neg_idx_pairs)}")
    print(f"Decision Tree accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()