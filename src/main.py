import csv
from pathlib import Path
import os
import numpy as np

from decision_tree_classifier import train_decision_tree_from_pairs, test_decision_tree_classifier
from pca import compute_pca_embeddings
from preprocess import preprocess_dir, preprocess_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_FACES_ROOT = PROJECT_ROOT / "data" / "train-faces"
TRAIN_RELATIONSHIPS = PROJECT_ROOT / "data" / "train_relationships.csv"

TEST_FACES_ROOT = PROJECT_ROOT / "data" / "test-private-faces"
SPLIT_TEST_RELATIONS = PROJECT_ROOT / "data" / "test-private-lists"
TEST_FACES_RELATIONSHIPS = PROJECT_ROOT / "data" / "test_relationships.csv"
TEST_FACES_LABELS = PROJECT_ROOT / "data" / "test-private-labels"

SEED = 42
FEATURES_AMOUNT = 20
IMG_SIZE = (64, 64)


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


def preprocess_img(person_dir, target_size=IMG_SIZE):
    if not person_dir.exists() or not person_dir.is_dir():
        return None

    vectors = preprocess_dir(str(person_dir), target_size=target_size)
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


def build_negative_pairs(person_dirs, amount, rng, faces_root):
    family_groups = {}

    for person_path in person_dirs:
        rel = person_path.relative_to(faces_root)
        family_id = rel.parts[0]
        family_groups.setdefault(family_id, []).append(person_path)

    family_ids = sorted(family_groups.keys())
    negatives = set()

    while len(negatives) < amount:
        f1, f2 = rng.choice(family_ids, size=2, replace=False)
        p1 = family_groups[f1][int(rng.integers(0, len(family_groups[f1])))]
        p2 = family_groups[f2][int(rng.integers(0, len(family_groups[f2])))]
        pair = tuple(sorted((p1, p2)))
        negatives.add(pair)

    return negatives


def get_prepared_train_data(relationships_file, faces_img_root):
    rng = np.random.default_rng(SEED)

    positive_pairs, candidate_person_dirs = load_relationship_pairs(relationships_file, faces_img_root)

    vectors = {}
    for person_dir in candidate_person_dirs:
        vec = preprocess_img(person_dir)
        if vec is not None:
            vectors[person_dir] = vec

    positive_pairs = [(a, b) for a, b in positive_pairs if a in vectors and b in vectors]
    person_dirs = sorted(vectors.keys())
    image_matrix = np.array([vectors[p] for p in person_dirs], dtype=np.float32)

    centering = np.mean(image_matrix, axis=0)
    image_matrix = image_matrix - centering

    wk = compute_pca_embeddings(image_matrix, max_features_amount=FEATURES_AMOUNT)
    embeddings = image_matrix @ wk

    image_to_idx = {p: i for i, p in enumerate(person_dirs)}
    pos_idx_pairs = [(image_to_idx[a], image_to_idx[b]) for a, b in positive_pairs]
    neg_path_pairs = build_negative_pairs(person_dirs, amount=len(pos_idx_pairs), rng=rng, faces_root=faces_img_root)
    neg_idx_pairs = [(image_to_idx[a], image_to_idx[b]) for a, b in neg_path_pairs]

    pairs = pos_idx_pairs + neg_idx_pairs
    labels = [1] * len(pos_idx_pairs) + [0] * len(neg_idx_pairs)

    return pairs, labels, embeddings, wk, centering


def get_prepared_test_data(wk, centering):
    imgs, positive_pairs, negative_pairs = create_test_relations()

    person_dirs = sorted(imgs.keys())
    image_matrix = np.array([imgs[p] for p in person_dirs], dtype=np.float32)
    image_matrix = image_matrix - centering
    test_embeddings = image_matrix @ wk

    image_to_idx = {p: i for i, p in enumerate(person_dirs)}
    pos_idx_pairs = [(image_to_idx[a], image_to_idx[b]) for a, b in positive_pairs]
    neg_idx_pairs = [(image_to_idx[a], image_to_idx[b]) for a, b in negative_pairs]

    pairs = pos_idx_pairs + neg_idx_pairs
    labels = [1] * len(pos_idx_pairs) + [0] * len(neg_idx_pairs)

    return pairs, labels, test_embeddings


def main():
    pairs, labels, train_embeddings, wk, centering = get_prepared_train_data(TRAIN_RELATIONSHIPS, TRAIN_FACES_ROOT)
    model = train_decision_tree_from_pairs(train_embeddings, pairs, labels)

    test_pairs, test_labels, test_embeddings = get_prepared_test_data(wk, centering)
    accuracy, report, matrix = test_decision_tree_classifier(model, test_embeddings, test_pairs, test_labels)

    print(f"Decision Tree accuracy: {accuracy:.4f}")
    print(f"Decision Tree report:\n{report}")
    print(f"Decision Tree confusion matrix:\n{matrix}")

if __name__ == "__main__":
    main()