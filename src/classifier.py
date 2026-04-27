import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from similarity import build_pair_feature_matrix
from constants import *


def fit_standard_scaler(X):
    mean = np.mean(X, axis=0)
    std = np.sqrt(np.mean((X - mean) ** 2, axis=0))
    std[std == 0] = 1.0
    return mean, std


def transform_standard_scaler(X, mean, std):
    return (X - mean) / std


def _split_by_family_disjointness(train_pairs, train_labels, family_ids, val_fraction=0.2):
    rng = np.random.default_rng(SEED)
    families = np.array(sorted(set(family_ids)))

    if len(families) < 2:
        idx = np.arange(len(train_labels))
        return train_test_split(idx, test_size=val_fraction, random_state=SEED, stratify=train_labels)

    n_val_families = max(1, int(round(len(families) * val_fraction)))
    val_family_set = set(rng.choice(families, size=n_val_families, replace=False).tolist())

    train_idx = []
    val_idx = []

    for sample_idx, (i, j) in enumerate(train_pairs):
        fi = family_ids[i]
        fj = family_ids[j]
        if fi in val_family_set or fj in val_family_set:
            val_idx.append(sample_idx)
        else:
            train_idx.append(sample_idx)

    y = np.asarray(train_labels, dtype=int)
    if len(train_idx) == 0 or len(val_idx) == 0:
        idx = np.arange(len(train_labels))
        return train_test_split(idx, test_size=val_fraction, random_state=SEED, stratify=train_labels)

    train_y = y[np.asarray(train_idx)]
    val_y = y[np.asarray(val_idx)]
    if len(np.unique(train_y)) < 2 or len(np.unique(val_y)) < 2:
        idx = np.arange(len(train_labels))
        return train_test_split(idx, test_size=val_fraction, random_state=SEED, stratify=train_labels)

    return np.asarray(train_idx), np.asarray(val_idx)


def train_model(embeddings, train_pairs, train_labels, family_ids):
    x_all = build_pair_feature_matrix(embeddings, train_pairs)
    y_all = np.asarray(train_labels, dtype=int)
    mean, std = fit_standard_scaler(x_all)
    x_all_scaled = transform_standard_scaler(x_all, mean, std)

    train_idx, val_idx = _split_by_family_disjointness(train_pairs, train_labels, family_ids, val_fraction=0.2)
    x_train_scaled = x_all_scaled[train_idx]
    y_train = y_all[train_idx]
    y_val = y_all[val_idx]

    x_train_raw = x_all[train_idx]
    x_val_raw = x_all[val_idx]

    assert len(x_train_scaled) == len(y_train)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y_train))
    x_train_raw = x_train_raw[perm]
    y_train = y_train[perm]

    print("Train: fitting Random Forest...")
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=SEED
    )
    model.fit(x_train_raw, y_train)
    train_pred = model.predict(x_train_raw)
    val_pred = model.predict(x_val_raw)

    print("Train accuracy:", accuracy_score(y_train, train_pred))
    print("Validation accuracy (family-disjoint):", accuracy_score(y_val, val_pred))

    if hasattr(model, "oob_score_"):
        print("Random Forest OOB score:", model.oob_score_)

    return model, mean, std


def test_model(model, mean, std, embeddings, test_pairs, test_labels):
    x_test = build_pair_feature_matrix(embeddings, test_pairs)
    y_test = np.array(test_labels, dtype=int)

    assert len(x_test) == len(y_test)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    return accuracy, report, matrix
