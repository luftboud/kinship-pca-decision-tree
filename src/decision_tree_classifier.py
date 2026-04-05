import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from similarity import build_pair_feature_matrix


def check_leakage(pairs_train, pairs_test):
    train_indices = set(i for pair in pairs_train for i in pair)
    test_indices = set(i for pair in pairs_test for i in pair)

    intersection = train_indices.intersection(test_indices)
    if len(intersection) > 0:
        print(f"[WARNING] Possible leakage: {len(intersection)} shared identities between train and test")


def train_decision_tree_from_pairs(
    embeddings, pairs, labels, test_size=0.2, random_state=42
):
    x = build_pair_feature_matrix(embeddings, pairs)
    y = np.asarray(labels, dtype=int)

    if len(x) != len(y):
        raise ValueError(f"Pairs and labels count mismatch: {len(x)} vs {len(y)}")

    indices = np.arange(len(pairs))
    idx_train, idx_test = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )

    x_train, x_test = x[idx_train], x[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    pairs_train = [pairs[i] for i in idx_train]
    pairs_test = [pairs[i] for i in idx_test]

    check_leakage(pairs_train, pairs_test)

    param_grid = {
        "max_depth": [2, 3, 4, 5, 6, 8, 10, None],
        "min_samples_leaf": [1, 2, 4, 8, 16],
        "min_samples_split": [2, 4, 8, 16],
        "criterion": ["gini", "entropy", "log_loss"],
    }

    search = GridSearchCV(
        DecisionTreeClassifier(
            random_state=random_state,
            class_weight="balanced"
        ),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="accuracy",
    )

    search.fit(x_train, y_train)
    model = search.best_estimator_

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    return model, accuracy, report, matrix