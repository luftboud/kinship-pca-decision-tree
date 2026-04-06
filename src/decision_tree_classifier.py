import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from similarity import build_pair_feature_matrix


def train_decision_tree_from_pairs(
    embeddings, train_pairs, train_labels, random_state=42
):
    x_train = build_pair_feature_matrix(embeddings, train_pairs)
    y_train = np.asarray(train_labels, dtype=int)

    if len(x_train) != len(y_train):
        raise ValueError(f"Train pairs and labels count mismatch: {len(x_train)} vs {len(y_train)}")

    tree = DecisionTreeClassifier(
        random_state=random_state,
        class_weight="balanced"
    )

    tree.fit(x_train, y_train)
    model = tree.tree_

    return model


def test_decision_tree_classifier(model, embeddings, test_pairs, test_labels):
    x_test = build_pair_feature_matrix(embeddings, test_pairs)
    y_test = np.array(test_labels, dtype=int)

    if len(x_test) != len(y_test):
        raise ValueError(f"Test pairs and labels count mismatch: {len(x_test)} vs {len(y_test)}")

    y_pred = np.argmax(model.predict(x_test), axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    return accuracy, report, matrix
