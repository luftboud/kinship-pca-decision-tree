import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from similarity import build_pair_feature_matrix


def train_decision_tree_from_pairs(embeddings, pairs, labels, test_size=0.2, random_state=42):
    x = build_pair_feature_matrix(embeddings, pairs)
    y = np.asarray(labels, dtype=int)

    if len(x) != len(y):
        raise ValueError(f"Pairs and labels count mismatch: {len(x)} vs {len(y)}")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    param_grid = {
        "max_depth": [2, 3, 4, 5, 6, 8, 10, None],
        "min_samples_leaf": [1, 2, 4, 8, 16],
        "min_samples_split": [2, 4, 8, 16],
        "criterion": ["gini", "entropy", "log_loss"],
    }

    search = GridSearchCV(
        DecisionTreeClassifier(random_state=random_state),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="accuracy",
    )
    search.fit(x_train, y_train)
    model = search.best_estimator_

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy