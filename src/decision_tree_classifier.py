import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

from similarity import build_pair_feature_matrix
from src.constants import *


def fit_standard_scaler(X):
    mean = np.mean(X, axis=0)
    std = np.sqrt(np.mean((X - mean) ** 2, axis=0))
    std[std == 0] = 1.0
    return mean, std


def transform_standard_scaler(X, mean, std):
    return (X - mean) / std


def train_model(embeddings, train_pairs, train_labels):
    x_train = build_pair_feature_matrix(embeddings, train_pairs)
    mean, std = fit_standard_scaler(x_train)
    x_train = transform_standard_scaler(x_train, mean, std)
    y_train = np.asarray(train_labels, dtype=int)

    assert len(x_train) == len(y_train)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y_train))
    x_train = x_train[perm]
    y_train = y_train[perm]

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        learning_rate_init=1e-5,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=SEED
    )

    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    print("Train accuracy:", accuracy_score(y_train, train_pred))

    return model, mean, std


def test_model(model, mean, std, embeddings, test_pairs, test_labels):
    x_test = build_pair_feature_matrix(embeddings, test_pairs)
    x_test = transform_standard_scaler(x_test, mean, std)
    y_test = np.array(test_labels, dtype=int)

    assert len(x_test) == len(y_test)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    return accuracy, report, matrix
