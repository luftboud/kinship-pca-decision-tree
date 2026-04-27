from classifier import train_model, test_model
from constants import *

import train_preparation
import test_preparation


def main():
    pairs, labels, train_embeddings, family_labels, wk, centering = train_preparation.get_prepared_train_data(TRAIN_RELATIONSHIPS, TRAIN_FACES_ROOT)
    model, mean, std = train_model(train_embeddings, pairs, labels, family_labels)

    test_pairs, test_labels, test_embeddings = test_preparation.get_prepared_test_data(wk, centering)
    accuracy, report, matrix = test_model(model, mean, std, test_embeddings, test_pairs, test_labels)

    print(f"{model_label} accuracy: {accuracy:.4f}")
    print(f"{model_label} report:\n{report}")
    print(f"{model_label} confusion matrix:\n{matrix}")


if __name__ == "__main__":
    main()
