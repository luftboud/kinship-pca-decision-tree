from classifier import train_model, test_model
from constants import *

import train_preparation
import test_preparation


def main():
    pairs, labels, train_embeddings, wk, centering = train_preparation.get_prepared_train_data(TRAIN_RELATIONSHIPS, TRAIN_FACES_ROOT)
    model, mean, std = train_model(train_embeddings, pairs, labels)

    test_pairs, test_labels, test_embeddings = test_preparation.get_prepared_test_data(wk, centering)
    accuracy, report, matrix = test_model(model, mean, std, test_embeddings, test_pairs, test_labels)

    print(f"Decision Tree accuracy: {accuracy:.4f}")
    print(f"Decision Tree report:\n{report}")
    print(f"Decision Tree confusion matrix:\n{matrix}")


if __name__ == "__main__":
    main()
