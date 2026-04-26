from pathlib import Path

SEED = 42
FEATURES_AMOUNT = 64
IMG_SIZE = (64, 64)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_FACES_ROOT = PROJECT_ROOT / "data" / "test-public-faces"
TRAIN_RELATIONSHIPS = PROJECT_ROOT / "data" / "train_relationships.csv"

TEST_FACES_ROOT = PROJECT_ROOT / "data" / "test-private-faces"
SPLIT_TEST_RELATIONS = PROJECT_ROOT / "data" / "test-private-lists"
TEST_FACES_RELATIONSHIPS = PROJECT_ROOT / "data" / "test_relationships.csv"
TEST_FACES_LABELS = PROJECT_ROOT / "data" / "test-private-labels"
