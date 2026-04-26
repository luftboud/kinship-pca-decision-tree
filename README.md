# A Project of Kinship Detection with Dimensionality Reduction and Tree-Based Models
This project explores kinship verification from facial images using linear algebra and machine learning methods. 
We use PCA for dimensionality reduction and feature extraction, construct similarity-based feature vectors for pairs of faces, and classify them with a Decision Tree model. 
The project is based on the Families in the Wild (FIW) dataset.

## Project Goal

The goal of this project is to develop an efficient and easy-to-use system 
for kinship verification, capable of determining whether two individuals 
are kins based on facial image analysis.

## Pipeline

1. Load facial images from the dataset.
2. Detect, crop, resize, and normalize faces.
3. Flatten images into vectors.
4. Apply mean-centering.
5. Compute PCA embeddings.
6. Build face-pair feature vectors using similarity measures.
7. Train a Decision Tree classifier.
8. Evaluate the model on the test set.

## Methods Used

- **PCA (Principal Component Analysis)** for dimensionality reduction
- **Feature engineering** based on similarity between two embeddings
- **Decision Tree** for binary classification
- **FIW dataset** for kinship image pairs

## Dataset

This project uses the **Families in the Wild (FIW)** dataset, which contains facial images of family members along with kinship pair labels.
The dataset provides predefined relationships such as brother-brother, father-son, mother-daughter, and others.

## Repository Structure

```bash
.
├── data/                            # dataset files / metadata
├── src/                             # source code
│   ├── constants.py                 # project constants and configs
│   ├── preprocess.py                # image preprocessing (resize, normalize, flatten)
│   ├── pca.py                       # PCA computation and embeddings
│   ├── similarity.py                # similarity feature computation
│   ├── train_preparation.py         # train data preparation
│   ├── test_preparation.py          # test data preparation
│   ├── classifier.py  # model training and evaluation
│   └── main.py                      # pipeline entry point
├── .gitignore
├── requirements.txt                 # dependencies
├── report.pdf
└── README.md
```
## Installation

Clone the repository:

```bash
git clone https://github.com/luftboud/kinship-pca-decision-tree.git
cd kinship-pca-decision-tree
```
Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the main script:
```bash
python3 src/main.py
```
If needed, update dataset paths in the source code or configuration files before running.

## Evaluation

The model is evaluated as a binary classification system that predicts whether a pair of face images represents relatives (kin) or non-relatives.

The following evaluation metrics are used:

- **Accuracy** — overall proportion of correctly classified pairs
- **Precision** — how many predicted kin pairs are actually correct
- **Recall** — how many true kin pairs are correctly identified
- **F1-score** — harmonic mean of precision and recall
- **Confusion matrix** — detailed breakdown of correct and incorrect predictions for both classes

These metrics provide both a general performance estimate and a more detailed understanding of model behavior across classes.

## Authors
This project was developed as part of a university linear algebra course project.
Conducted by:
- Iia Maharyta
- Vladyslav Danylyshyn
- Arsen Botsko

## Project Presentation Videos

- [Project explanation by Iia](https://youtu.be/AIsf5mwg7ws?si=JDw6CCrSNRswZm14)
- [Project explanation by Vladyslav](https://youtu.be/Gq7dwD0i39g?si=cdAExuVHlWNkjk5a)
- [Project explanation by Arsen](https://youtu.be/uSCKeW5xQq4?si=gF-Cupe1jQIJxBB1)


## Report

The full project report is [available](report.pdf) in the repository.