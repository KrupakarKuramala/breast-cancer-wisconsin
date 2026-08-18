# Breast Cancer Wisconsin — Multi-Model Classification & Streamlit App

An end-to-end machine learning project that trains five classification models on the **Breast Cancer Wisconsin (Diagnostic)** dataset, evaluates them on six metrics, and serves an interactive **Streamlit** web app for exploring the results.

---

## a. Problem Statement

Predict whether a tumor is **malignant (0)** or **benign (1)**. This is a **binary classification** problem. The goal is to train several standard classifiers, compare them on a common held-out test set, and expose the evaluation through a deployed web application.

## b. Dataset Description

| Property | Value |
|----------|-------|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI ML Repository / `sklearn.datasets.load_breast_cancer` |
| Instances | 569 (≥ 500 required) |
| Features | 30 numeric (≥ 12 required) |
| Target | Binary — malignant (0) = 212, benign (1) = 357 |
| Missing values | None |


A stratified 80/20 train/test split (seed = 42) is used. The 20% held-out set (114 rows) is exported as `test_data.csv` and is what the Streamlit app evaluates (the assignment asks to upload **only test data** because the free tier has limited capacity).

## c. GitHub Repository Link

https://github.com/KrupakarKuramala/breast-cancer-wisconsin
https://github.com/KrupakarKuramala/breast-cancer-wisconsin.git

## d. Models Used

All five models are trained on the **same dataset** and the **same split**. Each model is a scikit-learn `Pipeline` with `StandardScaler` + classifier, so the raw uploaded CSV is scaled internally (scaling is fit on train data only to avoid leakage). No serialized model files are used — the app **trains the models in-memory at startup** (cached) using the same stratified split (seed = 42), then evaluates on the uploaded test CSV.

### Comparison Table (evaluated on the held-out test set, 114 rows)

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression    | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree          | 0.9123   | 0.9157 | 0.9559    | 0.9028 | 0.9286 | 0.8174 |
| kNN                    | 0.9561   | 0.9788 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes            | 0.9298   | 0.9868 | 0.9444    | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble)| 0.9561  | 0.9932 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Best overall. The classes are close to linearly separable after standardization, so a linear decision boundary fits extremely well — top scores on every metric (Accuracy 0.9825, AUC 0.9954, MCC 0.9623). |
| Decision Tree | Weakest model here. A single unpruned tree overfits the training data and generalizes worst (Accuracy 0.9123, AUC 0.9157, MCC 0.8174). Its low AUC reflects hard, high-variance splits rather than calibrated probabilities. |
| kNN | Strong and balanced (Accuracy 0.9561, MCC 0.9054). Benefits heavily from feature scaling; performs well because similar tumors cluster in feature space. |
| Naive Bayes | Good ranking ability (AUC 0.9868) but slightly lower accuracy (0.9298), since its feature-independence assumption is violated (many features are correlated, e.g. radius/perimeter/area). |
| Random Forest (Ensemble) | Very strong and robust (Accuracy 0.9561, AUC 0.9932, MCC 0.9054). Ensembling reduces the single tree's variance dramatically — same accuracy as kNN but with much better AUC. |
| **Overall Winner** | **Logistic Regression** — highest on all six metrics for this dataset, with Random Forest a close, more robust runner-up. |

---

## Project Structure

```
breast-cancer-ml-app/
├── app.py                     
├── requirements.txt            # Dependencies (pinned)
├── README.md                   # This file
├── test_data.csv               # Held-out test split (features + target)
└── model/
    ├── train_models.ipynb      # Notebook version of the same
```

https://breast-cancer-wisconsin-2025ac05993.streamlit.app/

