from pathlib import Path

import pandas as pd
import streamlit as st

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TEST_CSV = PROJECT_ROOT / "test_data.csv"
TARGET_COLUMN = "target"
RANDOM_STATE = 42
TEST_SIZE = 0.20

st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")


def build_models():
    """Five models, each wrapped with StandardScaler (fit on train only)."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB()),
        ]),
        "Random Forest (Ensemble)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200,
                                           random_state=RANDOM_STATE)),
        ]),
    }


@st.cache_resource(show_spinner="Training models (one-time)...")
def train_models():
    """Train all models once per session on the training split."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name=TARGET_COLUMN)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    models = build_models()
    for model in models.values():
        model.fit(X_train, y_train)
    return models, list(X.columns)


@st.cache_data
def load_default_test_data():
    if DEFAULT_TEST_CSV.exists():
        return pd.read_csv(DEFAULT_TEST_CSV)
    return None


def positive_class_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def split_features_target(df):
    if TARGET_COLUMN in df.columns:
        return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]
    return df.iloc[:, :-1], df.iloc[:, -1]


models, feature_names = train_models()

st.title("Breast Cancer Wisconsin — Classifier")
st.caption(
    "Binary classification (malignant vs benign): Logistic Regression, "
    "Decision Tree, kNN, Naive Bayes, Random Forest."
)

# --------------------------------------------------------------------------
# (a) upload + (b) model dropdown  -- on the main page for visibility
# --------------------------------------------------------------------------
st.header("1. Upload test data & select a model")
col_upload, col_model = st.columns(2)

with col_upload:
    uploaded = st.file_uploader(
        "Upload test data (CSV)",
        type=["csv"],
        help="CSV with the 30 feature columns and a 'target' column "
             "(0 = malignant, 1 = benign).",
    )

with col_model:
    selected_model_name = st.selectbox("Select a model", list(models.keys()))

# --------------------------------------------------------------------------
# Load data (uploaded or bundled fallback)
# --------------------------------------------------------------------------
if uploaded is not None:
    df = pd.read_csv(uploaded)
    data_source = "uploaded file"
else:
    df = load_default_test_data()
    data_source = "bundled test_data.csv (no file uploaded)"

if df is None:
    st.warning("Upload a test CSV to begin.")
    st.stop()

X, y = split_features_target(df)

missing = [c for c in feature_names if c not in X.columns]
if missing:
    st.error(f"Uploaded CSV is missing feature columns (up to 5): {missing[:5]}")
    st.stop()
X = X[feature_names]

st.success(f"Loaded {len(df)} rows from {data_source}.")

# --------------------------------------------------------------------------
# Results for the selected model
# --------------------------------------------------------------------------
model = models[selected_model_name]
y_pred = model.predict(X)
y_score = positive_class_scores(model, X)

# (c) Evaluation metrics
st.header(f"2. Evaluation Metrics — {selected_model_name}")
metrics = {
    "Accuracy": accuracy_score(y, y_pred),
    "AUC": roc_auc_score(y, y_score),
    "Precision": precision_score(y, y_pred),
    "Recall": recall_score(y, y_pred),
    "F1": f1_score(y, y_pred),
    "MCC": matthews_corrcoef(y, y_pred),
}
cols = st.columns(6)
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, f"{value:.4f}")

# (d) Confusion matrix + classification report (plain tables)
st.header("3. Confusion Matrix & Classification Report")
left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual malignant (0)", "Actual benign (1)"],
        columns=["Predicted malignant (0)", "Predicted benign (1)"],
    )
    st.table(cm_df)

with right:
    st.subheader("Classification Report")
    report = classification_report(
        y, y_pred,
        target_names=["malignant (0)", "benign (1)"],
        output_dict=True,
    )
    st.dataframe(pd.DataFrame(report).T.round(4))
