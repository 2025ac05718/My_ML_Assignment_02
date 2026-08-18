import os
import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Set page configuration
st.set_page_config(
    page_title="Machine Learning Assignment 2 - Model Deployment",
    layout="wide"
)

# Custom minimal styling for metric cards
st.markdown("""
    <style>
    .metric-box {
        background-color: #1e2530;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border: 1px solid #2e3846;
    }
    .metric-label {
        color: #9ba3af;
        font-size: 13px;
        font-weight: 500;
        text-transform: uppercase;
    }
    .metric-val {
        color: #38bdf8;
        font-size: 22px;
        font-weight: 700;
        margin-top: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Define directories and model mappings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

MODEL_MAP = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'kNN': 'knn.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
    'Random Forest (Ensemble)': 'random_forest.pkl'
}


@st.cache_resource
def load_saved_model(model_filename):
    path = os.path.join(MODEL_DIR, model_filename)
    with open(path, 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler():
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.title("ML Assignment 2")
st.sidebar.subheader("Model & Data Controls")

selected_model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(MODEL_MAP.keys()),
    index=0
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV (Optional)",
    type=['csv'],
    help="Upload test data with feature columns and target."
)

default_test_csv = os.path.join(BASE_DIR, 'test_data.csv')

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.sidebar.success("Loaded uploaded dataset successfully.")
elif os.path.exists(default_test_csv):
    test_df = pd.read_csv(default_test_csv)
    st.sidebar.info("Using default hold-out test dataset (205 rows).")
else:
    st.error("No test data found. Please place test_data.csv in the directory or upload a CSV file.")
    st.stop()


# ---------------------------------------------------------
# Main Application Content
# ---------------------------------------------------------
st.title("Heart Disease Classification - Evaluation Dashboard")
st.write(
    "This application demonstrates the performance of multiple classification models trained on the Heart Disease dataset. "
    "Use the sidebar to choose a model or upload test data to view the evaluation metrics."
)
st.divider()

# Load model and scaler
model = load_saved_model(MODEL_MAP[selected_model_name])
scaler = load_scaler()

# Prepare features and target
if 'target' in test_df.columns:
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']
else:
    X_test = test_df
    y_test = None

# Apply scaling for scale-dependent models
if selected_model_name in ['Logistic Regression', 'kNN', 'Naive Bayes']:
    X_test_eval = scaler.transform(X_test)
else:
    X_test_eval = X_test

# Generate predictions
y_pred = model.predict(X_test_eval)

if hasattr(model, 'predict_proba'):
    y_prob = model.predict_proba(X_test_eval)[:, 1]
else:
    y_prob = y_pred

# Display metrics
if y_test is not None:
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    st.subheader(f"Performance Metrics: {selected_model_name}")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Accuracy</div><div class="metric-val">{acc:.4f}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box"><div class="metric-label">AUC Score</div><div class="metric-val">{auc:.4f}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Precision</div><div class="metric-val">{prec:.4f}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Recall</div><div class="metric-val">{rec:.4f}</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-box"><div class="metric-label">F1 Score</div><div class="metric-val">{f1:.4f}</div></div>', unsafe_allow_html=True)
    with col6:
        st.markdown(f'<div class="metric-box"><div class="metric-label">MCC Score</div><div class="metric-val">{mcc:.4f}</div></div>', unsafe_allow_html=True)

    st.divider()

    # Visualizations
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['No Disease (0)', 'Disease (1)'],
                    yticklabels=['No Disease (0)', 'Disease (1)'])
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'{selected_model_name} Confusion Matrix')
        st.pyplot(fig)

    with col_right:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

# ---------------------------------------------------------
# Comparison Table Across All 5 Models
# ---------------------------------------------------------
st.divider()
st.subheader("Model Comparison Table")

if y_test is not None:
    comparison_data = []
    for m_name, m_file in MODEL_MAP.items():
        loaded_m = load_saved_model(m_file)
        X_m = scaler.transform(X_test) if m_name in ['Logistic Regression', 'kNN', 'Naive Bayes'] else X_test
        p = loaded_m.predict(X_m)
        pb = loaded_m.predict_proba(X_m)[:, 1] if hasattr(loaded_m, 'predict_proba') else p

        comparison_data.append({
            'ML Model Name': m_name,
            'Accuracy': round(accuracy_score(y_test, p), 4),
            'AUC': round(roc_auc_score(y_test, pb), 4),
            'Precision': round(precision_score(y_test, p, zero_division=0), 4),
            'Recall': round(recall_score(y_test, p, zero_division=0), 4),
            'F1': round(f1_score(y_test, p, zero_division=0), 4),
            'MCC': round(matthews_corrcoef(y_test, p), 4)
        })

    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True)

# ---------------------------------------------------------
# Observations Section
# ---------------------------------------------------------
st.divider()
st.subheader("Observations on Model Performance")

observations = [
    {
        "ML Model Name": "Logistic Regression",
        "Observation about model performance": "Delivers the highest overall accuracy (99.51%) and AUC (1.0000), effectively capturing linear relations between clinical risk factors."
    },
    {
        "ML Model Name": "Decision Tree",
        "Observation about model performance": "Provides high interpretability with 90.73% accuracy, but suffers from lower recall (53.57%) due to greedy axis-aligned partitioning."
    },
    {
        "ML Model Name": "kNN",
        "Observation about model performance": "Achieves 92.68% accuracy after standard scaling; performance is stable across local neighborhood clusters."
    },
    {
        "ML Model Name": "Naive Bayes",
        "Observation about model performance": "Performs robustly (92.68% accuracy, 0.9514 AUC) despite the conditional independence assumption among attributes."
    },
    {
        "ML Model Name": "Random Forest (Ensemble)",
        "Observation about model performance": "Achieves perfect precision (1.0000) with zero false positive predictions, though recall is comparatively lower on this test split."
    },
    {
        "ML Model Name": "Overall Winner for your dataset?",
        "Observation about model performance": "Logistic Regression is the overall winner based on balanced performance across Accuracy, AUC, F1 Score, and MCC."
    }
]

st.table(pd.DataFrame(observations))
