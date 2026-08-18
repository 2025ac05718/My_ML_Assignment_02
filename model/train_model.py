"""
Machine Learning - Assignment 2
Implementation of Classification Models on Heart Disease Dataset
Student ID: 2025AC05718
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    matthews_corrcoef
)


def load_and_preprocess_data(dataset_path):
    # Load dataset
    data = pd.read_csv(dataset_path)
    
    # Separate features and target
    X = data.drop(columns=['target'])
    y = data['target']
    
    # 80-20 train-test split with fixed seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Scale continuous/numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def train_and_evaluate_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the 5 classifiers required by the assignment
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'kNN': KNeighborsClassifier(n_neighbors=7),
        'Naive Bayes': GaussianNB(),
        'Random Forest (Ensemble)': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    }
    
    file_mapping = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'kNN': 'knn.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest (Ensemble)': 'random_forest.pkl'
    }
    
    # Save standard scaler
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    metrics_summary = []
    
    for name, clf in classifiers.items():
        # Models sensitive to scale are trained on scaled features
        if name in ['Logistic Regression', 'kNN', 'Naive Bayes']:
            clf.fit(X_train_scaled, y_train)
            preds = clf.predict(X_test_scaled)
            probs = clf.predict_proba(X_test_scaled)[:, 1]
        else:
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            probs = clf.predict_proba(X_test)[:, 1]
            
        # Compute evaluation metrics
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        mcc = matthews_corrcoef(y_test, preds)
        
        metrics_summary.append({
            'ML Model Name': name,
            'Accuracy': round(acc, 4),
            'AUC': round(auc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'MCC': round(mcc, 4)
        })
        
        # Save model object
        with open(os.path.join(output_dir, file_mapping[name]), 'wb') as f:
            pickle.dump(clf, f)
            
    summary_df = pd.DataFrame(metrics_summary)
    
    # Save metrics JSON for fast loading in the app
    summary_df.to_json(os.path.join(output_dir, 'metrics_summary.json'), orient='records')
    
    return summary_df


if __name__ == '__main__':
    # Determine base and model directories whether run from root or model/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(current_dir) == 'model':
        model_save_dir = current_dir
        root_dir = os.path.dirname(current_dir)
    else:
        model_save_dir = os.path.join(current_dir, 'model')
        root_dir = current_dir
        
    data_path = os.path.join(root_dir, 'heart_disease_dataset.csv')
    
    print("Loading data and training models...")
    X_tr, X_te, y_tr, y_te, X_tr_sc, X_te_sc, scl = load_and_preprocess_data(data_path)
    
    # Export test split for evaluation and streamlit upload demo
    test_export = X_te.copy()
    test_export['target'] = y_te
    test_export.to_csv(os.path.join(root_dir, 'test_data.csv'), index=False)
    print("Saved test_data.csv ({} instances)".format(len(test_export)))
    
    results = train_and_evaluate_models(
        X_tr, X_te, y_tr, y_te, X_tr_sc, X_te_sc, scl, model_save_dir
    )
    
    print("\n" + "="*50)
    print("MODEL EVALUATION SUMMARY")
    print("="*50)
    print(results.to_string(index=False))
    print("="*50)
    print("All models successfully trained and serialized to {}/".format(model_save_dir))
