# breast_cancer_training_advanced.py

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")

# ✅ Config
DATA_PATH = r'D:\projects\breast cnacer project\data.csv'
MODEL_PATH = r'D:\projects\breast cnacer project\cancer_model.pkl'
SCALER_PATH = r'D:\projects\breast cnacer project\scaler.pkl'

# ✅ Features to drop
DROP_COLS = ['id', 'Unnamed: 32']

# ✅ 1. Load and clean data
def load_data(path):
    df = pd.read_csv(path)
    df.drop(columns=[col for col in DROP_COLS if col in df.columns], inplace=True)
    df.dropna(inplace=True)
    df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})
    print(f"✅ Data loaded and cleaned. Shape: {df.shape}")
    return df

# ✅ 2. Split features and target
def split_features_target(df):
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    return X, y

# ✅ 3. Train and evaluate model
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=10000, random_state=42))
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    print("\n✅ Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    print(f"✅ 5-fold Cross-Validation Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    return pipeline

# ✅ 4. Optional: Feature importance for tree-based models
def train_random_forest(X, y):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n⭐ Random Forest Feature Importances:")
    print(importances)
    return rf

# ✅ 5. Save model and scaler separately
def save_model_and_scaler(pipeline):
    # Extract scaler and model from pipeline
    scaler = pipeline.named_steps['scaler']
    model = pipeline.named_steps['classifier']

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\n✅ Model saved to: {MODEL_PATH}")
    print(f"✅ Scaler saved to: {SCALER_PATH}")

# ✅ 6. Main
def main():
    print("🚀 Starting Breast Cancer Model Training...")

    df = load_data(DATA_PATH)
    X, y = split_features_target(df)

    # Train logistic regression pipeline
    pipeline = train_and_evaluate(X, y)
    save_model_and_scaler(pipeline)

    # Optional Random Forest feature importance
    _ = train_random_forest(X, y)

    print("\n🎯 Training Complete.")

if __name__ == '__main__':
    main()
