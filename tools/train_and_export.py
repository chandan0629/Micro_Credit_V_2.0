"""
Train a small RandomForest on the provided credit_risk_data.csv and export a joblib model
to `api/rf_model.joblib` for use in the Vercel serverless function.

Usage:
  python tools/train_and_export.py --csv ../credit_risk_data.csv

This script keeps training lightweight and uses only scikit-learn so the exported model
can be deployed to Vercel serverless functions.
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
import os

FEATURES = ['total_transactions', 'avg_transaction_amount', 'payment_consistency_score',
            'business_age_months', 'digital_footprint_score']

def derive_labels(df):
    labels = []
    for _, row in df.iterrows():
        risk_score = 0
        if row['payment_consistency_score'] < 70:
            risk_score += 40
        elif row['payment_consistency_score'] < 85:
            risk_score += 20
        if row['business_age_months'] < 12:
            risk_score += 25
        elif row['business_age_months'] < 36:
            risk_score += 15
        if row['total_transactions'] < 5000:
            risk_score += 20
        elif row['total_transactions'] < 15000:
            risk_score += 10
        if row['digital_footprint_score'] < 60:
            risk_score += 15
        elif row['digital_footprint_score'] < 80:
            risk_score += 8
        labels.append(1 if risk_score > 50 else 0)
    return np.array(labels)

def main(csv_path):
    df = pd.read_csv(csv_path)
    X = df[FEATURES].values
    y = derive_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train_scaled, y_train)

    score = clf.score(X_test_scaled, y_test)
    print(f'Test accuracy: {score:.4f}')

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'api')
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'rf_model.joblib')
    dump(clf, model_path)
    print(f'Exported model to {model_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to credit_risk_data.csv')
    args = parser.parse_args()
    main(args.csv)
