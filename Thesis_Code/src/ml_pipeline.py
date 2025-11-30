# src/ml_pipeline.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report

def train_model(df):
    print("\n--- Training Random Forest ---")
    df['LogLambda'] = np.log10(df['Lambda'])
    df['Compactness'] = df['Mass'] / df['Radius']
    
    X = df[['Mass', 'Radius', 'LogLambda', 'Compactness']]
    y = df['Label']
    groups = df['Curve_ID']

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Hadronic', 'Quark']))
    
    return rf, X_test, y_test

def analyze_candidates(model):
    print("\n" + "="*80 + "\n" + f"{'INFERENCE ON REAL ASTROPHYSICAL DATA':^80}" + "\n" + "="*80)
    candidates = [
        {"Name": "GW170817",     "M": 1.40, "sM": 0.10, "R": 11.90, "sR": 1.40, "L": 190, "sL": 120},
        {"Name": "PSR J0740+66", "M": 2.08, "sM": 0.07, "R": 12.35, "sR": 0.75, "L": 300, "sL": 300},
        {"Name": "PSR J0030+04", "M": 1.44, "sM": 0.15, "R": 13.02, "sR": 1.06, "L": 300, "sL": 300},
        {"Name": "HESS J1731",   "M": 0.77, "sM": 0.17, "R": 10.40, "sR": 0.78, "L": 800, "sL": 400},
        {"Name": "GW190814(sec)","M": 2.59, "sM": 0.09, "R": 12.00, "sR": 3.00, "L": 10,  "sL": 10}
    ]
    print(f"{'Candidate':<20} | {'Mass (Msun)':<12} | {'Radius (km)':<12} | {'P(Quark)':<10} | {'Verdict':<10}")
    print("-" * 80)

    for star in candidates:
        n_mc = 5000
        m_s = np.random.normal(star['M'], star['sM'], n_mc)
        r_s = np.random.normal(star['R'], star['sR'], n_mc)
        l_s = np.abs(np.random.normal(star['L'], star['sL'], n_mc))
        
        valid = (r_s > 8.0) & (l_s >= 1.0)
        X_mc = pd.DataFrame({
            'Mass': m_s[valid], 'Radius': r_s[valid], 
            'LogLambda': np.log10(l_s[valid]), 'Compactness': m_s[valid]/r_s[valid]
        })
        probs = model.predict_proba(X_mc)[:, 1]
        mean_p = np.mean(probs)
        verdict = "QUARK" if mean_p > 0.5 else "HADRONIC"
        print(f"{star['Name']:<20} | {star['M']:<12.2f} | {star['R']:<12.2f} | {mean_p*100:5.1f}%     | {verdict:<10}")
    print("="*80 + "\n")