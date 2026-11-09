"""ECG MI classification with patient-disjoint evaluation.
Run: python src/improved_modelling.py
Uses the original feature CSV archives (not the previously normalised dataset).
"""
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)
FEATURES = ['RR Interval (s)', 'HR (bpm)', 'QRS Duration (s)', 'QT Interval (s)']


def load_raw_feature_csvs():
    rows = []
    for archive, label, prefix in [('PTB_Data_MI_csv.zip', 1, 'MI'), ('PTB_H_control_csv.zip', 0, 'Healthy')]:
        with zipfile.ZipFile(DATA / 'raw' / archive) as z:
            names = sorted(n for n in z.namelist() if n.lower().endswith('.csv'))
            if len(names) != 50:
                raise ValueError(f'{archive}: expected 50 files, found {len(names)}')
            for name in names:
                frame = pd.read_csv(z.open(name))
                if not set(FEATURES).issubset(frame.columns):
                    raise ValueError(f'Unexpected columns: {name}: {frame.columns.tolist()}')
                frame = frame[FEATURES].copy()
                frame['Patient_ID'] = f'{prefix}_{Path(name).stem}'
                frame['Label'] = label
                rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def metrics(y_true, y_pred, probabilities):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, probabilities),
    }


def main():
    df = load_raw_feature_csvs()
    df.to_csv(DATA / 'combined_unscaled_features.csv', index=False)
    patients = df[['Patient_ID', 'Label']].drop_duplicates().sort_values('Patient_ID')
    train_ids, test_ids = train_test_split(
        patients['Patient_ID'], test_size=0.25, random_state=42,
        stratify=patients['Label']
    )
    assert set(train_ids).isdisjoint(set(test_ids))
    train = df[df.Patient_ID.isin(train_ids)].copy()
    test = df[df.Patient_ID.isin(test_ids)].copy()
    print(f'Patients: {len(train_ids)} train / {len(test_ids)} test; observations: {len(train)} train / {len(test)} test')
    print('Train patient classes:', train[['Patient_ID','Label']].drop_duplicates()['Label'].value_counts().to_dict())
    print('Test patient classes:', test[['Patient_ID','Label']].drop_duplicates()['Label'].value_counts().to_dict())

    models = {
        'Random Forest': make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(random_state=42, n_jobs=-1)),
        'Linear SVM': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LinearSVC(random_state=42, max_iter=5000)),
        'Logistic Regression': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(random_state=42, max_iter=1000)),
    }
    results = []
    for name, model in models.items():
        model.fit(train[FEATURES], train.Label)
        # Linear SVM provides a decision score rather than a calibrated probability.
        is_svm = name == 'Linear SVM'
        score = model.decision_function(test[FEATURES]) if is_svm else model.predict_proba(test[FEATURES])[:, 1]
        pred = model.predict(test[FEATURES])
        row = {'model': name, 'level': 'observation', **metrics(test.Label, pred, score)}
        results.append(row)
        # Average observation probabilities within each held-out patient.
        patient_results = test[['Patient_ID','Label']].copy()
        patient_results['score'] = score
        patient_results = patient_results.groupby('Patient_ID', as_index=False).agg(Label=('Label','first'), score=('score','mean'))
        patient_pred = (patient_results.score >= (0 if is_svm else 0.5)).astype(int)
        results.append({'model': name, 'level': 'patient', **metrics(patient_results.Label, patient_pred, patient_results.score)})
        print(f'{name}: observation accuracy={row["accuracy"]:.3f}; patient accuracy={results[-1]["accuracy"]:.3f}; patient AUC={results[-1]["roc_auc"]:.3f}')
        ConfusionMatrixDisplay.from_predictions(patient_results.Label, patient_pred, labels=[0,1], display_labels=['Healthy','MI'])
        plt.title(f'{name} — held-out patients')
        plt.tight_layout()
        plt.savefig(RESULTS / f'{name.lower().replace(" ", "_")}_patient_confusion.png', dpi=150)
        plt.close()
        RocCurveDisplay.from_predictions(patient_results.Label, patient_results.score)
        plt.title(f'{name} — held-out patient ROC')
        plt.tight_layout()
        plt.savefig(RESULTS / f'{name.lower().replace(" ", "_")}_patient_roc.png', dpi=150)
        plt.close()
    pd.DataFrame(results).to_csv(RESULTS / 'metrics.csv', index=False)
    print('\nSaved combined_unscaled_features.csv, metrics.csv and patient-level plots.')
    print('Caution: ECG features and their extraction accuracy have not been independently validated; this is an educational analysis, not a clinical diagnostic model.')


if __name__ == '__main__':
    main()
