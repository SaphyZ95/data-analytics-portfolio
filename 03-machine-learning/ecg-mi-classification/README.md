# ECG-Based Myocardial Infarction Classification Using Machine Learning

## Project Overview

This project investigates the use of machine learning to classify ECG observations associated with myocardial infarction (MI) and healthy control subjects.

Originally developed as part of my Master of Predictive Analytics at Curtin University in 2024, the project demonstrates ECG signal processing, feature extraction, data preprocessing, machine learning classification and model evaluation using Python.

## Dataset

The processed dataset contains:
- 100 patients: 50 MI and 50 healthy control subjects
- 17,734 ECG observations
- Four extracted ECG features
- Patient identifiers and binary classification labels

**Features:**
- RR Interval (seconds)
- Heart Rate (bpm)
- QRS Duration (seconds)
- QT Interval (seconds)

The dataset was generated from ECG recordings using signal processing, feature extraction, outlier removal and Min-Max normalisation.

## Tools and Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- SciPy and PyWavelets
- Matplotlib and Seaborn

## Methodology

1. Extracted Lead II ECG signals from the original recordings.
2. Applied wavelet denoising and Butterworth filtering.
3. Extracted four ECG parameters.
4. Removed outliers and normalised the extracted features.
5. Trained Random Forest, Support Vector Machine and Logistic Regression classifiers.
6. Evaluated performance using accuracy, precision, recall, F1 score and ROC-AUC.

## Original University Results

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Random Forest | 86.1% | 87.6% | 0.931 |
| Support Vector Machine | 75.1% | 77.4% | 0.816 |
| Logistic Regression | 66.9% | 69.9% | 0.727 |

Random Forest achieved the highest performance in the original experiment. QT interval was identified as the most influential feature in that model.

## Evaluation Limitations

The original evaluation randomly split ECG observations rather than separating patients. Consequently, observations from the same patient could appear in both training and testing sets, potentially inflating performance estimates.

The dataset was also normalised separately for each patient before model training.

An improved evaluation using patient-separated training and testing is being developed to address the patient-overlap limitation and provide a more reliable assessment of generalisation.

## Project Structure

- `data/` — Processed ECG feature dataset
- `src/Modelling.py` — Original machine learning implementation
- `src/improved_modelling.py` — Patient-separated evaluation (planned)
- `results/` — Model evaluation outputs and visualisations (planned)

## Project Status

**In progress:** Updating the evaluation methodology, improving reproducibility and preparing visualisations.

## Disclaimer

This is an academic machine-learning project for educational and research purposes. The models are not clinically validated and must not be used for medical diagnosis.
