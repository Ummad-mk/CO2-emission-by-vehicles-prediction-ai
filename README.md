# 🚗💨 CO2 Emission by Vehicles Prediction — AI System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)](https://xgboost.readthedocs.io)
[![Google Colab](https://img.shields.io/badge/Open%20in-Google%20Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **AIC354 – Machine Learning Fundamentals | Assignment 3**
> Applying, Evaluating, and Comparing Multiple ML Algorithms on a Large Real-World Dataset

---

## 📌 Problem Statement

Predict the **CO2 emissions (g/km)** of vehicles based on their specifications using **5 distinct machine learning algorithms** — evaluated and compared using **10-Fold Cross-Validation** on the full Canada CO2 Emissions dataset.

---

## 📂 Repository Structure

```
co2-emission-by-vehicles-prediction-ai/
├── dataset/
│   └── CO2 Emissions_Canada.csv        # Full dataset (7,385 vehicle records)
├── notebooks/
│   ├── notebook1_100_instances_3fold.ipynb   # 100 instances, 3-Fold CV (quick demo)
│   └── notebook2_full_dataset_10fold.ipynb   # Full dataset, 10-Fold CV (main submission)
├── models/
│   └── best_model.pkl                  # Saved best-performing trained model
├── webapp/
│   ├── app.py                          # Flask web application
│   ├── requirements.txt                # Python dependencies
│   ├── templates/
│   │   └── index.html                  # Frontend UI
│   └── static/
│       └── style.css                   # Custom styling
├── .gitignore
└── README.md
```

---

## 🤖 Machine Learning Algorithms Used

> **Note:** These algorithms are intentionally different from those in the Car Price Prediction System (Linear Regression, KNN, Decision Tree, Random Forest, Gradient Boosting).

| # | Algorithm | Key Configuration |
|---|-----------|-------------------|
| 1 | **Ridge Regression** | `alpha=1.0`, L2 regularization |
| 2 | **Lasso Regression** | `alpha=0.01`, L1 regularization |
| 3 | **Support Vector Regression (SVR)** | `kernel='rbf'`, `C=100`, `epsilon=0.1` |
| 4 | **Extra Trees Regressor** | `n_estimators=200`, `random_state=42` |
| 5 | **XGBoost Regressor** | `n_estimators=200`, `learning_rate=0.05`, `max_depth=6` |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | Government of Canada Open Data |
| **File** | `CO2 Emissions_Canada.csv` |
| **Records** | 7,385 vehicle entries |
| **Features** | 11 (Make, Model, Engine Size, Cylinders, Transmission, Fuel Type, etc.) |
| **Target** | `CO2 Emissions(g/km)` |

---

## 🚀 How to Run on Google Colab

### Notebook 2 (Full Dataset — Main Submission)
1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `notebooks/notebook2_full_dataset_10fold.ipynb`
3. Click **Runtime -> Run all**
4. Upload `dataset/CO2 Emissions_Canada.csv` when prompted
5. View results: RMSE comparison table + charts

### Notebook 1 (Quick Demo — 100 instances)
1. Upload `notebooks/notebook1_100_instances_3fold.ipynb`
2. Click **Runtime -> Run all**
3. Upload the dataset when prompted

---

## 🌐 Web App Deployment

The trained best model is deployed as a Flask web application.

**Live Demo:** `[Deployment link — add here after deploying]`

### Run Locally
```bash
cd webapp
pip install -r requirements.txt
python app.py
# Visit http://localhost:5000
```

---

## 📈 Results Summary

All models are evaluated using **RMSE** (Root Mean Square Error) — lower is better.

> Full results with charts are available inside `notebook2_full_dataset_10fold.ipynb`

---

## 👤 Author

**Student** | AIC354 – Machine Learning Fundamentals
Submitted for Assignment 3

---

## 📜 License

This project is for educational purposes. Dataset from Government of Canada Open Data Portal.
