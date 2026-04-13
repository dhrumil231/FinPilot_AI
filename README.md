# FinPilot_AI
<div align="center">

# 🧠 FinPilot AI
### Smart Budget & Financial Advisor

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557C?style=for-the-badge)](https://matplotlib.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-27ae60?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-2ecc71?style=for-the-badge)]()
[![Hackathon](https://img.shields.io/badge/Global%20Fusion%20Hackathon-FinTech%20Track%202026-e74c3c?style=for-the-badge)]()

<br/>

> **Financial advice has always been a luxury product.**
> **FinPilot AI gives the other 3.5 billion people something better.**

<br/>

*An end-to-end machine learning pipeline that analyzes personal financial data across 32,424 real user profiles — predicting spending patterns, classifying loan risk, tiering savings readiness, and delivering a personalised 0–100 Financial Health Score with actionable budget recommendations in plain English.*

<br/>

</div>

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [System Architecture](#-system-architecture)
- [Dataset](#-dataset)
- [Feature Engineering](#-feature-engineering)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Model Results](#-model-results)
- [Visualizations](#-visualizations--26-figures)
- [FinPilot Budget Advisor Engine](#-finpilot-budget-advisor-engine)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Sample Output](#-sample-output)
- [Project Impact](#-project-impact)
- [Future Roadmap](#-future-roadmap)
- [Author](#-author)
- [License](#-license)

---

## 🎯 Problem Statement

Over **3.5 billion people** globally have no access to a financial advisor.

The tools that exist — bank apps, spending dashboards, budgeting widgets — show you what already happened. They produce a report of last month and call it guidance. No one tells you what to do **next**.

The consequences are real and measurable:

- The average person saves less than 5% of their monthly income
- Half the population carries debt with a DTI ratio above 0.43 — the risk threshold
- Most people could not survive 3 months on their savings if their income stopped tomorrow
- The majority retire underprepared — not from carelessness, but from the absence of a plan

This is not a financial literacy problem. It is an **access problem**. A real financial advisor costs hundreds of dollars an hour. FinPilot AI is built to close that gap.

---

## 💡 Solution Overview

FinPilot AI is a complete data science and machine learning pipeline that transforms raw personal finance data into four concrete outputs:

| Output | Description |
|---|---|
| **Predictions** | What will a user spend next month? What is their loan risk profile? |
| **Classifications** | Which savings tier do they fall into across a 32,000-user benchmark? |
| **Recommendations** | A personalised, rule-based budget plan across 9 financial health dimensions |
| **A Score** | A 0–100 Financial Health Score with a letter grade (A / B / C / D) |

The entire system runs on a single dataset, one notebook, and zero external APIs. It is built to be extended, deployed, and scaled.

---

## ✨ Key Features

- 🔬 **3 independent ML prediction tasks** — regression, binary classification, and 5-class multi-class classification, each with its own leakage-free feature set and evaluation framework
- 🤖 **11 ML algorithms benchmarked** — Ridge, Lasso, Random Forest, Extra Trees, Gradient Boosting, Hist Gradient Boosting, SVR (RBF), MLP Neural Network, KNN, AdaBoost, Decision Tree
- 📊 **26 production-quality visualizations** — spanning EDA, model evaluation, calibration curves, permutation importance, learning curves, income band analysis, time trends, and residual diagnostics
- 🧮 **13 engineered financial features** — derived ratios including expense ratio, EMI-to-income, savings runway in months, loan burden, net monthly cashflow, and interest cost estimate
- 💬 **Rule-based Budget Advisor Engine** — 9-category personalised recommendations with a composite 0–100 health score and A/B/C/D letter grade
- 🌍 **Global benchmark dataset** — 32,424 users across 5 regions: Africa, Asia, Europe, North America, and Other
- 🔁 **3-fold stratified cross-validation** — on all model families with boxplot stability analysis
- 📈 **Learning curve diagnostics** — bias-variance tradeoff analysis for all new model families
- 🎯 **Calibration analysis** — assessing whether predicted probabilities are reliable, not just accurate
- 📉 **Permutation feature importance** — model-agnostic importance computed across all 3 prediction tasks simultaneously

---

## 🛠 Tech Stack

| Category | Tools & Libraries |
|---|---|
| **Language** | Python 3.9+ |
| **Data Manipulation** | Pandas 2.0+, NumPy 1.24+ |
| **Machine Learning** | Scikit-Learn 1.3+ |
| **Statistical Analysis** | SciPy 1.11+ |
| **Visualization** | Matplotlib 3.7+, Seaborn 0.12+ |
| **Notebook Environment** | Jupyter Notebook / Google Colab |
| **Preprocessing** | StandardScaler, LabelEncoder, MinMaxScaler |
| **Model Evaluation** | StratifiedKFold, cross_val_score, permutation_importance, calibration_curve, learning_curve |
| **ML Models** | Ridge, Lasso, RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, SVR, MLPClassifier, KNeighborsClassifier, AdaBoost, DecisionTree |

---

## 🏗 System Architecture
