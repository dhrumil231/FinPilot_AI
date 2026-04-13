# FinPilot_AI
# 💰 FinPilot AI — Personal Finance Intelligence & Budget Advisory System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blueviolet?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Project Overview

**FinPilot AI** is an end-to-end machine learning pipeline built on a real-world personal finance dataset. The project combines supervised learning (regression + multi-class classification), rich exploratory data analysis, and a rule-based budget advisory engine — all packaged in a single, reproducible notebook.

The core question driving this project: *Can we predict how individuals manage their money, and can a data-driven advisor provide actionable, personalized financial guidance?*

### Key Capabilities
- **Predict monthly expenses** from income, credit, demographics, and loan features
- **Classify loan risk** (binary: has loan / no loan) using multiple classifiers
- **Tier savings behavior** into 5 categories (Very Low → Very High) via multi-class classification
- **Advise users** with a rule-based budget advisor that evaluates emergency funds, expense ratios, credit health, and EMI burden
- **Visualize everything** — 26 publication-quality charts covering distributions, feature importance, ROC curves, calibration, learning curves, and more

---

## 📂 Repository Structure

```
FinPilotAI/
│
├── FinPilotAI_2_.ipynb            # Main notebook — full ML pipeline
├── Personal_Finance_Dataset_1.csv # Input dataset (required to run)
├── finpilot_plots/                # Auto-generated folder with all 26 plot PNGs
│   ├── fig01_distributions.png
│   ├── fig02_...
│   └── ...
├── finpilot_combined_board.png    # High-res merged visualization board
└── README.md
```

---

## 🗃️ Dataset

**File:** `Personal_Finance_Dataset_1.csv`

| Feature | Description |
|---|---|
| `monthly_income_usd` | Individual's gross monthly income |
| `monthly_expenses_usd` | Total monthly spending (target: regression) |
| `savings_usd` | Current savings balance |
| `credit_score` | Numeric credit score |
| `debt_to_income_ratio` | Debt as a proportion of income |
| `savings_to_income_ratio` | Savings relative to income |
| `loan_amount_usd` | Outstanding loan principal |
| `loan_interest_rate_pct` | Loan interest rate |
| `has_loan_binary` | Binary label — 1 if the individual holds a loan (target: binary classification) |
| `savings_tier` | Ordinal savings category: Very Low / Low / Medium / High / Very High (target: multi-class) |
| `gender`, `education_level`, `employment_status`, `region`, `loan_type`, `age_group` | Demographic & categorical features |
| `record_date` | Date of the financial record |

---

## ⚙️ Pipeline Architecture

```
Raw CSV
  │
  ▼
[1] Data Loading & Overview        → shape, dtypes, missing values, summary stats
  │
  ▼
[2] Exploratory Data Analysis      → 8 numeric distributions, correlation heatmap,
  │                                   category breakdowns, scatter plots (fig01–fig12)
  ▼
[3] Feature Engineering            → date parsing → year/month extraction
  │                                   derived features: net_monthly_cashflow,
  │                                   expense_ratio, emi_burden_ratio
  ▼
[4] Preprocessing                  → LabelEncoder (6 categorical cols)
  │                                   StandardScaler / MinMaxScaler per model
  ▼
[5] Model 1 — Regression           → predict monthly_expenses_usd
  │
[6] Model 2 — Binary Classification → predict has_loan_binary
  │
[7] Model 3 — Multi-Class          → predict savings_tier (5 classes)
  │
  ▼
[8] Post-Training Analysis         → cross-validation comparison, confusion matrices,
  │                                   ROC curves, feature importance (fig13–fig16)
  ▼
[9] FinPilot Budget Advisor        → rule-based personalized recommendation engine
  │
  ▼
[10] Enhancement Block             → 3 additional model families + 10 new charts
                                      (fig17–fig26)
```

---

## 🤖 Models Trained

### Model 1 — Monthly Expenses Prediction (Regression)

| Model | Metric |
|---|---|
| Ridge Regression | MAE, RMSE, R² |
| Lasso Regression | MAE, RMSE, R² |
| Random Forest Regressor | MAE, RMSE, R² |
| Extra Trees Regressor | MAE, RMSE, R² |
| **Gradient Boosting Regressor** *(enhanced)* | MAE, RMSE, R² |
| **HistGradient Boosting Regressor** *(enhanced)* | MAE, RMSE, R² |
| **SVR (RBF kernel)** *(enhanced)* | MAE, RMSE, R² |

### Model 2 — Loan Risk Classification (Binary)

| Model | Metric |
|---|---|
| Logistic Regression | Accuracy, ROC-AUC |
| Random Forest Classifier | Accuracy, ROC-AUC |
| Extra Trees Classifier | Accuracy, ROC-AUC |
| Decision Tree Classifier | Accuracy, ROC-AUC |
| **Gradient Boosting Classifier** *(enhanced)* | Accuracy, ROC-AUC |
| **MLP Neural Network** *(enhanced)* | Accuracy, ROC-AUC |
| **K-Nearest Neighbors** *(enhanced)* | Accuracy, ROC-AUC |

### Model 3 — Savings Tier Classification (Multi-Class, 5 Labels)

| Model | Metric |
|---|---|
| Logistic Regression | Accuracy |
| Random Forest Classifier | Accuracy |
| Extra Trees Classifier | Accuracy |
| Decision Tree Classifier | Accuracy |
| **AdaBoost** *(enhanced)* | Accuracy |
| **HistGradient Boosting Classifier** *(enhanced)* | Accuracy |

---

## 💡 FinPilot Budget Advisor

The rule-based advisor takes a user's financial profile as input and produces a personalized report with actionable recommendations. It evaluates:

| Signal | Threshold / Logic |
|---|---|
| Emergency Fund | < 3 months of expenses → **Critical Alert** |
| Expense Ratio | > 80% of income → **Overspending Warning** |
| Savings Opportunity | 20–50% income → **Savings Target** |
| Credit Score | Poor / Fair / Good / Very Good / Excellent bands |
| Debt-to-Income | > 0.4 → **High Debt Risk** |
| EMI Burden | EMI / Income > 30% → **EMI Overload** |
| Retirement Readiness | Age > 50 with low savings → **Retirement Alert** |

**Example usage:**

```python
finpilot_advisor(
    monthly_income      = 5000,
    monthly_expenses    = 3800,
    savings             = 4000,
    credit_score        = 620,
    debt_to_income_ratio = 0.45,
    has_loan            = True,
    loan_amount         = 15000,
    monthly_emi         = 450,
    age                 = 34
)
```

**Sample output:**
```
  ┌─────────────────────────────────────────────────────────┐
  │         FinPilot AI — Personalised Financial Report      │
  └─────────────────────────────────────────────────────────┘

  📊 Financial Snapshot
     Monthly Income    : $5,000.00
     Monthly Expenses  : $3,800.00
     Monthly EMI       : $450.00
     Net Cashflow      : $750.00
     Expense Ratio     : 76.0%
     Savings           : $4,000.00  (1.1 months of expenses)
     Credit Score      : 620  (Fair)
     Debt-to-Income    : 0.45

  💡 FinPilot Recommendations
  -------------------------------------------------------
  🚨 Emergency Fund   Your savings cover only 1.1 months ...
  ⚠️  High Debt-to-Income  Your DTI of 0.45 exceeds ...
  📈 Credit Improvement   A score of 620 limits your ...
```

---

## 📊 Visualizations (26 Charts)

### Original EDA & Model Plots (fig01 – fig16)
| Figure | Description |
|---|---|
| fig01 | Distributions of 8 key numeric features |
| fig02 | Correlation heatmap |
| fig03 | Income vs. Expenses scatter by employment status |
| fig04 | Savings distribution by age group |
| fig05–fig08 | Category breakdowns (gender, education, region, loan type) |
| fig09–fig12 | Feature pair plots and outlier analysis |
| fig13 | Cross-validation comparison — all models |
| fig14 | Confusion matrices (binary + multi-class) |
| fig15 | ROC curves — binary classifiers |
| fig16 | Feature importance — top regression & classification features |

### Enhancement Plots (fig17 – fig26)
| Figure | Description |
|---|---|
| fig17 | Learning curves — regression models |
| fig18 | Learning curves — binary classifiers |
| fig19 | Precision-Recall curves |
| fig20 | Calibration curves (reliability diagrams) |
| fig21 | Permutation importance — regression |
| fig22 | Permutation importance — classification |
| fig23 | Residual distribution (regression) |
| fig24 | Predicted vs. Actual scatter |
| fig25 | Model performance summary heatmap |
| fig26 | Multi-class ROC (one-vs-rest) |

All charts are saved to `finpilot_plots/` and merged into a single high-resolution board (`finpilot_combined_board.png`).

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `scipy` |
| Machine Learning | `scikit-learn` |
| Regression | Ridge, Lasso, RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, SVR |
| Classification | LogisticRegression, RandomForest, ExtraTrees, DecisionTree, GradientBoosting, HistGradientBoosting, MLP, KNN, AdaBoost |
| Evaluation | MAE, RMSE, R², Accuracy, ROC-AUC, Precision-Recall, Calibration |
| Preprocessing | LabelEncoder, StandardScaler, MinMaxScaler, StratifiedKFold |
| Environment | Python 3, Jupyter Notebook / Google Colab |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/dhrumil231/FinPilotAI.git
cd FinPilotAI
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### 3. Add the dataset
Place `Personal_Finance_Dataset_1.csv` in the project root directory.

### 4. Run the notebook
```bash
jupyter notebook FinPilotAI_2_.ipynb
```
Or open directly in **Google Colab** — the notebook includes Colab-compatible display cells for all visualizations.

---

## 📈 Results Summary

| Task | Best Model | Key Metric |
|---|---|---|
| Monthly Expenses Regression | HistGradient Boosting / Gradient Boosting | R², MAE, RMSE |
| Loan Risk (Binary) | Gradient Boosting / MLP | ROC-AUC |
| Savings Tier (Multi-Class) | HistGradient Boosting | Accuracy |

> *Exact metric values depend on dataset version and random seed. Run the notebook to reproduce results.*

---

## 🔭 Future Enhancements

- [ ] Integrate XGBoost / LightGBM / CatBoost for benchmark comparison
- [ ] Build a Streamlit web app for the FinPilot Budget Advisor
- [ ] Add SHAP explainability for model transparency
- [ ] Time-series forecasting on `record_date` for income/expense trends
- [ ] Hyperparameter tuning with Optuna or GridSearchCV
- [ ] Deploy as an API (FastAPI + Docker)

---

## 👤 Author

**Dhrumil Shah**
MS Engineering Management — Syracuse University  
[GitHub](https://github.com/dhrumil231) · [LinkedIn](https://www.linkedin.com/in/dhrumilshah231/)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
