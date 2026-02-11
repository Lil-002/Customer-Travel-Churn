# Customer Travel Churn Prediction System

A complete end-to-end Machine Learning pipeline for predicting customer churn in a travel company, including model benchmarking, SHAP explainability, and a deployable Flask web application.

---

## Project Overview

This project addresses a key business problem: a **23% customer churn rate** in a travel company.

Losing nearly 1 in 5 customers impacts revenue, increases acquisition costs, and weakens long-term growth. The company previously had no predictive mechanism to identify at-risk customers.

This system provides:

- A full ML pipeline (data → preprocessing → training → evaluation)
- Benchmark comparison across 4 models
- SHAP-based model explainability
- A deployable Flask web application prototype
- Business-oriented churn interpretation

---

## Business Objective

The company required:

- A predictive model achieving:
  - ≥ 75% F1-score  
  - ≥ 80% Recall  
- Identification of key churn drivers  
- A deployable prototype suitable for CRM integration  

The primary evaluation metric chosen was **F1-score**, since churn data is imbalanced and both Precision and Recall are important.

---

## Dataset Description

The dataset (`Customertravel.csv`) contains anonymised customer data with:

### Features

- Age
- FrequentFlyer
- AnnualIncomeClass
- ServicesOpted
- AccountSyncedToSocialMedia
- BookedHotelOrNot

### Target Variable

- `1` → Churned  
- `0` → Stayed  

Observed churn rate: **~23%**

---

## Repository Structure

```
Customer-Travel-Churn/
│
├── data_preparation.py
├── model_training.py
├── shap_analysis.py
├── EDA_Analysis.py
├── app.py
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   └── js/app.js
│
├── Customertravel.csv
├── CustTravelChurn.pdf
└── README.md

```

---

## Methodology (CRISP–DM Framework)

### Data Understanding

Exploratory Data Analysis was conducted using:

- Target distribution visualisation
- Age group churn rates
- Categorical churn rate comparisons
- Statistical summaries

(See `EDA_Analysis.py`)

---

### Data Preparation

Pipeline includes:

- Missing value handling (mode for categorical, median for numeric)
- One-hot encoding (`drop='first'`)
- Standard scaling for numerical features
- Stratified 80/20 train-test split
- Prevention of data leakage
- Feature metadata saving for SHAP compatibility

(See `data_preparation.py`)

---

## Model Development

Four models were trained using identical preprocessing and split:

| Model | Purpose |
|--------|---------|
| Logistic Regression | Baseline |
| Random Forest | Tree-based ensemble |
| Gradient Boosting | Boosted ensemble |
| Artificial Neural Network | Deep learning benchmark |

(See `model_training.py`)

### Key Design Decisions

- `class_weight='balanced'` for imbalance handling
- Deterministic seeds (42) for reproducibility
- No validation split for ANN (fair benchmarking)
- Model performance saved for deployment

---

## Model Performance

### Best Performing Model: Random Forest

- **F1-score:** 75.5%
- **Recall:** 82.2%
- Meets project success criteria

The Random Forest model was selected as the production model.

---

## Model Explainability (SHAP)

SHAP was used to:

- Explain individual predictions
- Compute global feature importance
- Translate technical results into business-friendly insights

### SHAP Explainers Used

- TreeExplainer → Random Forest / Gradient Boosting
- LinearExplainer → Logistic Regression
- KernelExplainer → ANN

(See `shap_analysis.py`)

### Key Insight

FrequentFlyer status was the strongest churn driver, followed by Age.

---

## Web Application Deployment

A working Flask prototype was built.

- Backend: `app.py`
- Frontend: `index.html`
- JavaScript: `app.js`

### Features

- Customer input form
- Real-time churn probability prediction
- Risk visualisation (probability circle)
- SHAP top-10 feature contribution display
- Business insight suggestions
- Model health endpoint
- Production-ready API structure

---

##  How to Run the Project

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:

* pandas
* numpy
* scikit-learn
* tensorflow
* shap
* flask
* matplotlib
* seaborn
* joblib

---

### Step 2: Prepare Data

```
python data_preparation.py
```

This will:

* Split the dataset
* Encode and scale features
* Save processed files into /prepared_data
* Save metadata for SHAP

---

### Step 3: Train Models

```
python model_training.py
```

This will:

* Train all 4 models
* Evaluate performance
* Save best model to /models
* Save best_model_info.json

---

### Step 4: Run SHAP Analysis

```
python shap_analysis.py
```

This generates:
* SHAP summary plots
* Global feature importance
* Business interpretation tables

---
 
### Step 5: Launch Web App

```
python app.py
```

Then open your browser and navigate to:

```
http://127.0.0.1:5000/
```

The web dashboard allows you to:
* Input customer details
* Generate churn probability
* View SHAP explanation
* Inspect risk drivers
* Review model performance

---

## Business Impact

This solution enables:

### Revenue Stabilisation
Identify high-risk customers before they leave.

### Targeted Retention
Focus retention campaigns on customers most likely to churn.

### Strategic Insight
Understand which customer behaviours influence churn risk.

### Operational Efficiency
Reduce unnecessary blanket marketing costs.

---

## Technical Highlights

* Full end-to-end ML lifecycle
* Reproducible pipeline
* Multiple model benchmarking
* Explainable AI integration
* Production-style deployment
* Clean separation of backend and frontend
* Deterministic model training
* Metadata-driven SHAP compatibility

---

## Limitations

* Dataset limited to narrow age band (27–38)
* Small dataset size
* No hyperparameter tuning via GridSearch
* No live CRM integration
* No cloud deployment (AWS / Azure)

---

## Future Improvements

* Hyperparameter optimisation
* Cross-validation
* Cloud deployment
* Real-time CRM integration
* Monitoring pipeline (MLOps)

---

## Author

[Ethan Ong]
AI / Data Science Portfolio Project
