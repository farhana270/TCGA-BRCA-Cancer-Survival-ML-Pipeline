# 🔬 TCGA-BRCA Cancer Survival ML Pipeline

> **Can clinical data alone predict whether a breast cancer patient will survive beyond 5 years?**  
> This project builds a reproducible, end-to-end machine learning pipeline to answer that question — going from raw genomic clinical data all the way to statistically validated survival curves.

---

## 📌 The Problem

Breast cancer is the most common cancer among women worldwide. When a patient is diagnosed, one of the most critical questions is: *how aggressive is this cancer likely to be?*

Oncologists use tumor stage, age, lymph node involvement, and treatment history to make this judgment — but these assessments are often subjective and experience-dependent. A data-driven model that can **automatically stratify patients into risk groups** could help clinicians prioritize monitoring, escalate treatment for high-risk patients sooner, and allocate resources more effectively.

**This pipeline asks:** Using only structured clinical data available at or shortly after diagnosis, can we predict which patients are at high risk of dying within 5 years?

---

##  Objective

Train and compare three machine learning classifiers on **TCGA-BRCA clinical data** (The Cancer Genome Atlas — Breast Invasive Carcinoma) to:

1. **Predict 5-year mortality risk** from clinical features alone (age, stage, lymph nodes, treatment type, cancer subtype)
2. **Compare model performance** using AUC-ROC and Average Precision, with proper cross-validation
3. **Validate predictions biologically** by showing that model-predicted risk groups have statistically different survival outcomes via Kaplan-Meier curves

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [GDC Data Portal](https://portal.gdc.cancer.gov) — TCGA-BRCA project |
| Patients | 1,036 (after cleaning) |
| Raw files | `clinical.tsv`, `follow_up.tsv`, `pathology_detail.tsv` |
| High-risk patients (died ≤5yr) | 102 (9.8%) |
| Low-risk patients | 934 (90.2%) |

The dataset covers breast cancer patients diagnosed across multiple US institutions, with follow-up times ranging from a few days to over 20 years.

---

##  Why These Methods?

### Feature Engineering — not just raw columns

The GDC data is messy: missing values encoded as `'--`, ages stored in days, 5,546 rows that actually represent only 1,098 patients (one row per treatment received). Before any modeling, the pipeline:

- **Deduplicates** patients (multiple treatment rows → one row per patient)
- **Converts** age from days to years
- **Maps** AJCC tumor stage strings to ordinal numbers (Stage I=1 through Stage IV=4)
- **One-hot encodes** race and cancer subtype
- **Aggregates** all treatments a patient received into binary flags (did they get chemo? radiation? hormone therapy?)
- **Merges** menopause status from follow-up data and lymph node counts from pathology data

This results in **20 clean features** — no raw columns, no leakage.

### Why Binary Classification (not regression)?

We define a binary target: `high_risk = 1` if the patient died within 1,825 days (5 years), else `0`. This is a clinically meaningful threshold — 5-year survival is the standard benchmark in oncology for declaring a patient "disease-free."

### Why Three Different Models?

Each model has a different inductive bias, and comparing them tells us something about the data:

| Model | Why we used it |
|---|---|
| **Logistic Regression** | Linear baseline — interpretable coefficients, shows which features linearly predict risk |
| **Random Forest** | Non-linear ensemble — captures feature interactions, gives feature importances, robust to outliers |
| **SVM (RBF kernel)** | Non-linear, margin-based — good for high-dimensional spaces with class imbalance |

All three use `class_weight='balanced'` because the dataset is heavily imbalanced (9.8% positive rate). Without this, a naive model learns to predict "low risk" for everyone and achieves 90% accuracy while being completely useless.

### Why Pipelines with StandardScaler inside?

Fitting the scaler on the full dataset *before* the train/test split would let the model "see" test data statistics during training — a form of **data leakage**. By wrapping the scaler inside a `sklearn.Pipeline`, it only ever fits on training data (or training folds during cross-validation). This is a subtle but important ML engineering best practice.

### Why AUC-ROC instead of Accuracy?

With 9.8% positive rate, a model that always predicts "low risk" achieves **90.2% accuracy** while being completely clinically useless. AUC-ROC measures the model's ability to *rank* patients correctly — it's threshold-independent and robust to class imbalance. Average Precision (AP) is even stricter, focusing specifically on minority class performance.

### Why Kaplan-Meier curves?

AUC tells us the model ranks patients correctly. But does that ranking correspond to *real differences in survival*? Kaplan-Meier curves answer this directly: split patients by predicted risk group and plot their actual survival trajectories. If the curves separate significantly — the model is finding something biologically real, not just fitting noise.

---

##  Results

### Model Performance

| Model | CV AUC (5-fold) | Test AUC | Test AP |
|---|---|---|---|
| **Random Forest** | **0.803 ± 0.031** | **0.716** | **0.297** |
| SVM | 0.803 ± 0.030 | 0.713 | 0.224 |
| Logistic Regression | 0.769 ± 0.037 | 0.700 | 0.211 |

> **Context:** AUC 0.70–0.80 on clinical-only TCGA-BRCA data is consistent with published literature. Studies using genomic features (gene expression, mutations) typically reach 0.80–0.90. The ceiling for clinical-only models is well-established — the value here is the pipeline, not just the number.

### Top Predictive Features (Random Forest)

The three most important features were: **age at diagnosis**, **lymph nodes positive**, and **tumor stage** — all of which are well-established prognostic factors in breast cancer biology. This is a strong sanity check: the model learned the right things.

### Survival Curve Validation — The Key Result

![KM Summary Panel](outputs/plots/km_summary_panel.png)

**Panel A (top left)** shows the main result: patients predicted as high-risk by the Random Forest model have dramatically worse survival than those predicted as low-risk. The **log-rank test p = 0.0001** (highly significant) confirms this is not random variation.

- High Risk group: 26 deaths out of 104 patients (25% mortality)
- Low Risk group: 5 deaths out of 104 patients (4.8% mortality)

**Panel C (bottom left)** validates the data pipeline: Stage I patients (blue) have the best survival, Stage IV (red) the worst — exactly as clinical literature predicts. This confirms the preprocessing is correct.

**Panel D (bottom right)** shows age-stratified curves: the 75+ group drops fastest, consistent with known biology.

---

##  Project Structure

```
tcga-brca-survival-pipeline/
├── data/
│   ├── raw/                    ← GDC downloaded TSV files
│   └── processed/
│       ├── brca_features.csv   ← Clean feature matrix (1,036 × 24)
│       └── test_risk_scores.csv ← Model predictions for KM analysis
│
├── notebooks/
│   ├── 01_eda.ipynb            ← Exploratory data analysis
│   ├── 02_preprocessing.ipynb  ← Feature engineering & cleaning
│   ├── 03_modeling.ipynb       ← Model training, CV, AUC comparison
│   └── 04_survival_analysis.ipynb ← Kaplan-Meier curves
│
├── outputs/
│   ├── plots/                  ← All generated figures (PNG)
│   └── models/                 ← Saved model pipelines (.pkl)
│
├── requirements.txt
└── README.md
```

---

##  How to Reproduce

```bash
# 1. Clone the repository
git clone https://github.com/farhana270/tcga-brca-survival-pipeline
cd tcga-brca-survival-pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data from GDC portal
#    → https://portal.gdc.cancer.gov → Repository → Cases
#    → Filter: Project = TCGA-BRCA, Data Category = Clinical
#    → Place TSV files in data/raw/

# 4. Run notebooks in order
jupyter notebook notebooks/02_preprocessing.ipynb
jupyter notebook notebooks/03_modeling.ipynb
jupyter notebook notebooks/04_survival_analysis.ipynb
```

---

##  Requirements

```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.1
lifelines>=0.27
matplotlib>=3.6
seaborn>=0.12
jupyter
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

##  Honest Limitations

This project is built to be transparent about what it can and cannot claim:

**Why CV AUC (0.80) is higher than Test AUC (0.71):**  
With only 102 positive cases (high-risk patients) in the full dataset, the model's performance has high variance on small test splits. This gap is expected and documented in clinical ML literature for small cohorts — it is not evidence of overfitting in the traditional sense.

**Reverse causation in treatment features:**  
The model identifies "targeted therapy" as a high-risk predictor. This is *reverse causation* — sicker patients are more likely to receive targeted therapy, so the feature correlates with poor outcomes not because therapy causes harm, but because it was prescribed to those already at high risk. This is a known limitation of observational clinical data.

**Clinical-only features:**  
This pipeline uses structured clinical features only. Adding genomic features (gene expression from RNA-seq, somatic mutations) would substantially improve predictive power. The pipeline is designed to make this extension straightforward.

**This is a research pipeline, not a clinical tool.** No model output here should be used to guide actual patient care.

---

## Skills Demonstrated

`pandas` · `scikit-learn` · `lifelines` · `Kaplan-Meier` · `log-rank test` · `ROC-AUC` · `Average Precision` · `stratified cross-validation` · `sklearn.Pipeline` · `class imbalance handling` · `feature engineering` · `survival analysis` · `reproducible ML` · `TCGA data processing` · `GDC portal`

---

##  Author

**Farhana Sayed** — Biomedical data science, computational oncology  
Built as part of a bioinformatics portfolio following research at ACTREC (Advanced Centre for Treatment, Research and Education in Cancer), Mumbai.

---

*Dataset: The Cancer Genome Atlas Breast Invasive Carcinoma (TCGA-BRCA), accessed via GDC Data Portal. This project is for research and educational purposes only.*
