# 🧠 Smart Customer Insights
### Data Cleaning · Exploratory Analysis · K-Means Segmentation

---

## 📌 Overview

This project demonstrates an **end-to-end data science pipeline** applied to a real-world-style messy customer dataset. It covers data inspection, cleaning, exploratory data analysis (EDA), feature engineering, and unsupervised machine learning (K-Means clustering) to produce actionable customer segments.

---

## 🗂️ Project Structure

```
smart_customer_insights/
│
├── customer_data.csv                    ← Raw dataset (messy, real-world style)
├── customer_insights_analysis.ipynb     ← Main Jupyter Notebook (all code + insights)
├── requirements.txt                     ← Python dependencies
├── README.md                            ← This file
│
└── output/
    ├── customer_data_cleaned.csv        ← Cleaned dataset (after all preprocessing)
    ├── customer_data_segmented.csv      ← Final dataset with cluster labels
    ├── cluster_summary.csv              ← Cluster profile statistics
    ├── 00_executive_dashboard.png       ← High-level summary dashboard
    ├── 01_numeric_distributions.png     ← Histograms + boxplots
    ├── 02_scatterplots.png              ← Pairwise scatterplots
    ├── 03_correlation_heatmap.png       ← Correlation matrix
    ├── 04_categorical_distributions.png ← Bar charts (Gender, Region, Category)
    ├── 05_avg_spend_by_segment.png      ← Avg spend by category & region
    ├── 06_spend_by_gender_violin.png    ← Violin plot by gender
    ├── 07_elbow_silhouette.png          ← Optimal k selection charts
    ├── 08_cluster_scatter.png           ← PCA + spend/frequency cluster plots
    ├── 09_cluster_profiles.png          ← Cluster feature mean bar charts
    └── 10_segment_donut.png             ← Segment size doughnut chart
```

---

## 🎯 Objectives

1. **Load & Inspect** — understand the raw data structure, types, and quality issues.
2. **Clean** — fix type errors, impute missing values, cap outliers, normalise categories.
3. **EDA** — explore distributions, relationships, and patterns with visualisations.
4. **Feature Engineer** — create `CustomerValueScore`, `AgeGroup`, encode categoricals.
5. **Segment** — use K-Means clustering to identify distinct customer groups.
6. **Insights** — translate clusters into actionable business recommendations.

---

## 🛠️ Methods & Tools

| Task | Tool / Method |
|------|--------------|
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Outlier handling | IQR-based capping (Winsorisation) |
| Missing values | Median imputation (numeric), mode / 'Unknown' (categorical) |
| Scaling | `StandardScaler` |
| Clustering | `KMeans` (k-means++, n_init=50) |
| Optimal k | Elbow method + Silhouette score |
| Dimensionality reduction | `PCA` (2-component projection for visualisation) |

---

## 📊 Key Findings

- **Frequency and PurchaseAmount are strongly correlated (r ≈ 0.97)** — customers who buy often also spend more.
- **Electronics** buyers have the highest average spend; **Groceries** buyers the lowest.
- **Regional differences in spend are minimal** — product category is a stronger predictor.
- **K-Means identified 4–5 distinct segments** (optimal k chosen by silhouette score).

---

## 👥 Customer Segments

| Segment | Description | Strategy |
|---------|-------------|----------|
| 🏆 Champions | Highest spend & frequency | Retain — VIP programs, referrals |
| ⭐ Loyal Customers | Regular buyers, solid spend | Upsell — subscriptions, bundles |
| 🌱 Potential Loyalists | Growing engagement | Nurture — personalised recommendations |
| ⚠️ At-Risk Customers | Declining engagement | Re-engage — win-back campaigns |
| 💤 Low-Engagement | Occasional / one-time buyers | Activate — onboarding, first-purchase offers |

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Open the notebook

```bash
jupyter notebook customer_insights_analysis.ipynb
```

Or open in **VS Code** → install the *Jupyter* extension → open `.ipynb` file.

### 3. Run all cells

`Kernel → Restart & Run All`  
All outputs will be saved to the `output/` folder automatically.

---

## 📦 Requirements

See `requirements.txt`. Core packages:
- Python ≥ 3.8
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- jupyter / notebook

---

## 💡 Skills Demonstrated

- ✅ Data cleaning & messy dataset handling
- ✅ Outlier detection & treatment (IQR / Winsorisation)
- ✅ Missing value imputation strategies
- ✅ Categorical normalisation
- ✅ Exploratory Data Analysis (EDA)
- ✅ Feature engineering
- ✅ K-Means clustering & hyperparameter tuning
- ✅ PCA for visualisation
- ✅ Silhouette score evaluation
- ✅ Business-oriented insights & recommendations
- ✅ Professional, portfolio-ready code with embedded documentation

---

*Built with Python · Pandas · Scikit-learn · Matplotlib · Seaborn*
