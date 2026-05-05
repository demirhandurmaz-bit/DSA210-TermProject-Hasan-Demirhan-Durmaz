# DSA210 Term Project — AI Hype Cycles & Stock Market Behavior

**Hasan Demirhan Durmaz** | Sabancı University — DSA210 Introduction to Data Science

---

## Project Overview

This project investigates whether public AI hype — measured through Google Trends search volume and Reddit mention counts — has a measurable effect on the stock prices of AI-exposed companies (NVDA, MSFT, AAPL).

The central question: **Does AI hype drive stock prices, or do stock prices drive AI hype?**

To answer this, I build a composite "AI Hype Index" from two independent signals, correlate it with abnormal stock returns around major AI product launches, and apply machine learning to model the relationship.

---

## Hypothesis

> AI hype, as captured by Google Trends and Reddit activity, is positively correlated with short-term abnormal returns and elevated volatility in AI-exposed stocks — particularly around major AI product launch events.

---

## Data Sources

| Source | What it provides | Period |
|--------|-----------------|--------|
| Yahoo Finance (`yfinance`) | Daily closing prices, log returns, rolling volatility | 2020–2024 |
| Google Trends (`pytrends`) | Weekly search interest for "artificial intelligence", "ChatGPT", "machine learning", "Nvidia AI" | 2020–2024 |
| Reddit JSON API | Daily post counts from r/MachineLearning, r/artificial, r/stocks, r/investing | 2020–2024 |

The two hype signals are normalized to [0, 1] and averaged into a single **composite Hype Index**.

---

## AI Events Studied

| Event | Date |
|-------|------|
| GPT-3 Release | 2020-06-11 |
| GitHub Copilot | 2021-06-29 |
| DALL-E 2 | 2022-04-06 |
| ChatGPT Launch | 2022-11-30 |
| GPT-4 Release | 2023-03-14 |
| Llama 2 | 2023-07-18 |
| GPT-4o Release | 2024-05-13 |

---

## Project Structure

```
DSA210-TermProject-Hasan-Demirhan-Durmaz/
│
├── Project/
│   └── Notebook/
│       ├── 01_data_collection.ipynb     # Data pulling & merging
│       ├── 02_eda.ipynb                 # Exploratory data analysis
│       ├── 03_hypothesis_testing.ipynb  # Statistical tests
│       └── 04_ml_models.ipynb          # Machine learning models
│
├── data/
│   ├── raw/                            # Raw downloaded data
│   └── processed/                      # Merged & cleaned datasets
│
├── figures/                            # All generated plots
├── requirements.txt
└── README.md
```

---

## Methodology

### Milestone 1 — Data & Analysis

**Exploratory Data Analysis**
- Time series plots of hype index vs stock prices
- Correlation heatmaps across all signals
- Event-aligned visualizations around AI product launches

**Hypothesis Testing**
- **H1:** Hype periods → higher returns (Welch t-test)
- **H2:** Hype periods → higher volatility (Levene test)
- **H3:** Hype(t) predicts return(t+1) (Pearson correlation + Granger causality)
- **H4:** Post-hype price correction exists (Event study CAR + one-sample t-test)

### Milestone 2 — Machine Learning

**Regression**
- OLS, Ridge, Lasso — predicting 5-day forward returns from hype features
- Lasso feature selection across all three tickers

**Classification** (target: `is_hype_period`)
- Logistic Regression
- K-Nearest Neighbors (k tuned via CV)
- Decision Tree (depth tuned via CV)
- Bagging
- Random Forest + feature importance analysis
- Gradient Boosted Decision Trees + learning curve
- Voting Ensemble (soft vote: LR + RF + GBT)

All classifiers evaluated with **Stratified 5-Fold CV** — CV AUC ± std reported for every model.

**Clustering**
- K-Means (elbow method + silhouette score for k selection)
- Hierarchical Clustering (Ward linkage, dendrogram)
- K-Means vs Hierarchical comparison (Adjusted Rand Index)

---

## Key Findings

- **Hype periods are associated with elevated volatility** across all three tickers — the effect is stronger for NVDA than MSFT or AAPL, consistent with NVDA's direct exposure to AI infrastructure demand.
- **Linear regression explains very little variance** in forward returns (low R²), consistent with market efficiency at the linear level. The relationship is nonlinear.
- **Ensemble models (Random Forest, Gradient Boosting) substantially outperform** logistic regression and a single decision tree at classifying hype periods, confirming that hype dynamics involve nonlinear feature interactions.
- **Clustering naturally recovers market regimes** that align with the hype cycle — pre-ChatGPT calm, the post-November 2022 hype surge, and the post-peak normalization.

---

## How to Run

```bash
# clone the repo
git clone https://github.com/<your-username>/DSA210-TermProject-Hasan-Demirhan-Durmaz.git
cd DSA210-TermProject-Hasan-Demirhan-Durmaz

# install dependencies
pip install -r requirements.txt

# run notebooks in order
jupyter notebook
```

Run notebooks in this order:
1. `01_data_collection.ipynb`
2. `02_eda.ipynb`
3. `03_hypothesis_testing.ipynb`
4. `04_ml_models.ipynb`

---

## Dependencies

See `requirements.txt`. Main libraries:
`yfinance`, `pytrends`, `requests`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `statsmodels`

---

## Milestones

| Milestone | Tag | Contents |
|-----------|-----|----------|
| Milestone 1 | `milestone1` | Data collection, EDA, hypothesis testing |
| Milestone 2 | `milestone2` | ML models, clustering |
