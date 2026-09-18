# Capstone 24.1 — Final Report: Macroeconomic Stress Scenarios & Portfolio Risk

## Project Title
Agentic Macro Stress Scenarios: Measurable Portfolio Risk Impact (Final Report)

## Author
Imran Duggal

## Executive summary
Investment teams need a practical answer to a simple question: if markets enter a stress episode like the Global Financial Crisis or the COVID crash, how much worse does portfolio risk become? This project builds a repeatable baseline for answering that question and for checking future scenarios proposed by an agentic workflow (an automated assistant that suggests macroeconomic shocks).

The workflow combines public macroeconomic series with a simple 60/40 SPY/TLT portfolio. It labels two historical stress windows, measures changes in volatility, drawdown, Value at Risk (VaR), and Expected Shortfall (ES), and trains several models to recognize stress from lagged macroeconomic data. Gradient Boosting was selected after StratifiedKFold GridSearchCV, with mean cross-validation ROC-AUC of 0.999 ± 0.001, test ROC-AUC of 0.998, and stress-class F1 of 0.93. The results provide measurable acceptance criteria for future agent-generated scenarios rather than relying on narrative judgment alone.

## Rationale

### Problem, goals, and benefits
Manual stress-scenario writing is slow and difficult to audit. A generated scenario is useful only if it is coherent and if its effect on portfolio risk can be measured and repeated. The project therefore aims to:

- combine rates, inflation, unemployment, market volatility, and portfolio prices in one clean historical dataset;
- label known stress windows and compare them with non-stress periods;
- show the risk change in numbers, not just in a written market story;
- compare multiple classification algorithms using a consistent evaluation process; and
- create a reusable baseline against which later agent-proposed shocks can be scored.

The main challenges are rare stress labels (208 stress rows versus 5,614 non-stress rows), macro series released on different calendars, and too few historical episodes for a reliable time-based holdout. The benefit is a transparent starting point for risk teams: it shows what a plausible stress signal and risk delta look like, while making the limitations visible.

## Research Question
Can an agentic workflow generate coherent macroeconomic stress scenarios and quantify how those scenarios change a portfolio’s risk metrics in a way that is measurable and repeatable?

## Data Sources

The project uses multiple sources because no single source describes the complete stress picture:

| Source | Series / tickers | Why included |
|---|---|---|
| FRED | **FEDFUNDS** | Policy-rate path and changes in financing conditions |
| FRED | **CPIAUCSL** | Inflation pressure and the real-rate/equity-bond trade-off |
| FRED | **UNRATE** | Labor-market stress, which tends to rise in recessions |
| FRED | **VIXCLS** | Implied equity volatility, a widely used market fear gauge |
| Yahoo Finance | **SPY**, **TLT** | Liquid equity and long-Treasury ETFs for a simple 60/40 portfolio proxy |
| Manual labels | 2008-09–2009-03 and 2020-02–2020-04 | Historical stress windows for supervised learning |

Cached inputs are in [`data/`](data/). The first notebook provides exploratory charts and the visualization of data potential: it shows how macro variables, market volatility, and portfolio behavior move around stress windows. The notebooks reload cached files and only download a missing file.

## Methodology

### Model outcomes
This is a **supervised classification** problem. Each row is labeled as stress or non-stress. The expected model output is both:

- **P(stress):** a probability that the row represents a stress day; and
- a **stress label** after applying a decision threshold.

The probability is especially useful for ranking and scenario monitoring, while the label supports a simple alert.

### Preprocessing and features

1. FRED data and SPY/TLT prices are downloaded or loaded from cache.
2. All series are aligned to the trading-day calendar of the portfolio prices.
3. Sparse monthly macro values are forward-filled onto daily dates.
4. Missing values are handled by dropping rows that still lack the required engineered features after alignment and filling.
5. Features include daily returns, 60/40 portfolio returns, 21-day annualized portfolio volatility, cumulative drawdown, rate and unemployment changes, VIX changes, and one-day lags of the macro features. The lags help prevent using same-day information to predict that day.
6. `is_stress = 1` is assigned inside the two historical windows.

A chronological train/test split was tested but failed as a useful evaluation because its test period contained zero stress days. The final workflow uses a stratified 75/25 train/test split and **StratifiedKFold** cross-validation so both classes appear in the evaluation folds. This is a practical compromise for the small number of historical stress episodes, not proof of performance on a genuinely new crisis.

### Algorithms chosen
The final analysis compares multiple models, each tuned with **GridSearchCV** and five-fold **StratifiedKFold** cross-validation:

- **Logistic Regression:** an interpretable, class-balanced linear baseline.
- **Random Forest:** a tree ensemble that can capture non-linear feature interactions.
- **Gradient Boosting:** sequential trees that often perform well on structured tabular data.

### Evaluation
The primary metric is **ROC-AUC** because stress is rare and accuracy could look high even for a model that misses crises. ROC-AUC measures how well the predicted probabilities rank stress days above calm days without depending on one arbitrary cutoff. Stress precision, recall, and F1 are secondary metrics because they show how many alerts are correct and how many stress days are caught.

## Results

### Model results
The final model results are recorded in [`data/final_model_results.json`](data/final_model_results.json). Gradient Boosting was selected because it had the highest mean cross-validation ROC-AUC, very low fold-to-fold variation, and a strong stress-class F1. Its tree structure can capture non-linear interactions among lagged VIX, rolling portfolio volatility, and unemployment changes that a linear model only partly captures.

| Model | CV ROC-AUC (mean ± std) | Test ROC-AUC | Stress F1 | Best parameters (short) |
|---|---:|---:|---:|---|
| Logistic Regression | 0.951 ± 0.019 | 0.948 | 0.44 | C=10, L2, lbfgs |
| Random Forest | 0.999 ± 0.001 | 0.999 | 0.86 | 100 trees, unlimited depth, min leaf=5 |
| **Gradient Boosting (selected)** | **0.999 ± 0.001** | **0.998** | **0.93** | learning rate=0.1, max depth=3, 200 trees |

For the selected model, stress precision is **0.979** and recall is **0.885**. The exact values are available in the JSON results file.

### Portfolio risk deltas
Stress periods show materially worse portfolio outcomes than non-stress periods:

| Metric | Stress | Non-stress | Change in stress |
|---|---:|---:|---:|
| Mean annualized 21-day volatility | 27.1% | 8.7% | +18.4 percentage points |
| Mean drawdown | -14.5% | -3.6% | 11.8 percentage points deeper |
| Daily 5% VaR | -3.37% | -0.94% | 2.43 percentage points worse |
| Daily 5% Expected Shortfall | -4.29% | -1.36% | 2.93 percentage points worse |

Mean VIX was approximately 46 during stress versus approximately 18 outside the labeled windows. These deltas are the historical benchmark for assessing whether a future agent-generated scenario produces a plausible direction and scale of risk change.

### Chronological caveat
The chronological 80/20 holdout contained **zero stress days**, so it could not measure out-of-time stress detection. The stratified split and StratifiedKFold procedure are required for this dataset, but the very high tree-model AUCs should not be treated as proof that the model will generalize to a new crisis. Rolling volatility and VIX already spike inside the labeled windows, and more independent stress episodes are needed for a stronger chronological test.

## Next steps

- Generate agent-proposed shocks to rates, VIX, unemployment, and other macro variables, then score them against the historical risk deltas.
- Add more stress episodes and test how sensitive the results are to label boundaries.
- Use purged or expanding-window cross-validation when enough labeled episodes are available.
- Extend the portfolio beyond SPY/TLT and add richer scenario-level risk metrics.
- Calibrate predicted probabilities if the model will drive threshold-based alerts.

## Outline of project

- [`notebooks/01_initial_report_eda.ipynb`](notebooks/01_initial_report_eda.ipynb) — exploratory data analysis, visualizations, and the initial logistic baseline.
- [`notebooks/02_final_models.ipynb`](notebooks/02_final_models.ipynb) — final preprocessing, multiple models, GridSearchCV, evaluation, and portfolio risk deltas.
- [`data/`](data/) — cached FRED and SPY/TLT inputs plus `final_model_results.json`.
- `requirements.txt` — Python dependencies.

### Project layout

```text
capstone-macro-stress-eda/
  README.md
  requirements.txt
  notebooks/
    01_initial_report_eda.ipynb
    02_final_models.ipynb
  data/
  .venv/                  # local environment; do not commit
```

### How to re-run

```bash
cd capstone-macro-stress-eda
source .venv/bin/activate   # or recreate it from requirements.txt
jupyter execute notebooks/02_final_models.ipynb
```

The `.venv` directory is a local virtual environment and should not be committed.

## Contact
Imran Duggal
