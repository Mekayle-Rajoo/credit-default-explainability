# Credit Default Prediction: The Cost of Explainability in Consumer Credit Risk

Predicting loan default on the Home Credit Default Risk dataset, and testing whether an interpretable model (EBM) can match a black-box model (LightGBM).

Built as practice for the COS802 project in the MIT Big Data Science programme at the University of Pretoria.

## Key results

Mean of stratified 5-fold cross-validation. PR-AUC is the main metric because only 8.07% of applicants defaulted, so a random model scores 0.081.

| Model | Features | PR-AUC | ROC-AUC |
|---|---|---|---|
| Dummy (baseline) | - | 0.081 | 0.500 |
| Logistic regression | Application only (113) | 0.221 | 0.747 |
| Logistic regression | Full (158) | 0.244 | 0.765 |
| LightGBM | Application only (113) | 0.241 | 0.757 |
| LightGBM | Full (158) | 0.269 | 0.778 |
| EBM | Full (158) | 0.270 | 0.779 |

- **Credit history helps.** Adding the six supporting tables raised PR-AUC by 0.023 for logistic regression and 0.028 for LightGBM. That is a bigger gain than switching from logistic regression to LightGBM.
- **The interpretable model cost nothing measurable.** EBM matched LightGBM (0.270 vs 0.269 PR-AUC), well within the variation between folds. The cost was training time: about 5 times longer per fold.
- **Both models used gender.** The external credit scores and repayment history drove both models, and 13 of their top 15 features overlap. Both also relied on `CODE_GENDER`. EBM showed this directly in its own terms, while LightGBM needed SHAP to reveal it.
## Research questions

- **RQ1:** How much do bureau and repayment history tables improve default prediction over application data alone?
- **RQ2:** How much predictive performance is lost when an inherently interpretable model (EBM) is used instead of LightGBM?

## Data

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) on Kaggle: 307,511 labelled applicants across seven tables (about 58 million rows). The data is not included in this repo.

Download it from Kaggle (you need to accept the competition rules first) and put these files in `data/`:

```
application_train.csv
bureau.csv
bureau_balance.csv
previous_application.csv
POS_CASH_balance.csv
credit_card_balance.csv
installments_payments.csv
```

With the Kaggle API set up, this does the same thing:

```
kaggle competitions download -c home-credit-default-risk -p data
```

`application_test.csv` is not used because it has no labels.

## How to run

1. Use Python 3.13 and install the packages:
   ```
   pip install -r requirements.txt
   ```
2. Put the CSV files in `data/`.
3. Run the notebooks in order:

| Notebook | What it does |
|---|---|
| `01_eda.ipynb` | Data quality checks on all seven tables, anomalies, target imbalance and correlations. Saves the raw tables as parquet. |
| `02_feature_engineering.ipynb` | Cleaning, three ratio features, and aggregation of the six supporting tables to one row per applicant. Saves `features_app.parquet` (113 features) and `features_full.parquet` (158 features). |
| `03_modelling.ipynb` | 5-fold cross-validation for the dummy, logistic regression, LightGBM and EBM models, then SHAP and EBM feature importance. |

Notebook 03 takes roughly 40 minutes on a laptop, mostly for EBM.

## Repository structure

```
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modelling.ipynb
├── report/
│   ├── report.tex
│   └── report.pdf
├── presentation/
│   └── presentation.pdf
└── data/                (not tracked, see Data)
```

## Report and presentation

- [Report (ACM KDD format)](report/report.pdf)
- [Presentation](presentation/presentation.pdf)

## Limitations

The models were not tuned, EBM used fewer bagging rounds than its default to save time, gender and age are still in the features, and the data is not South African. More detail is in the report.

## Built with

pandas, scikit-learn, [LightGBM](https://github.com/microsoft/LightGBM), [InterpretML](https://github.com/interpretml/interpret) (EBM) and [SHAP](https://github.com/shap/shap).

## Author

Mekayle Rajoo, [LinkedIn](https://www.linkedin.com/in/mekayle-rajoo-7504161a5/)
