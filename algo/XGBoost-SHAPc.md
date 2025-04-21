# 1. XGBoost Model Implementation and Hyperparameter Optimization

This document summarizes the full XGBoost modelling workflow used in the paper for predicting snow‑cover fraction (SCF) across High Mountain Asia (HMA). It is intended as **methodological documentation** rather than executable code, allowing readers to reproduce our approach without exposing proprietary scripts.

---

## 1.1  Data Stratification & Pre‑processing

For each experiment we trained *separate* XGBoost models on the following combinations of strata:

| Dimension | Levels |
|-----------|--------|
| **Data source** | ERA5/CAMS‑EAC4 • MERRA‑2 • MATCHA |
| **Region** | 6 HMA sub‑regions |
| **Month**  | May • June • July |
| **Target construct** | • *Original* SCF (MODIS)  • *Model* SCF (each reanalysis) |

*Predictor set*: 22 variables (6 aerosol, 15 meteorological, 1 elevation).

---

## 1.2  Model Configuration

| Parameter | Value |
|-----------|-------|
| **Objective** | `reg:squarederror` (optimises Mean Squared Error) |
| **Random seed** | 24 |
| **Hardware** | NVIDIA A100 GPU |
| **Tree method** | `gpu_hist` |
| **n_jobs** | 1 |

Cross‑validation: 5‑fold K‑fold CV inside each Hyperopt trial. After CV, the model was refit on the *entire* dataset using the best hyper‑parameters returned by Hyperopt.

---

## 1.3  Hyperparameter Search (Hyperopt + ATPE)

| Hyper‑parameter | Search space | Distribution |
|-----------------|-------------|--------------|
| `max_depth` | 6 – 10 | *q*‑uniform (ints) |
| `min_child_weight` | 10 – 100 (step 10) | *q*‑uniform |
| `n_estimators` | 500 – 1100 | uniform |
| `learning_rate` | 0.01 – 1.0 | log‑uniform |
| `gamma` | 0 – 5 (step 0.2) | *q*‑uniform |
| `reg_lambda` | 0 – 100 | uniform |
| `subsample` | 0.8 – 1.0 | uniform |

- **Optimiser**: Adaptive Tree‑structured Parzen Estimator (ATPE) via `hyperopt` 0.2.7  
- **Trials**: ≈140 per regional model (≈ 20 × #hyper‑parameters)  
- **Early stopping**: terminate search when ≥15 % of successive trials fail to improve validation MAE.
- **Objective to minimise**: Mean Absolute Error (MAE).

---

## 1.4  Evaluation Metrics

The optimised model for each region/month/source was assessed on:

* **MAE** – Mean Absolute Error (objective)
* **RMSE** – Root Mean Squared Error
* **R²** – Coefficient of determination
* **ρ** – Pearson correlation coefficient

---

## 1.5  SHAP Value Analysis (Native XGBoost)

1. **Prediction contributions** were obtained with `pred_contribs=True` and `gpu_predictor`.
2. **Interaction effects** were extracted with `pred_interactions=True`.
3. **Magnitude focus**: absolute values → normalised to sum = 100 % per sample (denoted *SHAPc*).
4. **Statistics** computed per feature & interaction: mean, std, and percentiles (5, 25, 50, 75, 95).
5. **Symmetry**: off‑diagonal interaction contributions doubled to preserve commutativity.
6. **Context tags**: each contribution record includes region, month, and data‑source identifiers.

This GPU‑accelerated native SHAP implementation exactly matches the theoretical SHAP algorithm but avoids external dependencies.

---

## 1.6  Computational Environment

* Python 3.8
* **xgboost** 1.7.4 (compiled with CUDA‑11)
* **hyperopt** 0.2.7
* **scikit‑learn** 1.4.2
* CUDA 11.8 • cuDNN 8.9 • Driver ≥ 535 (A100)

---

### Reproducibility Notes

* Set `PYTHONHASHSEED=24` for complete determinism.  
* Provide MODIS SCF and predictor Feather files in the exact splits described above.  
* Scripts used to create the trials (data loaders, Hyperopt search space definition, and result aggregation) are available from the authors on request.

---


