# HX-MAP Framework

## Full name
**HX-MAP** = **Harmonized X-response Marginal Additive Prediction**

## Purpose
HX-MAP translates ensemble machine learning models into transparent, value-specific, point-based clinical decision tools.

## Core philosophy
HX-MAP extends the Feature Mapping Algorithm (FMA) philosophy from feature-level attribution to **value-specific nonlinear effect mapping**.

## Three-step framework

### 1. Quantification of Marginal X-response
For feature `X_j` at value `x`, compute the model-specific marginal predictor response relative to a reference value `x_ref` over the empirical background distribution of all other features:

`Δp_jm(x) = E[f_m(x, X_-j)] - E[f_m(x_ref_j, X_-j)]`

Operationally:
- keep the background cohort fixed
- replace only feature `j` with a candidate value `x`
- predict with model `m`
- compare the average predicted probability with the reference-value average

### 2. Harmonized Ensemble Contribution
For each feature value, harmonize marginal response using:
- feature importance within each base learner: `FI_jm`
- meta-model ensemble coefficient: `β_m`

`C_j(x) = Σ_m [ FI_jm × β_m × Δp_jm(x) ]`

This creates a value-specific contribution curve for each feature.

### 3. Additive Point Scale and Probability Mapping
Each contribution function `C_j(x)` is linearly transformed into a standardized point scale:

`Point_j(x)`

For a patient:
- total contribution = sum of `C_j(x_j)` over all features
- total points = sum of `Point_j(x_j)` over all features
- total contribution is mapped back to predicted probability through the ensemble meta-model link

## Required components
HX-MAP requires:
1. base learners with probability output
2. a stacking meta-model with accessible coefficients/intercept
3. background training data used for marginal-response estimation
4. feature-importance estimates for each base learner
5. a consistent preprocessing pipeline shared by train, test, and marginal-response simulation

## Recommended implementation architecture
- `src/hxmap/config.py`: project settings and feature schema
- `src/hxmap/data.py`: data loading and splitting
- `src/hxmap/train.py`: fit base learners, stacking model, and persist artifacts
- `src/hxmap/mapping.py`: build marginal response, contribution, and point mappings
- `src/hxmap/evaluate.py`: thresholding, test performance, calibration, bootstrap CI
- `src/hxmap/plotting.py`: feature-scale plots, total-points mapping, adaptive nomogram plot
- `src/run_hxmap.py`: main entrypoint with auto-train-if-missing-artifacts logic

## Artifact policy
If required model artifacts do not exist, HX-MAP should automatically:
1. read the raw dataset
2. split train/test with fixed random state
3. fit preprocessing
4. train base learners
5. train stacking model
6. save all required artifacts
7. continue to mapping/evaluation/export

## Output policy
HX-MAP should export:
- model artifacts
- feature value mapping tables
- points/probability mapping tables
- patient-level predictions
- performance summaries
- nomogram-related figures

## Publication principle
HX-MAP is intended to preserve predictive structure while improving interpretability, reproducibility, and clinical usability.
