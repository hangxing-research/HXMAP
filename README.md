# HX-MAP

**HX-MAP** stands for **Harmonized X-response Marginal Additive Prediction**.

This repository provides a modular implementation of the HX-MAP framework, which translates ensemble machine learning models into transparent, point-based clinical decision tools.

## Overview

HX-MAP is designed to improve interpretability of ensemble prediction models by transforming feature-level nonlinear marginal effects into an additive point-based system.

The framework follows three core steps:

1. **Quantification of marginal X-response**  
   For a given feature and value, HX-MAP estimates how the model prediction changes relative to a reference value across the empirical background distribution.

2. **Harmonized ensemble contribution**  
   Feature-specific marginal responses are combined across base learners using:
   - feature importance within each base learner
   - corresponding stacking meta-model coefficients

3. **Additive point scale and probability mapping**  
   The harmonized contribution functions are transformed into point scales, summed into total points, and mapped back to predicted outcome probabilities.

---

## Repository purpose

This public repository is intended to share:

- the HX-MAP code framework
- the modular pipeline structure
- the training / mapping / evaluation / plotting workflow
- the implementation logic for interpretable ensemble prediction

This repository does **not** include public data files.
