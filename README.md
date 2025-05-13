# Simultaneous Graphical DLM 

This script implements the Simultaneous Graphical DLM (SGDLM) of West & Harrison (1997) and Gruber & West (2016), extended so each DLM model stacks its own $p$ lags **and** the $p\$ cross-lags of all other series. This notebook walks through every step, from data prep to reconstructing the final dynamic VAR coefficients, using a decouple–recouple Variational Bayes + importance-sampling algorithm.


---

## Notebook Outline

1. **Data & Configuration**  
   - Load your multivariate time series $Y$ and specify the contemporaneous graph mask.  
   - Set lag order $p$, pandemic dummy horizon, Minnesota‐prior hyperparameters, and Monte Carlo draws $R$.

2. **Design & Prior Setup**  
   - Construct lagged and dummy‐augmented regressor matrices for each series.  
   - Compute AR(1) empirical moments and assemble Minnesota‐style priors for all coefficient blocks.

3. **Recoupling & VB-IS Refinement**  
   - Fuse marginal DLM outputs via sparse Monte Carlo draws under the graph mask.  
   - Compute VB moment updates (covariance, Mahalanobis traces, degrees of freedom, scales) using importance weights.  
   - Iterate coordinate VB and importance‐sampling corrections as needed.

4. **Decoupling: Parallel DLM Updates**  
   - Run independent univariate DLM filters for each series—incorporating own‐ and cross‐lags plus exogenous dummies.  
   - Obtain one‐step forecast gains, updated state means/covariances, and sample noise precisions.

5. **Results & Diagnostics**  
   - Reconstruct time‐varying VAR coefficient matrices and error covariance.  
   - Plot trace/histograms of precision chains, forecast densities.  
   - Unconditional k-step out-of-sample forecasts. 

---

## Dependencies

```bash
pip install pandas numpy scipy matplotlib 
```

## How to Use

1. Open `bayesian_sgdlm.ipynb` in Jupyter Notebook or JupyterLab.  
2. Run all cells in order to reproduce the full pipeline.  
3. Modify the **Parameters** cell (lags, dummies, λ, φ, γ, $R$, priors...) to experiment.  

---

## References

- West, M. & Harrison, J. (1997). *Bayesian Forecasting and Dynamic Models*. Springer.  
- Gruber, E. & West, M. (2016). “GPU‐Accelerated Bayesian Learning and Forecasting in SGDLM.” *Bayesian Analysis* 11(3): 205–225.  
