# Simultaneous Graphical DLM 

![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/bayesian-sgdlm)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/bayesian-sgdlm)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/bayesian-sgdlm)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/bayesian-sgdlm?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/bayesian-sgdlm?style=social)

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
---

## How to Use

### Option 1 — Notebook (all-in-one)  
1. Open **`bayesian_sgdlm.ipynb`** in Jupyter Notebook or JupyterLab.  
2. Run all cells in order to reproduce the **full pipeline end-to-end**.  
   - This includes priors, dummies, matrices, decoupling/recoupling, MCMC, and forecasting.  
3. Modify the **Parameters** cell (lags, dummies, λ, φ, γ, $R$, priors...) to experiment.  

👉 This notebook is ideal for people who like to see the entire workflow executed in a single run, without having to navigate between modules.  

---

### Option 2 — Modular Code (from `src/`)  
For users who prefer **modular code organization**, the implementation is structured under the `src/` folder:  

src/
├─ decouple_recouple.py # decoupling/recoupling steps, n_solve, s_sol
├─ dummies.py # covid_dummy and related
├─ forecast.py # forecasting routines (u_forecast, osf, us)
├─ matrices.py # Fj_matrix, Wjmatrix, A_matrix, etc.
├─ mcmc.py # mcmc_forward
├─ priors.py # prior_AM_pp, full_size, complete_AP
├─ proxies.py # proxy updates (proxy_lambdj, proxy_theta_j2, etc.)

Each file contains the corresponding building blocks, so you can import only the parts you need: 

```python
from src.matrices import Fj_matrix
from src.forecast import u_forecast
```

---

## References

- West, M. & Harrison, J. (1997). *Bayesian Forecasting and Dynamic Models*. Springer.  
- Gruber, E. & West, M. (2016). “GPU‐Accelerated Bayesian Learning and Forecasting in SGDLM.” *Bayesian Analysis* 11(3): 205–225.

## Contributing

Contributions are welcome! Please open issues or submit pull requests at  
https://github.com/pablo-reyes8

## License

This project is licensed under the Apache License 2.0. 
