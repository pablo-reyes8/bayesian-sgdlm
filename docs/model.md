# Model Specification

## Series-specific form

For equation `j`, the observation equation is:

```text
y[j,t] = x[j,t]' phi[j,t] + y[parents(j),t]' gamma[j,t] + error[j,t]
error[j,t] ~ Normal(0, 1 / lambda[j,t])
```

The state follows a random walk. Separate discount factors control the covariance evolution of
`phi` and `gamma`. The non-simultaneous state order is:

```text
[intercept,
 lag-1 of series 1..q,
 ...,
 lag-p of series 1..q,
 exogenous variables]
```

Contemporaneous coefficients follow that block. `parents[j, i] == True` means series `i` is a
parent in equation `j`; self-parent entries are removed.

## Multivariate form

With `Gamma[j, i] = gamma[j, i]` and a zero diagonal:

```text
(I - Gamma[t]) y[t] = mu[t] + error[t]
A[t] = inverse(I - Gamma[t])
Cov(y[t]) = A[t] inverse(Lambda[t]) A[t]'
```

The graph is fixed for one fit. Its coefficients, lag coefficients, exogenous effects, and
precisions change sequentially.

## Priors

`core.initial_state` estimates an AR(1) anchor and residual variance independently for each
series. The first own-lag prior mean uses that anchor. Lag variances use Minnesota-style overall
shrinkage, cross-series shrinkage, lag decay, and residual-variance scaling. No coefficient is
customized for the original dataset.
