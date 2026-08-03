# Sequential Inference

## 1. Evolution

Each equation's posterior covariance is inflated with `delta_state` and `delta_parent`.
Cross-block covariance receives the geometric combination of both discounts. State means follow a
random walk.

## 2. Decoupled update

Each equation is updated independently under a normal-gamma prior. The code calculates forecast
error, adaptive scale, posterior state mean/covariance, and direct draws of `(theta[j], lambda[j])`.

## 3. Recoupling

Stacked independent draws form the proposal. Draw `r` receives the correction:

```text
weight[r] proportional to abs(det(I - Gamma[r]))
```

`slogdet` is used for numerical stability. Effective sample size `1 / sum(weight**2)` is recorded
at every observation. Persistently low ESS is a reason to revisit the graph, prior, or draw count.

## 4. Variational decoupling

Weighted moments are projected back to independent normal-gamma margins. The mean uses
`E[lambda * theta] / E[lambda]`; covariance, scale, and degrees of freedom are moment-matched per
equation. A bracketed root finder solves the scalar degrees-of-freedom equation.

## Stored state

Artifacts store terminal draws for forecast bands and terminal IRFs. Compact weighted state and
precision means are always stored by date for dynamic IRFs. Full draws at every date are optional
through `store_history` because they can be large.
