# bayesian-sgdlm
bayesian-sgdlm is a Python script for fully Bayesian SGDLMs, treating each node as a VAR( 𝑝) DLM. It leverages decouple–recouple filtering with Variational Bayes and importance sampling to estimate sparse, time-varying cross-lag dependencies (including pandemic dummies) without ever inverting the full multivariate system.
