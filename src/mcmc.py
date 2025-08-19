
from decouple_recouple import *

def mcmc_forward(R, β, δϕ, δγ, λ0, s2, rho, pdims, Y, p, q, Windx, covid_periods, indx_covid, p_total):
    """
    Run the full forward MCMC/VB filtering for the SGDLM-VAR over all time points.

    This routine performs the three-stage decouple–recouple–VB cycle at each time step:

    1) **Initialize Priors**  
       - Call `complete_AP` to get the Minnesota‐style priors (m, C, n, s) and partition indices pdims.  
       - Compute total parameter dimension p_total via `full_size`.

    2) **Prepare Data**  
       - Drop the first p rows of Y to align lags, yielding y of shape (T, q).

    3) **Preallocate Storage**  
       - Θs: array of shape (p_total, T, R) for MCMC draws of the state vector θ_t.  
       - Λs: array of shape (q, T, R) for MCMC draws of the noise precision λ_t.

    4) **Loop over time t = 0…T–1**  
       a) **Recoupling** (`recoupling`):  
          Draw R joint samples (θs, λs) from the current priors (m, C, n, s) and data at time t.  
       b) **Store**  
          Save θs and λs into Θs and Λs for this time index.  
       c) **Compute Importance Weights**  
          Use `convert_gamma_sign_soft` to get α_r for each draw.  
       d) **Posterior λ Expectation**  
          Compute E[λ_j] via `exp_lambda`.  
       e) **VB Decoupling** (`decoupling`):  
          Update variational parameters (m, C, n, s) for the next time step.

    Returns:
      Θs : numpy array (p_total × T × R) of MCMC samples for θ over all times.
      Λs : numpy array (q × T × R) of MCMC samples for λ over all times.
    """

    # 1) Initialize priors
    m, C, n, s, pdims = complete_AP(q, p, λ0, s2, rho, covid_periods, Windx, Y)
    p_total = full_size(Y, p, Windx, covid_periods, indx_covid, q)

    # 2) Align data by dropping initial p rows
    y = Y[p:, :]
    T = y.shape[0]

    # 3) Allocate output arrays
    Θs = np.zeros((p_total, T, R))
    Λs = np.zeros((q, T, R))

    for t in range(T):

        # a) Recoupling: draw R samples of θ_t and λ_t
        θs, λs = recoupling(R, β, δϕ, δγ, m, C, n, s,pdims, t, Y, p, Windx,covid_periods, indx_covid, p_total)

         # b) Store draws
        Θs[:, t, :] = θs.T
        Λs[:, t, :] = λs.T

        # c) Compute importance weights
        alphas = convert_gamma(θs, pdims, Windx, covid_periods, q, p) 

        # d) Compute posterior expectation of λ
        Eλ     = exp_lambda(alphas, λs, q)

        # e) VB decoupling: update priors for next t
        m, C, n, s = decoupling(pdims, θs, λs, alphas, p_total, Eλ, q)

    return Θs, Λs

