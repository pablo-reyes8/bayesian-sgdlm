import numpy as np
from decouple_recouple import *

def idx_col_major(mask):
    """
    Convert a boolean mask into column-major (Fortran-style) index order.

    This helper takes a 2D boolean array `mask` and returns the row and column
    indices of all True entries in the order you would iterate over columns
    first (column-major order). 
    Returns:
      (rows, cols) : two 1D arrays of equal length, giving the positions of True
                     entries in column-first order.
    """
    rows, cols = np.nonzero(mask)
    order = np.lexsort((rows, cols))  
    return rows[order], cols[order]

def A_matrix(θs, pdims, Windx, covid_periods, q, p, r):
    """
    Build the structural impact matrix A = (I - Gamma')^{-1} for a VAR model.

    Args:
      theta_s       : array (R × p_total) of MCMC draws.
      pdims         : partition indices for each equation's parameter block.
      Windx         : (q×q) boolean mask for contemporaneous parents.
      covid_periods : number of COVID dummy parameters per equation.
      q             : number of series (equations).
      p             : VAR lag order.
      r             : 1-based draw index into theta_s.

    Returns:
      A : (q × q) structural impact matrix = (I - Gamma')^{-1}.
    """

    pphi = q*p + 1 + covid_periods
    Γ = np.eye(q)

    # Collect all contemporaneous gamma coefficients into one vector
    gammas = []
    W_tr=Windx.T
    
    # vector con TODOS los γ
    for j in range(q):
        start, stop = pdims[j], pdims[j+1]
        θj = θs[r-1, start:stop]
        gammas.extend( θj[pphi:] )
    gammas = np.asarray(gammas)

    # Determine positions for off-diagonal entries
    rows, cols = idx_col_major(W_tr)    
    if gammas.size != rows.size:
        raise ValueError("Windx y γ no concuerdan")

    Γ[rows, cols] = -gammas

    # Return the structural impact matrix
    return np.linalg.inv(Γ.T)


def us(θs,Y,pdims,Windx,covid_periods,q,p,r):
    """
    Construct the prediction vector us by accumulating, for each equation i,
    the dot product between the vector of regressors from the last observation
    (constant, p lags for all series and zeros for COVID periods)
    and the corresponding subset of coefficients θs.

    For each i from 0 to q-1:
    1. Extract the last p Y values ​​in reverse order and transpose them,
    flattening them to form the lag regressors.
    2. Prepare fj: a 1 for the intercept, those lags, and zeros to reserve space of length covid_periods.
    3. Count how many contemporaneous parameters are left out (pyj)
    according to the Windx mask in row i.
    4. Select from θs[r] the segment of coefficients φ_j that corresponds
    to equation i, excluding pyj parameters at the end.
    5. Compute us[i] as the dot product between fj and φ_j.

    The result is a column vector us of size q with the predicted
    line
    """
    us = np.zeros((q,1))

    for i in range(q):
        block = Y[-1 : -p-1 : -1, :]
          
        flat_block = block.T.reshape(-1, order='F')
        fj = np.concatenate(([1] ,  flat_block , np.zeros(covid_periods))) # Build regressor f_j: intercept + lags + placeholders for COVID dummies

        pyj = int(np.sum(Windx[i,:])) # Exclude contemporaneous coefficients as indicated by Windx[i]

        phij = θs[r-1, pdims[i]: pdims[i+1]-pyj] # Extract matching slice of phi_j from the state vector for series i

        us[i] = fj.T @ phij  # Compute the forecast for series i
    return us

def osf(mu_s, A_s):
    """
    Compute the structural out-of-sample forecast for all series.

    This function applies the structural impact matrix A_s to the vector of
    one-step ahead predictions mu_s to produce the final forecast.

    Returns:
      o_s : array of shape (q, 1)
        Final out-of-sample forecast combining structural impacts: A_s @ mu_s.
    """
    # Apply structural impact matrix to prediction vector
    o_s = A_s @ mu_s
    return o_s

def cova_f(As, λs, r):
    """
    Compute the structural predictive covariance matrix Σ for draw r.

    This uses the structural impact matrix A_s and the noise precisions λs[r]:
        Σ = A_s · diag(1/λs[r]) · A_sᵀ
    and enforces exact symmetry via eigen‐decomposition.
    Returns:
      Σ   : array (q × q)
        Symmetric predictive covariance matrix for the next observation,
        incorporating contemporaneous and lagged effects.
    """

    # 1) Extract the precision vector for draw r
    inv_lambda = 1.0 / λs[r-1, :]
    D = np.diag(inv_lambda)

    # 2) Form the raw covariance M = A_s · D · A_sᵀ (may be slightly non‐symmetric)
    M = As @ D @ As.T

    # 3) Enforce exact symmetry via eigen‐decomposition
    #    M = V · diag(w) · Vᵀ  → Σ = V · diag(w) · V
    w, v = np.linalg.eigh(M)
    Σ = v @ np.diag(w) @ v.T 
    return Σ

def u_forecast(H, Y, θs, λs, Windx, covid_periods, q, p,
               β, δϕ, δγ, r, pdims, indx_covid, p_total):
    """
    Generate H-step ahead out-of-sample forecasts via decoupling–recoupling.

    At each horizon step i = 0…H-1:
      1. Build structural impact matrix A_s from draw r of θs.
      2. Compute one-step ahead predictions μ_s using the most recent Y and θs[r].
      3. Apply contemporaneous impacts: ỹ = A_s @ μ_s (osf).
      4. Form the predictive covariance Σ = A_s · diag(1/λs[r]) · A_sᵀ (cova_f).
      5. Draw a shock ε ∼ N(0, Σ) via Cholesky, then forecast y_f = ỹ + ε.
      6. Append y_f to Y for the next iteration’s lag alignment.
      7. Recompute importance weights α = convert_gamma(θs…) and E[λ] = exp_lambda(…).
      8. VB decoupling → new priors (m, C, n, s) for the next out-of-sample step.
      9. Recoupling using these priors to redraw θs, λs for the new Y.

    Returns:
      y_pred : array (q × H), simulated out-of-sample forecasts for each series at each horizon step.

    """
    y_pred = np.zeros((q, H))
    
    for i in range(H):
        # 1–3) Structural one-step forecast
        Asf = A_matrix(θs , pdims, Windx, covid_periods, q, p, r) # Structural matrix A 
        mu_f = us(θs, Y, pdims, Windx, covid_periods, q, p, r) # Cumulative predictions 
        myf = osf(mu_f, Asf).flatten() 
        
        # 4) Predictive covariance
        Σ = cova_f(Asf, λs, r)
        L = np.linalg.cholesky(Σ)
        
        # 5) Simulation: add Gaussian shock
        yf = myf + L @ np.random.randn(q)
        y_pred[:, i] = yf
        
        # 6) Update Y with the new forecast
        Y = np.vstack((Y,   yf.reshape(1, -1)))
        y = Y[p:,:]
        T = y.shape[0] - 1
        
        # 7) Compute IS weights and VB moments
        alphas = convert_gamma(θs, pdims, Windx, covid_periods, q, p) 
        Eλ = exp_lambda(alphas, λs, q)
        m, C, n, s = decoupling(pdims, θs, λs, alphas, p_total, Eλ, q)

        # 8) Recouple to redraw θs, λs for the expanded Y
        θs, λs = recoupling(1500, β, δϕ, δγ,m, C, n, s,pdims, T, Y, p,Windx, covid_periods,indx_covid, p_total)

    return y_pred

