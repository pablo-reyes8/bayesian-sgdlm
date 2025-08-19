import numpy as np
from scipy.stats import gamma

def prxy_updating(bj , nj , Fj , yj , m1j , sj , C1j):
    """
    Perform one‐step posterior update for a univariate Bayesian DLM.

    This function takes the prior‐evolution outputs and the new observation yj,
    then computes the forecasting error, predictive variance, and updates
    the posterior state covariance and scale parameters.

    Key computations:
      1. Compute the prior gain vector Aj = C1j · Fj.
      2. Compute one‐step forecast variance hj = Fjᵀ·C1j·Fj + sj.
      3. Compute forecast error ej = yj − Fjᵀ·m_prev.
      4. Update degrees‐of‐freedom and scale: 
         n_t = bj·nj + 1, 
         zj = (bj·nj + ej²/hj) / n_t,
         s_t = sj·zj.
    Returns:
      Aj  : prior gain vector (shape p_j×1)
      hj  : one‐step predictive variance (scalar)
      ej  : one‐step forecast error (scalar)
      n_t : updated degrees‐of‐freedom (scalar)
      s_t : updated residual variance scale (scalar)
      zj  : scale adjustment factor (scalar)
    """

    # ensure yj is a scalar float
    yj = float(np.atleast_1d(yj).flatten()[0])
    Fj = np.atleast_2d(Fj) # ensure Fj is a column vector

    if Fj.shape[0] == 1 and Fj.shape[1] > 1:
        Fj = Fj.T

    # prior mean vector, reshaped if necessary to match Fj
    m_prev = np.atleast_2d(m1j)
    if m_prev.shape != Fj.shape:
        m_prev = m_prev.reshape(Fj.shape)

    Aj = C1j.dot(Fj) # Calculate the gain vector (Aj) as the product of the covariance matrix (C1j) by the predictor vector (Fj)

    hj = float((Fj.T.dot(C1j).dot(Fj))[0, 0] + sj) # Calculate the variance of the forecast (hj) using the quadratic form (Fj'·C1j·Fj) and adding the variance sj

    ej = yj - float((Fj.T.dot(m_prev))[0, 0]) # Calculate the variance of the forecast (hj) using the quadratic form (Fj'·C1j·Fj) and adding the variance sj

    n_t = bj * nj + 1 # Updates a scale parameter (nnj) by combining the factor bj and nj

    zj  = (bj * nj + ej**2 / hj) / n_t # Defines an adjustment factor (zj) that incorporates both the effect of the scale factor and the ratio of the squared error to the variance

    s_t = sj * zj # Update the variance, scaling it by the factor zj

    return Aj,hj,ej,n_t,s_t,zj


def proxy_lambdj(nnj,s1j,R):
    """
    Draw samples of the local observation‐noise scale λ_j from its posterior Gamma distribution.

    In the DLM/SGDLM framework, λ_j controls the variance of the observation noise for node j.
    After the one‐step update, its posterior is:
      λ_j ∼ Gamma(shape=nnj/2, scale=2/(nnj * s1j))
    Returns:
      samples (array of length R): draws from the posterior Gamma distribution
    """

    return gamma.rvs(a=nnj/2, scale=2/(nnj*s1j), size=R)

def proxy_theta_j2(mhat_j, Chat_j, Aj, hj, ej, n_tilde_j, s_tilde_j, zj):
    """
    Update the prior mean and covariance for node j, then compute its Cholesky factor.

    This step performs the posterior update of the state vector given the one‐step
    forecast results. It adjusts both mean and covariance, applies scaling, and
    returns the lower‐triangular Cholesky factor of the scaled covariance.

    Returns:
      m_tilde_j   : posterior mean vector
      Ch_C_bar_j  : lower‐triangular Cholesky factor of the scaled posterior covariance
    """

    # update the state mean using the gain and forecast error
    m_tilde_j = mhat_j + (Aj / hj) * ej

    #ensure Aj is a column vector for matrix operations
    if Aj.ndim == 1:
        Aj = Aj.reshape(-1, 1)
    
    # update the covariance: subtract outer‐product term then scale by zj
    C_tilde_j = (Chat_j - (Aj @ Aj.T) / hj) * zj

    # apply final scaling by the updated variance scale
    C_bar_j = C_tilde_j / s_tilde_j

    # compute Cholesky factor of the posterior covariance
    Ch_C_bar_j = np.linalg.cholesky(C_bar_j)

    return m_tilde_j, Ch_C_bar_j




