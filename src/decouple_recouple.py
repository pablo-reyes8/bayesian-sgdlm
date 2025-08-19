import numpy as np
from matrices import * 
from proxies import * 
from priors import *
from scipy.special import digamma, polygamma
from scipy.optimize import newton



def pred_θj(mj, Cj, phi, gamma, Windxj, covid_periods, q, p):
    """
    Perform the prior‐evolution step for node j in the SGDLM.

    This function takes the current prior mean vector mj and covariance matrix Cj,
    applies discount‐factor adjustments to the covariance (via Wjmatrix),
    and returns an updated prior for the next time step.

    Key steps:
      1. Keep the prior mean mj unchanged (no drift in the state mean).
      2. Compute the covariance adjustment Wj using φ (own‐parameter discount),
         γ (simultaneous‐parameter discount), parent mask Windxj, and model dimensions.
      3. Add Wj to Cj to inflate the prior covariance, reflecting parameter evolution.

    Returns:
      mj1 : same as mj (prior mean carries over)
      Cj1 : Cj + Wj (inflated prior covariance for time t+1)
    """

    # 1) carry forward the prior mean without change
    mj1 = mj

    # 2) build the covariance adjustment matrix Wj
    Wj = Wjmatrix(phi, gamma, Windxj, Cj, covid_periods, q, p)

    # 3) inflate the prior covariance by adding Wj
    Cj1 = Cj + Wj
    return mj1, Cj1



def recoupling(R, beta, delta_phi, delta_gamma,m, C, n, s, pdims, t, Y, p, Windx, covid_periods, indx_covid,p_total , q):
    """
    Perform the SGDLM recoupling step via Monte Carlo simulation.

    This function takes the decoupled, univariate updates for each series j
    and stitches them together into a coherent multivariate posterior sample
    of states θ and observation‐noise precisions λ. Follows Gruber & West (2017).

    Workflow:
      1. Initialize storage for R draws of each series’ precision λ_j and
         the full state vector θ (length p_total).
      2. Extract the current observation y[t] for each series, after dropping p lags.
      3. Loop over each series j:
         a. **Extract** the j‐th block of the prior mean m and covariance C
            using partition indices pdims.
         b. **Apply discounting** via pred_θj: update m_hat_j, C_hat_j
            with factors φ and γ for recoupling.
         c. **Build** the design vector Fj at time t (includes lags & COVID dummies).
         d. **One‐step update** with prxy_updating: compute gain Aj, forecast
            variance hj, error ej, and updated scale n_tilde_j, s_tilde_j, zj.
         e. **Sample** R draws of λ_j from its posterior via proxy_lambdj.
         f. **Posterior factorization**: update the posterior mean m_tilde_j and
            compute Cholesky factor Ch_C_tilde_j with proxy_theta_j2.
         g. **Generate** R state‐vector samples: for each draw r, scale the
            Cholesky factor by 1/√λ_j and add to m_tilde_j.

    Returns:
      thetas: array (R × p_total) of sampled state vectors
      lambdas: array (R × q) of sampled precisions for each series
    """

    lambdas = np.zeros((R, q))
    thetas  = np.zeros((R, p_total))

    # reduced data matrix (drop first p rows for lag alignment)
    y = Y[p:, :]  

    for j in range(q):
        # 1) Extract prior block for series j
        start, end = pdims[j], pdims[j+1]
        m_hat_j = m[start:end].copy()             # prior mean for node j
        C_hat_j = C[start:end, start:end].copy()  # prior cov block for node j

        # 2) Apply discount‐factor evolution (phi, gamma)
        m_hat_j, C_hat_j = pred_θj(m_hat_j, C_hat_j,delta_phi[j], delta_gamma[j],Windx[j, :], covid_periods, q, p)

        # 3) Build the design vector Fj at time t
        Fj = Fj_matrix_covid(Y, p, j+1,Windx, covid_periods,indx_covid, q)[t, :]
        Fj = Fj.reshape(-1, 1)

        # 4) One‐step univariate update: gain, forecast var, error, scale updates
        Aj, hj, ej, n_tilde_j, s_tilde_j, zj = prxy_updating(beta[j], n[j], Fj, y[t, j],m_hat_j, s[j], C_hat_j)

        # 5) Sample R draws of lambda_j
        lambdas[:, j] = proxy_lambdj(n_tilde_j, s_tilde_j, R)

        # 6) Posterior mean & Cholesky of cov for node j
        m_tilde_j, Ch_C_tilde_j = proxy_theta_j2(m_hat_j, C_hat_j,Aj, hj, ej, n_tilde_j, s_tilde_j, zj)

        pj_block = end - start # number of parameters in this block

        # 7) Generate R joint samples for θ in this block
        for r in range(R):
            inv_lam = 1.0 / lambdas[r, j]
            M = np.sqrt(inv_lam) * Ch_C_tilde_j # scale Cholesky factor by sqrt(1/λ_j)
            thetas[r, start:end] = (m_tilde_j.flatten()+ M @ np.random.randn(pj_block)) # draw standard normal vector and shift by m_tilde_j

    return thetas, lambdas


def n_solve(pdims, d, alphas, lambdas, j, E_lambda):
    """
    Solve for the posterior degrees‐of‐freedom n_{j,t} in the VB update.

    We find n > 0 satisfying the West & Harrison equation for node j:
        log(n + p_j – d) – ψ(n/2) – log(2 · E[λ_j]) + E[log λ_j] – (p_j – d)/n = 0

    using the substitution x = log(n) and Newton–Raphson.
    Returns:
      n_jt      : positive scalar solution for the degrees‐of‐freedom at time t
    """

    # block dimension for node j
    pj = pdims[j+1] - pdims[j]

    # compute E[log λ_j] under importance weights
    log_lambda_j = np.log(lambdas[:, j])
    Elog_lambda_j = np.sum(alphas * log_lambda_j)

    # define target function in x = log(n)
    def f(x):
        n = np.exp(x)
        return ( np.log(n + pj - d)- digamma(n/2)- np.log(2 * E_lambda[j])+ Elog_lambda_j- (pj - d) / n )

    # derivative f'(x) for Newton’s method
    def fprime(x):
        n = np.exp(x)
        return n * ( 1/(n + pj - d)- 0.5 * polygamma(1, n/2)+ (pj - d) / (n**2) )

    # initial guess x0 = log(5)
    x0 = np.log(5.0)
    x_star = newton(f, x0, fprime, tol=1e-8, maxiter=50)
    return np.exp(x_star)

def s_sol(pdims,dj,Eλ,nj,j):
    """
    Compute the posterior scale parameter s_j for series j in VB.

    Implements the West & Harrison VB formula for the Gamma posterior scale:
        s_j = (n_j + p_j – d_j) / (n_j · E[λ_j])

    where:
      - n_j      : updated degrees of freedom for series j
      - p_j      : number of parameters in block j = pdims[j+1] – pdims[j]
      - d_j      : weighted Mahalanobis trace for block j
      - E[λ_j]   : expected precision for series j

    The outcome s_j controls the residual variance in the univariate model after recoupling and decoupling using VB.
    Returns:
      s_j   : posterior Gamma scale parameter controlling residual variance
    """

    pj = pdims[j+1] - pdims[j] # dimensión del bloque de parámetros para la serie j
    return (nj+pj-dj) / (nj * Eλ[j]) # aplicación directa de la fórmula VB para s_j


def decoupling(pdims, theta_s, lambda_s, alphas, p_total, E_lambda, q):
    """
    Perform the VB decoupling step after recoupling.
    Given:
      - theta_s   : (R × p_total) Monte Carlo draws of the full state vector θ.
      - lambda_s  : (R × q) draws of observation‐noise precisions λ_{r,j}.
      - alphas    : (R,) normalized importance weights α_r.
      - E_lambda  : (q × 1) posterior expectations E[λ_j].
      - pdims     : partition indices delimiting each series’ parameter block in θ.
      - q         : number of series.
      - p_total   : total number of parameters across all series.

    Computes VB priors for the next period:
      - m : (p_total × 1) mean vector, where for each block j:
          m_j = \frac{\sum_{r=1}^R \alpha_r\,\lambda_{r,j}\,\theta_{r,j}}{E[\lambda_j]}

      - C : (p_total × p_total) block-diagonal covariance, with each block
          V_j = \sum_{r=1}^R \alpha_r\,\lambda_{r,j}\,(\theta_{r,j}-m_j)(\theta_{r,j}-m_j)^\top,\quad
          C_j = s_j\,V_j

      - n : (q × 1) updated degrees-of-freedom for each series, solving
          \log(n_j + p_j - d_j) - \psi\!\bigl(\tfrac{n_j}{2}\bigr) - \log\bigl(2\,E[\lambda_j]\bigr) + E[\log \lambda_j]
          - \frac{p_j - d_j}{n_j} = 0

      - s : (q × 1) scale parameters
          s_j = \frac{n_j + p_j - d_j}{n_j\,E[\lambda_j]}.
    Returns:
      m, C, n, s
    """
    C = np.zeros((p_total, p_total))
    n = np.zeros((q, 1))
    s = np.zeros((q, 1))

    # 1) Compute VB prior mean vector m_j = E[θ_j] / E[λ_j]
    m = m_vector(pdims, theta_s, lambda_s, alphas, p_total, q, E_lambda)

    # 2) For each series j, build covariance block and update n_j, s_j
    for j in range(q):
        start, end = pdims[j], pdims[j+1]
        Vj = V_matrix(pdims, alphas, theta_s, lambda_s, m, j)  # 2.1) VB covariance block V_j
        dj = d_matrix(pdims, alphas, theta_s, lambda_s, m, Vj, j) # 2.2) Mahalanobis term d_j for df update
        nj_val = n_solve(pdims, dj, alphas, lambda_s, j, E_lambda) # 2.3) Solve for VB degrees of freedom n_j
        n[j, 0] = nj_val

        sj_val = s_sol(pdims, dj, E_lambda, nj_val, j) # 2.4) Compute scale s_j
        s[j, 0] = sj_val 

        # 2.5) Fill block in global covariance matrix
        C[start:end, start:end] = Vj * sj_val

    #m: the column vector of approximate posterior means, concatenated for each series j
    #C: the approximate posterior covariance matrix, constructed block by block as Cj = Vj * sj
    #n: the vector with the degrees of freedom for each series
    #s: the vector of scales sj

    return m, C, n, s


