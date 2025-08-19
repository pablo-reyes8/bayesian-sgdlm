import numpy as np
from matrices import *

def get_resid_var(Y):
    """
    Estimate first-order autocorrelation and residual variance for each series in Y.

    For each column i in the data matrix Y:
      1. Align two vectors:  
         - y_reg = values from row 4 onward (target)  
         - lagged = values from row 3 up to the penultimate (predictor)
      2. Fit a simple linear regression  
         y_reg = intercept + rho * lagged + error  
      3. Extract:
         - rho_i: the slope coefficient (autocorrelation estimate)
         - sigma2_i: mean squared error of the fit (residual variance)

    Returns:
      sigma2: (q×1) array of residual variances
      rho:    (q×1) array of estimated lag-1 coefficients
    """

    y = Y[4:,:]
    T,q = y.shape
    sigma2 = np.zeros((q,1))
    rho = np.zeros((q,1))

    # loop over each series
    for i in range(0,q):
        Z = np.column_stack((np.ones(T), Y[3:-1, i])) # build design matrix [1, lagged_value]
        y_reg = Y[4:, i]

        tmpb = np.linalg.solve(Z.T @ Z, Z.T @ y_reg) # solve normal equations: beta = (Z'Z)^{-1} Z'y
        rho[i] = tmpb[1] # extract slope (rho_i)

        # compute residuals and their variance
        residual = y_reg - Z @ tmpb
        sigma2[i] = np.mean(residual**2)
    return sigma2 ,rho


def prior_AM_pp(rho,s2,lamb,covid_periods,Windx,q,p,var_i):
    """
    Construct Minnesota‐style and pandemic‐adjusted priors for one VAR/DLM node.

    This function builds the prior mean vector (m_j) and variance vector (V_j)
    for equation j by combining:
      1. **Intercept prior**: scaled by overall shrinkage λ[0] and diffuse hyperparameter λ[3].
      2. **Minnesota own‐lag prior**: mean = empirical ρ_j, variance ∝ (λ[0]/lag^λ[2])^2.
      3. **Minnesota cross‐lag prior**: variance ∝ (σ_j^2/σ_i^2)·(λ[0]·λ[1]/lag^λ[2])^2.
      4. **Pandemic‐period priors**: extra shrinkage using λ[4] over covid_periods.
      5. **Theoretical priors**: flat priors (variance=0.1) for any remaining parameters.
      6. **Custom mean adjustments**: fixed prior means for selected intercept/dummy terms
         when var_i is 1, 2, or 3.

    Returns:
      mj  : (k_j×1) vector of prior means
      Vj  : (k_j×1) vector of prior variances
      nj  : scalar degrees‐of‐freedom for inverse‐Gamma prior (fixed = 10)
      sj  : empirical variance σ_j^2 for this series
    """

    # count active “parent” regressors for series j
    pyj = int(np.sum(Windx[var_i-1,:]))
    # total number of parameters: intercept + q*p lags + pandemic dummies + active predictors
    kj = pyj + q*p + covid_periods + 1

    mj = np.zeros((int(kj),1))
    Vj = np.zeros((int(kj),1))

    # === Intercept Prior ===
    # The variance of the intercept is assigned in the first position (index 0) of Vj,
    # calculated as the square of the product (lamb[0] * lamb[3]) multiplied by s2 of the variable.
    Vj[0] = ((lamb[0]*lamb[3])**2) * s2[var_i-1]
    mj[var_i-1] = 0 # center intercept at zero
    
    # === Pandemic Prior === 
    # The loop iterates through the indices corresponding to these periods: from (q*p + 2) to (q*p + covid_periods + 1),
    # adjusting the indexing (subtracting 1)
    for i in range(q*p+2, q*p + covid_periods + 2):
        Vj[i-1] = ((lamb[0] * lamb[4])**2) * s2[var_i-1]

    # === Theoretical Priors ===
    # For the remaining parameters (beyond the pandemic periods), a fixed variance of 0.1 is assigned.
    for i in range(q*p + covid_periods + 2 ,int(kj) + 1):
        Vj[i-1] = 0.1

    # Adjustment of the means according to the variable of interest.
    if var_i== 1:
        mj[-1]= 0.75

    elif var_i==2:
        mj[-2]= 0.5
        mj[-1]= -0.5

    elif var_i==3:
        mj[-2]= 0.5
        mj[-1]= 0.75

    ## === Minnesota Prior (Litterman, 1986) ===
    # Adjusts the priors for the lag coefficients (own and cross lags).
    # The first loop iterates over each variable (i in 1..q) and the second over each lag (k in 1..p).
    # Different constraints are imposed for the "own lag" of variable 'var_i'
    # and for the lags of the other variables.
    for i in range(1, q + 1):
        for k in range(1, p + 1):
            index1 = 1 + var_i + (k - 1) * q  
            Vj[index1 - 1] = ((lamb[0] / (k ** lamb[2])) ** 2)

            if k == 1 and i == var_i:
                mj[i] = rho[var_i - 1] # set prior mean to empirical rho for first own‐lag

            if i != var_i:
                index2 = 1 + i + (k - 1) * q  
                Vj[index2 - 1] = (s2[var_i - 1] / s2[i - 1]) * (((lamb[0] * lamb[1]) / (k ** lamb[2])) ** 2) # index for cross‐lag of series i ≠ var_i
    nj=10
    sj=s2[var_i-1]

    return mj,Vj,nj,sj

def full_size(Y,p,Windx,covid_periods,indx_covid,q):
    """
    Compute total number of regression parameters across all q equations.

    This helper iterates over each series index i (0…q-1), calls `Fj_matrix_covid` to
    build the design matrix for equation i, and then counts its columns. Summing these
    counts yields the overall dimension of the global parameter vector θ.
    """
    suma= 0
    for i in range(q):
        matriz = Fj_matrix_covid(Y,p,i,Windx,covid_periods,indx_covid,q)[4,:]
        s1 = matriz.shape[0]
        suma = suma + s1 
    return suma 

# Full prior
def complete_AP(q,p,lambd,s2,rho,covid_periods,Windx,Y):
    """
    Assemble the full prior for a multivariate VAR/DLM system by concatenating
    individual priors from each of the q nodes.

    This function:
      1. Calls `full_size` to determine the total parameter dimension across all nodes.
      2. Initializes global containers for prior means (m) and variances (V).
      3. Iterates over each node i = 1…q:
         - Retrieves its local prior (mj, Vj, nj, sj) via `prior_AM_pp`.
         - Places mj and Vj into the global m and V at the correct offsets.
         - Records degrees-of-freedom (nj) and empirical variance (sj).
         - Updates a partition index array (ps) to mark block boundaries.
      4. Converts the stacked variance vector V into a diagonal matrix.

    Returns:
      m  : (p_total×1) stacked prior mean vector for all parameters.
      V  : (p_total×p_total) diagonal prior covariance matrix.
      n  : (q×1) array of degrees-of-freedom per node.
      s  : (q×1) array of empirical variances per node.
      ps : (q+1) index array marking start/end of each node’s parameter block.
    """
    indx_covid = 183 
    p_total = full_size(Y,p,Windx,covid_periods,indx_covid,q) 
    m , V = np.zeros((p_total,1)) ,np.zeros((p_total,1))
    n ,s = np.zeros((q,1)) ,np.zeros((q,1))
    ps = np.zeros(q + 1, dtype=int)  
    p_init=0  

    # loop through each node to fill in its prior block
    for i in range(1, q+1):

        # get local prior for node i
        mj, Vj, nj, sj = prior_AM_pp(rho, s2, lambd, covid_periods, Windx, q, p, i)

        pj = mj.shape[0] # number of parameters for node i
        p_t = p_init + pj # end index for this block

        # place local means & variances into global vectors
        m[p_init:p_t] = mj
        V[p_init:p_t] = Vj

        n[i - 1] = nj
        s[i - 1] = sj

        # record block boundary
        ps[i] = p_init + pj 
        p_init = p_t  # update start for next node
    
    # convert variance vector into full diagonal matrix
    return m, np.diag(V.flatten()), n, s, ps