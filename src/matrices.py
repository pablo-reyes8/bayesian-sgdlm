import numpy as np
from dummies import *
from math import sqrt as sqrt


def Fj_matrix(Y,p,j,Windx):
    """
    Build the design matrix F for series j with p lags and selected predictors.

    This function constructs a matrix of regressors by:
    1. Dropping the first p rows of Y to align lags with current observations.
    2. Creating columns for each lag 1…p of all q series in Y.
    3. Prepending a column of ones for the intercept.
    4. Appending only the “active” predictors for series j based on the boolean mask Windx[j-1].

    Internally, it:
    - Slices Y to get y = Y[p:, :] so that y[t] corresponds to Y[t+p].
    - Initializes F (without intercept) as all ones and then fills each block of q columns
      with the values of Y lagged by i periods.
    - Stacks the intercept, full lag block, and selected columns into one matrix.

    """
    y = Y[p:,:]
    T,q = y.shape
    k=q*p
    F = np.ones((T,k))

    # fill each lag-block: for lag i, copy Y shifted by i
    for i in range(0,p+1):
        if 1+(i-1)*q < 0:
            pass 
        else:

            F[:, ((i-1)*q):(i*q)] = Y[p - i : Y.shape[0] - i, :] # assign lagged values: Y[t+p-i] → F[t, start_col:end_col]

    ones_col = np.ones((T, 1))

    # combine intercept, all lags, and selected predictors
    return np.hstack([ones_col, F, y[:, Windx[j-1, :].astype(bool)]])

def Fj_matrix_covid(Y,p,j,Windx,covid_periods,indx_covid,q):
    """
    Extend the lagged‐regressors matrix by including pandemic‐period dummies.

    This function builds on Fj_matrix by:
      1. Dropping the first p rows of Y to align past values.
      2. Creating the full block of p lags for all q series.
      3. Generating covid dummy columns to capture discrete pandemic shocks.
      4. Appending only the “active” predictors for series j based on Windx[j-1].

    The final matrix has columns:
      [intercept | all lags | covid dummies | selected predictors]
    """

    y = Y[p:,:]
    T,q = y.shape
    k=q*p

    cd19 = covid_dummy(Y,p,covid_periods,indx_covid) # generate pandemic‐period dummy matrix (T × covid_periods)
    F = np.ones((T,k))

    # fill lagged block: for each lag i, shift Y by i periods
    for i in range(0,p+1):
        if 1+(i-1)*q < 0:
            pass 
        else:
            F[:, ((i-1)*q):(i*q)] = Y[p - i : Y.shape[0] - i, :] # assign Y[t+p-i, :] into F[t, start:end]
    ones_col = np.ones((T, 1)) 

    # combine intercept, lags, covid dummies, and active predictors
    return np.hstack([ones_col, F,cd19, y[:, Windx[j-1, :].astype(bool)]]) 


def Wjmatrix(phi , gamma ,Windxj,Cj,covid_periods,q,p): 
    """
    Build the discount‐adjustment matrix W for a single node in the SGDLM.

    This matrix W encodes how much to “inflate” the prior covariance blocks
    during recoupling, using:
      - φ (phi): discount factor for the node’s own parameters
        (intercept, lags, pandemic dummies)
      - γ (gamma): discount factor for the node’s simultaneous‐interaction
        coefficients (parents)

    W is partitioned into four blocks relative to Cj:
      1. Own‐parameter block (0:kj, 0:kj)
      2. Simultaneous‐parameter block (kj:, kj:)
      3. Cross‐blocks (off‐diagonals) between own and simultaneous parts

    If Windxj (the parent‐mask) sums to ≥1, both φ and γ adjustments apply;
    otherwise only φ is used for own‐parameters.
    Returns:
      W : (pj × pj) adjustment matrix to add to prior precision during recoupling
    """

    pgammaj = int(np.sum(Windxj))        # total number of simultaneous coefficients (parents)
    pj = Cj.shape[1]                     # total parameters in this node
    W = np.zeros((pj, pj))              
    kj = 1 + q * p + covid_periods       # compute split index for own‐parameter block
    Cphij = Cj[0:kj, 0:kj]               # extract own‐parameter covariance block

    if pgammaj >= 1:
        Cgammaj = Cj[kj:, kj:]            # extract simultaneous‐parameters block

        W[0:kj, 0:kj] = ((1/phi) - 1) * Cphij   # 1) Inflate own‐parameter block by factor (1/φ - 1)
        W[kj:, kj:] = ((1/gamma) - 1) * Cgammaj # 2) Inflate simultaneous block by factor (1/γ - 1)

        # 3) Inflate off‐diagonals between own and simultaneous by √(φ·γ)
        #    These ensure coherent discounting across blocks
        W[kj:, 0:kj] = ((1/sqrt(gamma * phi)) - 1) * Cj[kj:, 0:kj]
        W[0:kj, kj:] = ((1/sqrt(gamma * phi)) - 1) * Cj[0:kj, kj:]

    else:
        # if no parents, only adjust own‐parameters
        W[0:kj, 0:kj] = ((1/phi) - 1) * Cphij
    return W


def m_vector(pdims, θs, λs, alphas, p_total, q, Eλ):
    """
    Compute the VB prior mean vector m after recoupling and importance‐weighting.

    For each series j = 1…q:
      1. Extract the block θs[:, start:end] of shape (R × p_j), containing R draws of that series’ coefficients.
      2. Weight each draw by its precision λs[r, j].
      3. Compute the weighted sum across draws with importance weights α_r:
         numer_j = Σ_{r=1}^R α_r · [λ_{r,j} · θs[r, start:end]]
      4. Normalize by E[λ_j] to obtain
         m_j = numer_j / Eλ[j]

    This implements the VB update m_j = E[λ_j θ_j] / E[λ_j] (see Gruber & West).
    Returns:
      m : (p_total × 1) VB prior mean vector for the next iteration.
    """

    R = θs.shape[0]
    m = np.zeros((p_total, 1))


    for j in range(q):
        start, end = pdims[j], pdims[j+1]
        # R × p_j block of draws for series j
        block = θs[:, start:end]               
        # weight each draw by its λs value
        weighted = block * λs[:, j].reshape(R, 1)

        # importance‐weighted sum: (1 × R) @ (R × p_j) = (1 × p_j)
        numer = alphas.reshape(1, R) @ weighted  # (1 × pj)
        # normalize by E[λ_j]
        m[start:end, 0] = (numer / Eλ[j]).flatten() 
    return m


def V_matrix(pdims,alphas,θs,λs,m,j):
    """
    Compute the VB‐approximated prior covariance block V_j for node j after recoupling.

    Implements the expectation
        V_j = E[ λ_j · (θ_j − m_j)(θ_j − m_j)ᵀ ]
    using R Monte Carlo samples and normalized importance weights α_r.

    - θs[r, pdims[j]:pdims[j+1]] : the r-th draw of the full state vector for node j
    - m[pdims[j]:pdims[j+1]]     : the VB mean vector m_j for node j
    - λs[r, j]                   : the r-th draw of the precision λ_j
    - alphas[r]                  : the r-th normalized importance weight α_r

    The block dimension is pj = pdims[j+1] − pdims[j].

    Returns:
      V : (pj × pj) covariance matrix for node j
    """
    
    R = len(alphas)
    pj = -(pdims[j] - pdims[j+1])
    V = np.zeros((pj,pj))

    mm = m[pdims[j]:pdims[j+1]] # extract VB mean for block j
    err = θs[:,pdims[j]:pdims[j+1]] - mm.T # compute deviations for each sample: shape (R, pj)
    G = np.empty_like(V)  # temporary buffer for outer product
    
    for i in range(R):
        err_r = err[i,:]
        np.multiply.outer(err_r, err_r, out=G) # compute (θ_r − m)(θ_r − m)ᵀ in-place
        V = V + alphas[i] * (λs[i,j] * G) # weight by α_r and λ_{r,j}
    return V 


def d_matrix(pdims,alphas,θs,λs,m,V,j):
    """
    Compute the VB Mahalanobis‐distance term D_j for node j after recoupling.

    This implements the expectation needed to update the degrees of freedom in VB:
        D_j = E[λ_j · (θ_j − m_j)ᵀ · V_j⁻¹ · (θ_j − m_j)]
            = Σ_{r=1}^R α_r · λ_{r,j} · (θ_{r,j} − m_j)ᵀ · V_j⁻¹ · (θ_{r,j} − m_j)
    Returns:
      V : (p_j × p_j) VB prior covariance matrix for series j.
    """

    R = len(alphas)
    D=0
    # Invert the covariance block
    iV = np.linalg.inv(V)

    # Extract VB mean for node j
    mm = m[pdims[j]:pdims[j+1]]

    # Deviations of each draw: θ_{r,j} − m_j
    err = θs[:,pdims[j]:pdims[j+1]] - mm.T

    # Accumulate α_r · λ_{r,j} · err_rᵀ iV err_r
    for i in range(R):
        err_r = err[i,:]
        D = D + alphas[i] * λs[i,j] * (err_r.T @ iV @ err_r) # (err_r.T @ iV @ err_r) is precisely the Mahalanobis distance of the vector err with the covariance matrix
    return D 


def convert_gamma(theta_s, pdims, Windx, covid_periods, q, p):
    """
    Compute importance‐sampling weights α based on simultaneous‐interaction coefficients γ.

    For each Monte Carlo draw r of the full state vector θ_s[r]:
      1. Extract all simultaneous‐interaction coefficients γ_j for each series j:
         - Each block θ_s[r, pdims[j]:pdims[j+1]] contains [own‐lags; pandemic dummies; γ_j].
         - γ_j are the entries from index p_phi to end of that block.
      2. Stack γ_j from j=1…q into one long vector gamma_vec.
      3. Form the q×q simultaneous‐dependence matrix Γ_t = I_q with off‐diagonals:
         Γ_t[i,j] = –γ_vec[k] wherever Windx[i,j] is True (i≠j).
      4. Compute α_r = |det(Γ_t)|, the absolute determinant correction (see eqs. 11–12 of Gruber & West).
      5. Normalize all α_r so they sum to 1.
    Returns:
      alphas_normed : length-R array of normalized importance weights α_r.
    """
     
    R = theta_s.shape[0]
    p_phi = 1 + q*p + covid_periods # length of each block before simultaneous coefficients
    mask = Windx.astype(bool).T  # boolean mask for off‐diagonal entries in Γ_t
    alphas = np.zeros(R)
    Iq = np.eye(q)

    for r in range(R):
        # 1) collect all γ_j from each series block
        gamma_vec = []
        for j in range(q):
            start, stop = pdims[j], pdims[j+1]
            block = theta_s[r, start:stop]
            gamma_j = block[p_phi:]           # simultaneous coeffs for series j
            gamma_vec.append(gamma_j)
        gamma_vec = np.hstack(gamma_vec)   

        # 2) build Γ_t = I_q plus off‐diagonals = –γ entries
        Γt = Iq.copy()
        Γt[mask] = -gamma_vec

        # 3) compute weight = |determinant of Γ_t|
        alphas[r] = abs(np.linalg.det(Γt))

    # 4) normalize weights for importance sampling
    return alphas / alphas.sum()



def exp_lambda(alphas,λs,q):
    """
    Compute the posterior expectation of each observation‐noise precision λ_j
    after recoupling, using importance weights α_r.

    The expectation for series j is:
        E[λ_j] = Σ_{r=1}^R α_r * λ_{r,j}

    Args:
      alphas : array of length R containing normalized importance weights α_r
      λs     : array (R × q) of sampled precisions λ_{r,j}
      q      : number of series

    Returns:
      Elambda: (q × 1) array where each entry Elambda[j] is the weighted average
               of the R draws for λ_j.
    """

    Elambda = np.zeros((q,1))

    # for each series j, compute weighted sum of its R draws
    for i in range(q):
        Elambda[i]= np.sum(alphas* λs[:,i]) # E[λ_j] = Σ α_r λ_{r,j}
    return Elambda


