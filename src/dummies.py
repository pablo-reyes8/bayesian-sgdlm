
import numpy as np

def covid_dummy(Y,p,covid_periods,indx_covid):
    """
    Generate pandemic-period dummy variables.

    This function builds a T×covid_periods matrix where T = number of rows
    in Y after dropping the first p observations. Each column i corresponds
    to one pandemic period and has a single 1 at the row that marks the start
    of that period; all other entries remain 0. These dummies can be appended
    as exogenous regressors to capture discrete shocks.

    Returns:
      A numpy array of shape (T, covid_periods) with ones indicating
      the onset of each pandemic period.
    """

    covid_dummy1= np.zeros((Y[p:,:].shape[0], covid_periods))

    # for each pandemic period, set a single 1 at the correct time index
    for i in range(0,covid_periods):
        covid_dummy1[indx_covid + i-2, i]=1 # row = indx_covid + (i - 2) aligns the dummy start with pandemic timing
    return covid_dummy1

