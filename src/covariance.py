'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard
Copyright (c) 2024 Minh Ha Ho

Licensed under GNU LGPL.3, see LICENCE file
'''




import pandas as pd
import numpy as np

from helper_functions import isPD, nearestPD



class CovarianceSpecification(dict):

    def __init__(self, *args, **kwargs):
        super(CovarianceSpecification, self).__init__(*args, **kwargs)
        self.__dict__ = self
        # Add default values
        if self.get('method') is None: self['method'] = 'pearson'
        if self.get('check_positive_definite') is None: self['check_positive_definite'] = True


class Covariance:

    def __init__(self, spec: CovarianceSpecification = None, *args, **kwargs):
        self.spec = CovarianceSpecification(*args, **kwargs) if spec is None else spec

    def set_ctrl(self, *args, **kwargs) -> None:
        self.spec = CovarianceSpecification(*args, **kwargs)
        return None

    def estimate(self, X: pd.DataFrame) -> pd.DataFrame:

        estimation_method = self.spec['method']
        if estimation_method == 'pearson':
            covmat = cov_pearson(X)
        elif estimation_method == 'duv':
            covmat = cov_duv(X)
        elif estimation_method == 'linear_shrinkage':
            lambda_covmat_regularization = self.spec.get('lambda_covmat_regularization')
            covmat = cov_linear_shrinkage(X, lambda_covmat_regularization)
        elif estimation_method == 'ledoit_wolf':
            covmat = cov_ledoit_wolf(X)
        else:
            raise NotImplementedError('This method is not implemented yet')
        if self.spec.get('check_positive_definite'):
            if not isPD(covmat):
                covmat = nearestPD(covmat)

        return covmat




# --------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------

def cov_pearson(X):
    return X.cov()

def cov_duv(X):
    return np.identity(X.shape[1])

def cov_linear_shrinkage(X, lambda_covmat_regularization = None):
    # Applies a linear shrinkage (in the form of L2 penalty in the objective function) to a given covariance matrix.
    if lambda_covmat_regularization is None or np.isnan(lambda_covmat_regularization) or lambda_covmat_regularization < 0:
        lambda_covmat_regularization = 0
    sigmat = X.cov()
    if lambda_covmat_regularization > 0:
        d = sigmat.shape[0]
        sig = np.sqrt(np.diag(sigmat.to_numpy()))
        corrMat = np.diag(1.0 / sig) @ sigmat.to_numpy() @ np.diag(1.0 / sig)
        corrs = []
        for k in range(1, d):
            corrs.extend(np.diag(corrMat, k))
        sigmat = pd.DataFrame(sigmat.to_numpy() + lambda_covmat_regularization * np.mean(sig**2) * np.eye(d), columns=sigmat.columns, index=sigmat.index)
    return sigmat

def cov_ledoit_wolf(X):
    N, T = X.shape
    Y = X - X.mean(axis=1, keepdims=True)

    sample_cov = Y @ Y.T / T
    var = np.diag(sample_cov).reshape(-1, 1)

    sqrt_var = np.sqrt(var)
    unit_cor_var = sqrt_var @ sqrt_var.T

    sample_cor = sample_cov / unit_cor_var
    avg_cor = (sample_cor.sum() - N) / (N * (N - 1))
    F = avg_cor * unit_cor_var
    np.fill_diagonal(F, var)

    gamma = np.linalg.norm(sample_cov - F) ** 2
    if gamma == 0:
        return sample_cov

    Y2 = Y * Y
    pi_mat = (Y2 @ Y2.T) / T - sample_cov ** 2
    pi = pi_mat.sum()

    Y3 = Y2 * Y
    rho_mat = ((Y3 @ Y.T) / T - var * sample_cov) * var.T / unit_cor_var
    np.fill_diagonal(rho_mat, 0)

    rho = np.diag(pi_mat).sum() + avg_cor * rho_mat.sum()

    shrink = max(0, min(1, (pi - rho) / (gamma * T)))

    return (1 - shrink) * sample_cov + shrink * F
