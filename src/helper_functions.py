'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard
Copyright (c) 2024 Minh Ha Ho

Licensed under GNU LGPL.3, see LICENCE file
'''


############################################################################
### HELPER FUNCTIONS
############################################################################



from typing import Optional
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import numpy as np

from portfolio import Portfolio, Strategy




def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2]. The code below is written by Cyril.

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    k = 1
    while not isPD(A3):
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

def serialize_solution(name_suffix, solution, runtime):
    result = {
        'solution' : solution.x,
        'objective' : solution.obj,
        'primal_residual' :solution.primal_residual(),
        'dual_residual' : solution.dual_residual(),
        'duality_gap' : solution.duality_gap(),
        'runtime' : runtime
    }

    with open(f'{name_suffix}.pickle', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

def to_numpy(data):
    return None if data is None else data.to_numpy() if hasattr(data, 'to_numpy') else data


def output_to_strategies(output: dict) -> dict[int, Strategy]:

    N = len(output[list(output.keys())[0]])
    strategy_dict = {}
    for i in range(N):
        strategy_dict[f'q{i+1}'] = Strategy([])
        for rebdate in output.keys():
            weights = output[rebdate][f'weights_{i+1}']
            if hasattr(weights, 'to_dict'):
                weights = weights.to_dict()                 
            portfolio = Portfolio(rebdate, weights)
            strategy_dict[f'q{i+1}'].portfolios.append(portfolio)

    return strategy_dict



#------------------- Machine learning helpers -------------------

def calculate_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE)
    """
    rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred.values)) ** 2))
    return rmse


def calculate_mape(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) %
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

def show_result(predictions, y_test, y_actual, method = None):
    print(f'RMSE of linear regression: {calculate_rmse(y_test, predictions)}')
    print(f'MAPE of linear regression: {calculate_mape(y_test, predictions)}')

    plt.plot(y_actual, color = 'cyan')
    plt.plot(predictions, color = 'green')
    plt.legend(["True values", "Prediction"])
    plt.title(method)
    plt.show()




#------------------- Sampling long-only portfolios helpers -------------------

def sample_constrained_simplex(
    n: int,
    nsamples: int,
    lb: np.ndarray | list = None,
    ub: np.ndarray | list = None
) -> np.ndarray:
    """
    Sample points from canonical simplex with coordinate-wise upper and lower bounds.
    
    Args:
        n: Dimension of the simplex
        nsamples: Number of desired samples to keep
        ub: Vector of upper bounds for each coordinate
        lb: Vector of lower bounds for each coordinate (default: zeros)
    
    Returns:
        np.ndarray: Array of shape (nsamples, n) containing valid samples
    
    Raises:
        ValueError: If bounds dimensions don't match n or are inconsistent
    """
    # Convert inputs to numpy arrays and validate
    if ub is None:
        ub = np.ones(n)
    else:
        ub = np.asarray(ub)
    if lb is None:
        lb = np.zeros(n)
    else:
        lb = np.asarray(lb)
    
    # Input validation
    if len(ub) != n or len(lb) != n:
        raise ValueError(f"Bounds must have length {n}")
    if not np.all(ub > lb):
        raise ValueError("Upper bounds must be greater than lower bounds")
    if not np.all(lb >= 0):
        raise ValueError("Lower bounds must be non-negative")
    if np.sum(lb) > 1 or np.max(ub) < 1/n:
        raise ValueError("Bounds are incompatible with simplex constraints")
    
    valid_samples = []
    batch_size = max(1000, nsamples * 10)  # Adjust batch size based on target
    max_iter = 10000
    iter = 0
    success = False
    while len(valid_samples) < nsamples and iter < max_iter:
        # Generate exponential random variables
        Y = np.random.exponential(scale=1.0, size=(batch_size, n))
        
        # Normalize by row sums to get points on simplex
        row_sums = Y.sum(axis=1, keepdims=True)
        E = Y / row_sums
        
        # Filter points where coordinates are within bounds
        valid_mask = np.all(E <= ub, axis=1) & np.all(E >= lb, axis=1)
        valid_points = E[valid_mask]
        
        valid_samples.extend(valid_points)
        
        # Trim to exact size if we have enough samples
        if len(valid_samples) >= nsamples:
            valid_samples = valid_samples[:nsamples]
            success = True
            break
        iter += 1
    return np.array(valid_samples), success
