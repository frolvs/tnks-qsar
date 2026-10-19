#!/usr/bin/env python3

import numpy as np
from sklearn.model_selection import LeaveOneOut

def loo_cv(X, y, model_factory=LinearRegression):
    """LeaveOneOut CV. Returns q^2, rmse and preds."""
    X = np.asarray(X); y = np.asarray(y)
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        model = model_factory()        
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    q2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(ss_res / len(y))
    return q2, rmse, preds
    
def repeated_stratified_cv(X, y, groups, n_splits=5, n_repeats=20, model_factory=LinearRegression):
    """Repeated Stratified K-Fold CV. Returns q^2 and rmse mean ± std."""
    X = np.asarray(X); y = np.asarray(y)
    q2_list = []
    rmse_list = []
    
    for seed in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        preds = np.zeros(len(y))
    
        for tr, te in skf.split(X, groups):
            model = model_factory().fit(X[tr], y[tr])
            preds[te] = model.predict(X[te])
            
        q2 = 1 - np.sum((y - preds)**2) / np.sum((y - y.mean())**2)
        rmse = np.sqrt(np.mean((y - preds)**2))
        
        q2_list.append(q2)
        rmse_list.append(rmse)
        
    return np.mean(q2_list), np.std(q2_list), np.mean(rmse_list), np.std(rmse_list)
    
def y_randomization(X, y, n_iter=2000, random_state=42, model_factory=LinearRegression):
    """Randomized by y. Returns q^2 array."""
    rng = np.random.default_rng(random_state)
    q2_arr = np.zeros(n_iter)
    for i in range(n_iter):
        y_sh = rng.permutation(y)
        q2, _, _ = loo_cv(X, y_sh, model_factory=model_factory)
        q2_arr[i] = q2
    return q2_arr
