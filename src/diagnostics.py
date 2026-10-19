#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_pearson(x, y):
    """Линейная корреляция. Вход: два вектора. Выход: r, p."""
    r, p = pearsonr(x, y)
    return r, p
    
def calc_spearman(x, y):
    """Ранговая корреляция. Вход: два вектора. Выход: ρ, p."""
    rho, p = spearmanr(x, y)
    return rho, p
    
def correlate_with_target(df, features, target):
    """Корреляции всех признаков с таргетом. Выход: DataFrame."""
    rows = []
    for f in features:
        r, p_r = calc_pearson(df[f], df[target])
        rho, p_rho = calc_spearman(df[f], df[target])
        rows.append({
            'feature': f,
            'pearson_r': r,
            'pearson_p': p_r,
            'spearman_rho': rho,
            'spearman_p': p_rho,
        })
    return pd.DataFrame(rows).sort_values('spearman_rho', ascending=False)
    
def calc_vif(X_df):
    """VIF для всех признаков."""
    X = X_df.copy().astype(float)
    X['const'] = 1
    vifs = []
    for i, col in enumerate(X.columns):
        if col == 'const':
            continue
        vifs.append({
            'feature': col,
            'VIF': variance_inflation_factor(X.values, i),
        })
    return pd.DataFrame(vifs).sort_values('VIF', ascending=False)
    
def cooks_distance(X, y):
    """Cook's D для поиска leverage points."""
    import statsmodels.api as sm
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit()
    return model.get_influence().cooks_distance[0]
    

