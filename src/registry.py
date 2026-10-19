#!/usr/bin/env python3

import pandas as pd
from statsmodels.stats.multitest import multipletests

class ModelRegistry:
    def __init__(self):
        self.entries = []
        
    def add(self, name, features, q2, p_yrand, rmse=None):
        self.entries.append({
            'name': name,
            'features': ' + '.join(features) if isinstance(features, list) else features,
            'q2': q2,
            'rmse': rmse,
            'p_yrand': p_yrand,
        })
        
    def to_dataframe(self):
        return pd.DataFrame(self.entries)
        
    def apply_corrections(self):
        df = self.to_dataframe()
        if len(df) == 0 or df['p_yrand'].isna().all():
            return df
        
        mask = df['p_yrand'].notna()
        pvals = df.loc[mask, 'p_yrand'].values
        
        _, p_bonf, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')
        reject_bh, p_bh, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        
        df.loc[mask, 'p_bonf'] = p_bonf
        df.loc[mask, 'signif_bonf'] = p_bonf < 0.05
        df.loc[mask, 'p_bh'] = p_bh
        df.loc[mask, 'signif_bh'] = reject_bh
        
        return df
    
