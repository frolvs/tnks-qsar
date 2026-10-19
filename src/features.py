#!/usr/bin/env python3

# features.py
import numpy as np
import pandas as pd

def add_log_features(df):
    """Добавляет логарифмы."""
    df['logMW'] = np.log(df['ExactMolWt'])
    # df['logTPSA'] = np.log(df['TPSA'] + 1)
    return df

def add_scaffold_onehot(df, scaffold_col='scaffold'):
    """Создаёт one-hot для скаффолдов."""
    dummies = pd.get_dummies(df[scaffold_col], prefix='sc', drop_first=True).astype(float)
    return pd.concat([df, dummies], axis=1)

def center_features(df, cols):
    """Центрирует признаки (вычитает среднее)."""
    for col in cols:
        df[f'{col}_c'] = df[col] - df[col].mean()
    return df

def add_interactions(df, scaffold_cols, continuous_col):
    """Создаёт произведения скаффолдов на непрерывный признак."""
    for sc in scaffold_cols:
        df[f'{sc}_x_{continuous_col}'] = df[sc] * df[continuous_col]
    return df

def build_features(df, scaffold_col='scaffold'):
    """Собирает все признаки вместе."""
    df = add_log_features(df)
    df = add_scaffold_onehot(df, scaffold_col)
    
    # Получаем список скаффолд-колонок
    sc_cols = [c for c in df.columns if c.startswith('sc_') and '_x_' not in c]
    
    # Центрируем logMW
    df = center_features(df, ['logMW'])
    
    # Создаём interactions с центрированным logMW
    df = add_interactions(df, sc_cols, 'logMW_c')
    
    return df
