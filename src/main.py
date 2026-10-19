#!/usr/bin/env python3

## main.py
import pandas as pd
from rdkit import Chem
from data_loader import load_data
from molecules import calc_2d_descriptors, calc_3d_descriptors, classify_scaffold
from features import build_features
from diagnostics import calc_vif
from cv import loo_cv, repeated_cv, y_randomization
from tqdm import tqdm

tqdm.pandas(desc="Считаем...") 

def main():
    # 1. Загрузка
    df = load_data('data/tnks_series.csv')
    
    # 2. Базовые дескрипторы
    df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    # Проверяем, все ли распарсились
    if df['mol'].isna().any():
        print(f"Внимание: {df['mol'].isna().sum()} SMILES не распарсились")
    
    # Применяем функцию к каждой молекуле (получаем Series словарей)
    desc_2d_series = df['mol'].apply(calc_2d_descriptors)
    # Превращаем Series словарей в DataFrame
    # .tolist() превращает Series в список словарей, а pd.DataFrame создает таблицу
    desc_2d_df = pd.DataFrame(desc_2d_series.tolist())
    # Склеиваем с исходным DataFrame по индексу (axis=1)
    df = pd.concat([df, desc_2d_df], axis=1)
    
    desc_3d_series = df['mol'].progress_apply(calc_3d_descriptors)
    desc_3d_df = pd.DataFrame(desc_3d_series.tolist())
    df = pd.concat([df, desc_3d_df], axis=1)
    
    # 3. Фичи (логарифмы, one-hot, interactions)
    if 'scaffold' not in df.columns:
        print("Колонка 'scaffold' не найдена. Генерируем через classify_scaffold()...")
        df['scaffold'] = df['SMILES'].apply(classify_scaffold)
    
    df = build_features(df, scaffold_col='scaffold')
    
    # 4. Диагностика
    X = df[['logMW_c', 'sc_Me_x_logMW_c', 'RotBonds']].values
    y = df['TNKS-1_pIC50'].values
    calc_vif(X)  # проверка коллинеарности
    
    # 5. Валидация
    q2, rmse, _ = loo_cv(X, y)
    print(f"Q² = {q2:.3f}")
