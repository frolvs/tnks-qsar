#!/usr/bin/env python3

## main.py
import pandas as pd
from rdkit import Chem
from data_loader import load_csv
from molecules import calc_2d_descriptors, calc_3d_descriptors, classify_scaffold
from features import build_features
from diagnostics import calc_vif
from cv import loo_cv, repeated_stratified_cv, y_randomization
from tqdm import tqdm

tqdm.pandas(desc="Считаем...") 

def main():
    # 1. Загрузка
    df = load_csv('data/tnks_series.csv')
    
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
    # Берем базовые числовые колонки
    base_features = ['logMW_c', 'RotBonds']

    # Автоматически находим все колонки со скаффолдами и их взаимодействиями
    scaffold_features = [c for c in df.columns if c.startswith('sc_')]

    # Собираем всё вместе
    features = base_features + scaffold_features

    print(f"Используемые признаки: {features}")

    # Выбираем их из датафрейма
    X = df[features].values
    
    # выбираем активности
    y = df['TNKS-1_pIC50'].values
    calc_vif(df[features])  # проверка коллинеарности
    
    # 5. Валидация
    q2, rmse, _ = loo_cv(X, y)
    print(f"Q² = {q2:.3f}")
    
if __name__ == "__main__":
    main()
