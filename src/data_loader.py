#!/usr/bin/env python3

import pandas as pd
from rdkit import Chem

def load_csv(csv):
    """Input: csv. Output: pd.DataFrame"""
    df = pd.read_csv(csv)
    return df
    
def load_sdf(sdf):
    """Input: sdf. Output: pd.DataFrame"""
    supplier = Chem.SDMolSupplier(sdf)
    data = []
    for mol in supplier:
        if mol is not None:
            data.append({"SMILES": Chem.MolToSmiles(mol)})
    df = pd.DataFrame(data)
    return df
    
def fetch_chembl(target_id):
    """Input: target_id. Output: pd.DataFrame"""
    pass
    
def write_to_csv(df, output_path='output.csv'):
    """Input: pd.DataFrame. Output: csv"""
    df.to_csv(output_path, index=False)
    
