#!/usr/bin/env python3

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, Descriptors3D

def parse_smiles(smi):
    """Безопасный парсинг SMILES."""
    if not smi or not isinstance(smi, str):
        return None
    return Chem.MolFromSmiles(smi)

def standardize_smiles(smi):
    """Нейтрализация и канонизация."""
    mol = parse_smiles(smi)
    if mol is None:
        return None
    # Берем самый большой фрагмент (убираем соли)
    frags = Chem.GetMolFrags(mol, asMols=True)
    mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol)

def calc_2d_descriptors(mol):
    """Считает 2D дескрипторы."""
    if mol is None:
        return None
    return {
        'ExactMolWt': Descriptors.ExactMolWt(mol),
        'MolLogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'RotBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
    }
    
def calc_3d_descriptors(mol):
    """Генерирует 3D конформер и считает 3D дескрипторы."""
    if mol is None:
        return None
    
    mol_3d = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    
    if AllChem.EmbedMolecule(mol_3d, params) != 0:
        return None # Не удалось сгенерировать
        
    try:
        AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=500)
        return {
            'ISF': Descriptors3D.InertialShapeFactor(mol_3d),
            'PBF': Descriptors3D.PBF(mol_3d),
        }
    except Exception:
        return None
        
def classify_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return 'invalid'
    
    # 1. Найти C=O
    co_c = None
    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() != 2.0: continue
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if a1.GetSymbol() == 'C' and a2.GetSymbol() == 'O':
            co_c = a1; break
        if a2.GetSymbol() == 'C' and a1.GetSymbol() == 'O':
            co_c = a2; break
    if co_c is None: return 'no_c=o'
    
    # 2. C4a
    c4a = None
    for n in co_c.GetNeighbors():
        if n.GetSymbol() == 'C' and n.GetIsAromatic():
            c4a = n; break
    if c4a is None: return 'no_c4a'
    
    # 3. C8a — ароматический C-сосед C4a, НЕ C=O, с N-соседом
    c8a = None
    for n in c4a.GetNeighbors():
        if n.GetIdx() == co_c.GetIdx(): continue
        if n.GetSymbol() != 'C' or not n.GetIsAromatic(): continue
        if any(nn.GetSymbol() == 'N' for nn in n.GetNeighbors()):
            c8a = n; break
    if c8a is None: return 'no_c8a'
    
    # 4. Позиция 8 — ароматический C-сосед C8a, НЕ C4a
    pos8 = None
    for n in c8a.GetNeighbors():
        if n.GetIdx() == c4a.GetIdx(): continue
        if n.GetSymbol() != 'C': continue
        if not n.GetIsAromatic(): continue   # ← КЛЮЧЕВОЙ ФИКС
        pos8 = n; break
    if pos8 is None: return 'no_pos8'
    
    # 5. Заместитель
    for n in pos8.GetNeighbors():
        if n.GetIdx() == c8a.GetIdx(): continue
        if n.GetIsAromatic(): continue
        sym = n.GetSymbol()
        if sym == 'F':  return '8-F'
        if sym == 'Cl': return '8-Cl'
        if sym == 'Br': return '8-Br'
        if sym == 'C':
            if n.GetTotalNumHs() == 3: return '8-Me'
            return '8-C_other'
        if sym == 'O':
            if n.GetTotalNumHs() == 1: return '8-OH'
            return '8-OMe'
        if sym == 'N':
            has_dbl_O = any(
                nn.GetSymbol() == 'O'
                and mol.GetBondBetweenAtoms(n.GetIdx(), nn.GetIdx()).GetBondTypeAsDouble() == 2.0
                for nn in n.GetNeighbors()
            )
            if has_dbl_O: return '8-NO2'
            return '8-NH2'
    
    return '8-H'
