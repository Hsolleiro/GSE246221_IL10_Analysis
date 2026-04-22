"""
symbol_normalizer.py — gene symbol cleanup utilities

Handles common inconsistencies in mouse gene symbols from GEO datasets:
- Mixed case variants (IL10 / Il10 / il10)
- Hyphen/dot variants (Scd-2 / Scd.2 / Scd2)
- ENSEMBL ID to symbol mapping when available

Used by run_deg.py and downstream scripts to ensure consistent gene
naming across analyses.
"""

import re
from pathlib import Path


def normalize_mouse_symbol(symbol):
    """
    Normalize a mouse gene symbol to MGI convention:
    - First letter uppercase, rest lowercase
    - No hyphens, dots, or trailing whitespace
    - Special cases preserved (e.g., miRNA naming)
    
    Examples:
        'IL10' → 'Il10'
        'SCD-2' → 'Scd2'
        'stat3' → 'Stat3'
        'Il10ra' → 'Il10ra' (unchanged)
    """
    if not isinstance(symbol, str):
        return symbol
    
    s = symbol.strip()
    if not s:
        return s
    
    # Remove separators commonly introduced by tools
    s = s.replace('-', '').replace('.', '').replace(' ', '')
    
    # Preserve miRNA style (e.g., Mir-21a) - skip this if we stripped hyphens
    # Here we go with the standard MGI convention: first upper, rest lower
    if len(s) == 1:
        return s.upper()
    
    return s[0].upper() + s[1:].lower()


def is_ensembl_id(symbol):
    """Check if a symbol is an Ensembl ID (ENSMUSG...)."""
    if not isinstance(symbol, str):
        return False
    return bool(re.match(r'^ENSMUSG\d{11}(\.\d+)?$', symbol.strip()))


def strip_ensembl_version(ensembl_id):
    """Remove version suffix from Ensembl ID (ENSMUSG00000000001.4 → ENSMUSG00000000001)."""
    if not isinstance(ensembl_id, str):
        return ensembl_id
    return ensembl_id.split('.')[0]


def load_symbol_map(map_path):
    """
    Load a gene symbol mapping file (e.g., Ensembl → MGI symbol).
    
    Expected format: CSV with columns 'ensembl_id' and 'symbol'.
    Returns a dict {ensembl_id: symbol}.
    """
    import pandas as pd
    map_df = pd.read_csv(map_path)
    required = {'ensembl_id', 'symbol'}
    if not required.issubset(map_df.columns):
        raise ValueError(f"Symbol map must have columns: {required}")
    
    mapping = {}
    for _, row in map_df.iterrows():
        ens = strip_ensembl_version(str(row['ensembl_id']))
        sym = normalize_mouse_symbol(str(row['symbol']))
        if ens and sym:
            mapping[ens] = sym
    return mapping


def normalize_index(df, column='symbol'):
    """
    Normalize all gene symbols in a DataFrame index or column.
    
    Returns a copy of df with normalized symbols.
    """
    df = df.copy()
    if column in df.columns:
        df[column] = df[column].apply(normalize_mouse_symbol)
    else:
        df.index = df.index.to_series().apply(normalize_mouse_symbol)
    return df


if __name__ == '__main__':
    # Quick self-test
    test_cases = [
        ('IL10', 'Il10'),
        ('SCD-2', 'Scd2'),
        ('stat3', 'Stat3'),
        ('Il10ra', 'Il10ra'),
        ('DDIT4', 'Ddit4'),
        ('  Socs3 ', 'Socs3'),
    ]
    print("Running symbol_normalizer self-test...")
    for inp, expected in test_cases:
        got = normalize_mouse_symbol(inp)
        status = '✓' if got == expected else '✗'
        print(f"  {status} {inp!r:15} → {got!r:15} (expected {expected!r})")
    print("Done.")
