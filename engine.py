"""Transparent illustrative prop simulator. Not a predictive betting model."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

REQUIRED = ['sport','game_id','player','team','market','side','line','american_odds','projection','std_dev']

def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    if odds == 0: raise ValueError('American odds cannot be zero')
    return 1 + (odds / 100 if odds > 0 else 100 / abs(odds))

def decimal_to_american(decimal: float) -> str:
    if decimal <= 1: return 'N/A'
    x = (decimal - 1) * 100 if decimal >= 2 else -100 / (decimal - 1)
    return f'{x:+.0f}'

def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing: raise ValueError('Missing columns: ' + ', '.join(missing))
    out = df.copy().reset_index(drop=True)
    for c in ['sport','game_id','player','team','market','side']:
        out[c] = out[c].astype(str).str.strip()
    out['side'] = out['side'].str.upper()
    if not out['side'].isin(['OVER','UNDER']).all(): raise ValueError('side must be OVER or UNDER')
    for c in ['line','american_odds','projection','std_dev']:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    if out[['line','american_odds','projection','std_dev']].isna().any().any():
        raise ValueError('Numeric columns contain missing or invalid values')
    if (out['std_dev'] <= 0).any(): raise ValueError('std_dev must be greater than 0')
    if (out['american_odds'] == 0).any(): raise ValueError('American odds cannot be zero')
    if out['game_id'].eq('').any(): raise ValueError('game_id cannot be empty')
    out['decimal_odds'] = out['american_odds'].map(american_to_decimal)
    return out

def simulate(df: pd.DataFrame, n: int=20000, seed: int=42, game_rho: float=.12,
             team_rho: float=.16, player_rho: float=.22) -> tuple[pd.DataFrame,np.ndarray]:
    """Gaussian latent factors for game/team/player; rho are variance shares, not pairwise correlations.
    Continuous normal stat model: overs are strictly greater, unders strictly less than line.
    Exact-line pushes are not modeled. For integer props, use half-point lines.
    """
    if not 0 <= game_rho <= 1 or not 0 <= team_rho <= 1 or not 0 <= player_rho <= 1:
        raise ValueError('Factor shares must be in [0,1]')
    if game_rho + team_rho + player_rho >= 1:
        raise ValueError('Factor shares must sum to less than 1')
    df = validate(df)
    rng = np.random.default_rng(seed)
    m = len(df)
    z = rng.standard_normal((n,m)) * math.sqrt(1-game_rho-team_rho-player_rho)
    # Shared positive factors induce positive stat correlations, not necessarily positive bet correlations.
    for share, keys in [(game_rho, ['game_id']), (team_rho, ['game_id','team']),
                        (player_rho, ['game_id','player'])]:
        if share == 0: continue
        groups = df.groupby(keys, sort=False).indices
        for idx in groups.values():
            z[:,idx] += math.sqrt(share) * rng.standard_normal((n,1))
    values = df['projection'].to_numpy()[None,:] + z * df['std_dev'].to_numpy()[None,:]
    # Clip at zero for nonnegative counts/yards; simplified continuous approximation.
    values = np.maximum(values, 0)
    lines = df['line'].to_numpy()[None,:]
    overs = df['side'].to_numpy()[None,:] == 'OVER'
    wins = np.where(overs, values > lines, values < lines)
    probs = wins.mean(axis=0)
    result = df.copy()
    result['model_prob'] = probs
    result['implied_prob'] = 1 / result['decimal_odds']
    result['ev_per_dollar'] = result['model_prob'] * result['decimal_odds'] - 1
    result['fair_odds'] = [decimal_to_american(1/p) if p > 0 else 'N/A' for p in probs]
    return result, wins

def sgp_portfolio(df: pd.DataFrame, wins: np.ndarray, game_id: str,
                  min_legs: int=2, max_legs: int=3, top_k: int=10) -> pd.DataFrame:
    from itertools import combinations
    sub = df[df['game_id'] == game_id]
    idx = sub.index.to_list()
    if len(idx) < min_legs: return pd.DataFrame()
    rows = []
    # Limit candidate pool for responsiveness; preserve source row positions.
    idx = sorted(idx, key=lambda i: float(df.loc[i,'ev_per_dollar']), reverse=True)[:16]
    for k in range(min_legs, min(max_legs,len(idx))+1):
        for combo in combinations(idx,k):
            # Do not offer mutually exclusive over/under lines on same player+market.
            keys = [(df.loc[i,'player'],df.loc[i,'market']) for i in combo]
            if len(set(keys)) < len(keys): continue
            p = float(np.all(wins[:,list(combo)], axis=1).mean())
            # Independent product is NOT a sportsbook SGP quote; show only as diagnostic.
            independent = float(np.prod(df.loc[list(combo),'model_prob']))
            legs = ' | '.join(f"{df.loc[i,'player']} {df.loc[i,'market']} {df.loc[i,'side']} {df.loc[i,'line']:g}" for i in combo)
            rows.append({'legs':legs,'n_legs':k,'joint_probability':p,
                         'model_fair_odds':decimal_to_american(1/p) if p > 0 else 'N/A',
                         'independent_probability':independent,
                         'correlation_lift':p/independent if independent > 0 else np.nan})
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(['joint_probability','n_legs'],ascending=[False,True]).head(top_k).reset_index(drop=True)
