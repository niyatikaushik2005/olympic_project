import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def make_synthetic(train, features, frac=0.5, seed=0):
    """Labelled synthetic rows: interpolate between random pairs of real medal-winning rows
    (SMOTE-style). Result is flagged synthetic=True and kept apart from real data."""
    rng = np.random.default_rng(seed)
    pos = train[train.target == 1][features].values
    n = int(frac * len(pos))
    i, j = rng.integers(len(pos), size=n), rng.integers(len(pos), size=n)
    lam = rng.random((n, 1))
    syn = pd.DataFrame(pos[i] * lam + pos[j] * (1 - lam), columns=features)
    syn["target"], syn["synthetic"] = 1, True
    return syn

def validate(real, syn, features):
    """Compare synthetic vs real medal rows: KS test per feature + correlation difference."""
    real_pos = real[real.target == 1]
    ks = pd.DataFrame({f: ks_2samp(real_pos[f], syn[f]) for f in features}, index=["KS stat", "p-value"]).T
    corr_diff = (real_pos[features].corr() - syn[features].corr()).abs().mean().mean()
    return ks, corr_diff
