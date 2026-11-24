import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def allocate(cands, budget, max_sports, lo=0.5, hi=1.5):
    """MILP: choose sports (binary x) and money (continuous a) to maximise predicted medal opportunity.
    cands needs columns: Sport, prob, cost. If a sport is chosen its funding must be in [lo, hi] x cost.
    Variables: [x_1..x_n, a_1..a_n]."""
    n = len(cands)
    p, c = cands.prob.values, cands.cost.values
    obj = -np.concatenate([np.zeros(n), p / c])                       # maximise -> minimise negative
    I = np.eye(n)
    cons = [LinearConstraint(np.concatenate([np.zeros(n), np.ones(n)])[None], -np.inf, budget),   # budget
            LinearConstraint(np.concatenate([np.ones(n), np.zeros(n)])[None], -np.inf, max_sports),  # #sports
            LinearConstraint(np.hstack([-lo * c[:, None] * I, I]), 0, np.inf),                   # a >= lo*cost*x
            LinearConstraint(np.hstack([-hi * c[:, None] * I, I]), -np.inf, 0)]                  # a <= hi*cost*x
    res = milp(obj, constraints=cons, integrality=np.concatenate([np.ones(n), np.zeros(n)]),
               bounds=Bounds(np.zeros(2 * n), np.concatenate([np.ones(n), np.full(n, np.inf)])))
    x, a = np.round(res.x[:n]), res.x[n:]
    out = cands.assign(selected=x.astype(int), allocation=a.round(2))
    return out[out.selected == 1].sort_values("allocation", ascending=False), -res.fun
