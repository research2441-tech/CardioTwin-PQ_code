"""Shared data handling, splits, labels and metrics for CardioTwin-PQ experiments."""
import os
import pickle
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

DATA = os.environ.get("CT_DATA", os.path.join(_HERE, "data"))
RUNS = os.environ.get("CT_RUNS", os.path.join(_HERE, "runs"))
FIGS = os.environ.get("CT_FIGS", os.path.join(_HERE, "figs"))
ROUNDS = int(os.environ.get("CT_ROUNDS", "40"))
for _d in (DATA, RUNS, FIGS):
    os.makedirs(_d, exist_ok=True)
N_INT = 12          # discrete risk intervals
DELTA = 0.25        # years per interval
HORIZONS = (1.0, 2.0, 3.0)
INTERNAL = (0, 1, 2, 3, 4)
EXTERNAL = 5
T_TRAIN_END = 3.5   # calendar year: training data and labels are frozen here
ENROL_SPLIT = 2.5   # patients enrolled before this year form the development cohort
MODS = ("ecg", "ppg", "vit", "lab")


def load_sim():
    with open(f"{DATA}/sim.pkl", "rb") as f:
        d = pickle.load(f)
    return d["pats"], d["visits"]


def make_splits(pats, seed=0):
    rng = np.random.default_rng(1000 + seed)
    role = {}
    for p in pats:
        if p["site"] == EXTERNAL:
            role[p["pid"]] = "ext"
        elif p["enrol"] < ENROL_SPLIT:
            u = rng.random()
            role[p["pid"]] = "cal" if u < 0.2 else ("val" if u < 0.28 else "train")
        else:
            role[p["pid"]] = "test"
    return role


def visit_table(pats, visits):
    """Flatten visits into arrays with per-visit labels (full and training-censored)."""
    P = {p["pid"]: p for p in pats}
    n = len(visits)
    out = {k: np.zeros(n) for k in ("pid", "site", "vidx", "t", "cal", "age", "htn", "statin", "af",
                                     "has_e", "has_p", "has_l", "T", "E", "Ttr", "Etr", "male", "diab",
                                     "smoke", "pmi", "loghz", "shifted")}
    for i, v in enumerate(visits):
        p = P[v["pid"]]
        for k in ("pid", "site", "vidx", "t", "cal", "age", "htn", "statin", "af", "has_e", "has_p", "has_l", "loghz"):
            out[k][i] = v[k]
        for k in ("male", "diab", "smoke", "pmi", "shifted"):
            out[k][i] = p[k]
        rem = p["time"] - v["t"]               # time from visit to event/censoring
        out["T"][i] = rem
        out["E"][i] = p["event"]
        adm = T_TRAIN_END - p["enrol"] - v["t"]  # administrative censoring for development data
        out["Ttr"][i] = min(rem, max(adm, 0.0))
        out["Etr"][i] = p["event"] if rem <= adm else 0
    for k in ("pid", "site", "vidx", "htn", "statin", "af", "has_e", "has_p", "has_l", "E", "Etr", "male", "diab",
              "smoke", "pmi", "shifted"):
        out[k] = out[k].astype(int)
    out["vit"] = np.stack([v["vit"] for v in visits]).astype(np.float32)
    out["lab"] = np.stack([v["lab"] for v in visits]).astype(np.float32)
    out["z"] = np.stack([v["z"] for v in visits]).astype(np.float32)
    return out


def static_cov(V):
    """Time-varying EHR covariates used by every model (7 columns)."""
    return np.stack([(V["age"] - 60) / 10, V["male"], V["smoke"], V["diab"], V["pmi"], V["htn"], V["statin"]],
                    1).astype(np.float32)


def interval_labels(T, E):
    """Interval index of event/censoring (N_INT means beyond the last interval) and event flag."""
    k = np.minimum(np.floor(T / DELTA).astype(int), N_INT)
    ev = ((E == 1) & (k < N_INT)).astype(int)
    return k, ev


# ---------------------------------------------------------------- metrics
def td_auc(risk, T, E, h):
    """Cumulative/dynamic AUC at horizon h: cases event<=h, controls T>h; censored-before-h excluded."""
    case = (T <= h) & (E == 1)
    ctrl = T > h
    if case.sum() == 0 or ctrl.sum() == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([risk[case], risk[ctrl]]))
    n1, n0 = case.sum(), ctrl.sum()
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def harrell_c(risk, T, E, max_pairs=4_000_000, rng=None):
    rng = rng or np.random.default_rng(0)
    idx = np.where(E == 1)[0]
    n = len(T)
    if len(idx) == 0:
        return np.nan
    # sample comparable pairs (i event, j with T_j > T_i)
    m = min(max_pairs, len(idx) * n)
    i = rng.choice(idx, m)
    j = rng.integers(0, n, m)
    ok = T[j] > T[i]
    i, j = i[ok], j[ok]
    conc = (risk[i] > risk[j]).sum() + 0.5 * (risk[i] == risk[j]).sum()
    return conc / len(i)


def brier_h(risk, T, E, h):
    lab = ((T <= h) & (E == 1)).astype(float)
    known = ((T <= h) & (E == 1)) | (T > h)
    return np.mean((risk[known] - lab[known]) ** 2)


def ece_h(risk, T, E, h, bins=10):
    lab = ((T <= h) & (E == 1)).astype(float)
    known = ((T <= h) & (E == 1)) | (T > h)
    r, y = risk[known], lab[known]
    q = np.quantile(r, np.linspace(0, 1, bins + 1))
    b = np.clip(np.searchsorted(q, r, side="right") - 1, 0, bins - 1)
    e = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            e += m.mean() * abs(r[m].mean() - y[m].mean())
    return e


def evaluate(risks, T, E):
    """risks: (n, 3) cumulative risk at 1, 2, 3 years."""
    return dict(auc1=td_auc(risks[:, 0], T, E, 1.0), auc2=td_auc(risks[:, 1], T, E, 2.0),
                auc3=td_auc(risks[:, 2], T, E, 3.0), c=harrell_c(risks[:, 2], T, E),
                brier2=brier_h(risks[:, 1], T, E, 2.0), ece2=ece_h(risks[:, 1], T, E, 2.0))


def boot_patients(pid, fn, B=500, seed=0):
    """Patient-cluster bootstrap of a statistic fn(index_array)."""
    rng = np.random.default_rng(seed)
    up = np.unique(pid)
    groups = {u: np.where(pid == u)[0] for u in up}
    vals = []
    for _ in range(B):
        s = rng.choice(up, len(up))
        idx = np.concatenate([groups[u] for u in s])
        vals.append(fn(idx))
    return np.array(vals)
