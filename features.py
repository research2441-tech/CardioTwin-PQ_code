"""Hand-engineered clinical features per visit for the classical survival baselines."""
import numpy as np
from scipy.signal import find_peaks
from common import load_sim, visit_table, static_cov, DATA, RUNS
from ssl_encoder import detrend

FS = 100


def wave_feats(e, p, has_e, has_p):
    f = np.full(6, np.nan)
    if has_e:
        pk, _ = find_peaks(e, distance=30, prominence=np.std(e) * 1.5)
        if len(pk) >= 2:
            rr = np.diff(pk) / FS
            f[0] = 60 / rr.mean()
            f[1] = rr.std()
            f[2] = np.median(e[pk])
        if has_p and len(pk) >= 1:
            pp, _ = find_peaks(p, distance=30)
            pats = [((pp[pp > r][0] - r) / FS) for r in pk if (pp > r).any() and (pp[pp > r][0] - r) < 60]
            if pats:
                f[3] = np.median(pats)
    if has_p:
        f[4] = np.percentile(p, 95) - np.percentile(p, 5)
        f[5] = np.abs(np.diff(p)).mean()
    return f


def main():
    pats, visits = load_sim()
    V = visit_table(pats, visits)
    E = detrend(np.load(f"{DATA}/ecg.npy"))
    P = np.load(f"{DATA}/ppg.npy")
    W = np.stack([wave_feats(E[i], P[i], V["has_e"][i], V["has_p"][i]) for i in range(len(E))])
    vit = V["vit"]
    lab = V["lab"].copy()
    lab[V["has_l"] == 0] = np.nan
    X = np.concatenate([W, vit.mean(1), vit[:, :, 0].std(1, keepdims=True), lab, static_cov(V)], 1)
    names = ["hr_ecg", "rr_sd", "r_amp", "pat", "ppg_amp", "ppg_slope", "sbp", "dbp", "hr", "spo2", "resp", "sbp_sd",
             "ldl", "hdl", "hba1c", "egfr", "lnbnp", "lntrop", "age", "male", "smoke", "diab", "pmi", "htn", "statin"]
    np.savez(f"{RUNS}/features.npz", X=X.astype(np.float32), names=np.array(names))
    print(X.shape, np.isnan(X).mean(0).round(2))


if __name__ == "__main__":
    main()
