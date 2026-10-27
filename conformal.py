"""Site-Mondrian, label-conditional conformal risk sets with delayed-feedback adaptation,
innovation-CUSUM drift detection and innovation-gated deferral."""
import numpy as np

ALPHA = 0.10
H_CONF = 2.0


def label_known(T, E, h=H_CONF):
    return ((T <= h) & (E == 1)) | (T > h)


def label(T, E, h=H_CONF):
    return ((T <= h) & (E == 1)).astype(int)


def scores(p, y):
    """nonconformity 1 - p_hat(y) for binary horizon event"""
    return np.where(y == 1, 1 - p, p)


def conf_quantile(sc, alpha, w=None):
    sc = np.asarray(sc)
    n = len(sc)
    if n == 0 or alpha <= 0:
        return 1.0
    if alpha >= 1:
        return -1.0
    if w is None:
        k = int(np.ceil((n + 1) * (1 - alpha)))
        if k > n:
            return 1.0
        return np.sort(sc)[k - 1]
    o = np.argsort(sc)
    cw = np.cumsum(w[o]) / (w.sum() + 1.0)
    i = np.searchsorted(cw, 1 - alpha)
    return 1.0 if i >= n else sc[o][i]


def cusum(z, k=0.5, h=12.0, ref_mu=0.0, ref_sd=1.0):
    """One-sided CUSUM on standardised innovation statistics; returns alarm index or -1 and the path."""
    c, path = 0.0, []
    for i, zi in enumerate((z - ref_mu) / ref_sd):
        c = max(0.0, c + zi - k)
        path.append(c)
        if c > h:
            return i, np.array(path)
    return -1, np.array(path)


def nis_z(nis, nobs):
    return (nis - nobs) / np.sqrt(2 * np.maximum(nobs, 1))


def run_stream(p, y, known, t_pred, t_reveal, cal_s, cal_y, method="aci", alpha=ALPHA, gamma=0.01,
               gate=None, alarm_t=None, cons_factor=0.5, n_min=30, window=None, rescore=False):
    """Online label-conditional conformal sets for one site.

    p: predicted 2-y risk per test visit (calendar-ordered), y: true label, known: label determinable,
    t_pred: calendar time of prediction, t_reveal: calendar time label becomes available.
    method: 'static' (frozen calibration), 'aci' (adaptive alpha with delayed feedback + growing calibration),
            'drift' (aci + conservative alpha after alarm until n_min post-alarm labels + optional gate).
    Returns sets as (n, 2) boolean array [contains 0, contains 1] and a deferral flag.
    """
    n = len(p)
    order = np.argsort(t_pred)
    sets = np.zeros((n, 2), bool)
    a_t = {0: alpha, 1: alpha}
    pool = {c: list(cal_s[cal_y == c]) for c in (0, 1)}
    pool_t = {c: [-np.inf] * len(pool[c]) for c in (0, 1)}
    pending = []  # (reveal_time, idx)
    post_alarm = {0: 0, 1: 0}
    for idx in order:
        t = t_pred[idx]
        if method != "static":
            # reveal labels whose time has come
            still = []
            for (tr, j) in pending:
                if tr <= t:
                    c = y[j]
                    s = 1 - p[j] if c == 1 else p[j]
                    if rescore:
                        # delay-robust update: judge the revealed score against the current threshold
                        err = 0.0 if s <= conf_quantile(np.array(pool[c]), a_t[c]) else 1.0
                    else:
                        err = 0.0 if sets[j, c] else 1.0
                    a_t[c] = float(a_t[c] + gamma * (alpha - err))
                    pool[c].append(s)
                    pool_t[c].append(t_pred[j])
                    if alarm_t is not None and t_pred[j] >= alarm_t:
                        post_alarm[c] += 1
                else:
                    still.append((tr, j))
            pending = still
        for c in (0, 1):
            a = a_t[c]
            sc = np.array(pool[c])
            if method == "drift" and alarm_t is not None and t >= alarm_t and post_alarm[c] < n_min:
                a = a * cons_factor
            if window is not None and len(sc) > window:
                sc = sc[-window:]
            q = conf_quantile(sc, a)
            s = 1 - p[idx] if c == 1 else p[idx]
            sets[idx, c] = s <= q
        if known[idx] and method != "static":
            pending.append((t_reveal[idx], idx))
    defer = sets.sum(1) != 1
    if gate is not None:
        defer = defer | gate
    return sets, defer


def summarise(sets, defer, y, known, mask=None):
    m = known.copy() if mask is None else (known & mask)
    if m.sum() == 0:
        return {}
    cov = sets[np.arange(len(y)), y]
    ev = m & (y == 1)
    ne = m & (y == 0)
    nd = m & ~defer
    single1 = sets[:, 1] & ~sets[:, 0]
    single0 = sets[:, 0] & ~sets[:, 1]
    return dict(n=int(m.sum()), n_ev=int(ev.sum()), cov=float(cov[m].mean()), cov_ev=float(cov[ev].mean()) if ev.any() else np.nan,
                cov_ne=float(cov[ne].mean()) if ne.any() else np.nan, defer=float(defer[m].mean()),
                miss_ev=float((single0 & ev & ~defer).sum() / max(ev.sum(), 1)),
                acc_auto=float(((single1 & (y == 1)) | (single0 & (y == 0)))[nd].mean()) if nd.any() else np.nan)
