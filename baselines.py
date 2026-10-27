"""Baselines: classical survival models on engineered features, and neural discrete-time hazard models
(static MLP and recurrent GRU) trained centrally, locally, or with FedAvg / FedProx / Ditto."""
import time
import numpy as np
import jax, jax.numpy as jnp
import optax
from common import ROUNDS, N_INT, DELTA

DIN = 96 + 4 + 7 + 1
HID = 48


def init_seq(key, recurrent=True):
    ks = jax.random.split(key, 6)
    g = lambda k, i, o: jax.random.normal(k, (i, o)) * np.sqrt(1.0 / i)
    p = dict(inp=dict(w=g(ks[0], DIN, HID), b=jnp.zeros(HID)), out=dict(w=g(ks[1], HID, N_INT), b=jnp.full(N_INT, -3.5)))
    if recurrent:
        p["gru"] = dict(Wz=g(ks[2], 2 * HID, HID), Wr=g(ks[3], 2 * HID, HID), Wh=g(ks[4], 2 * HID, HID),
                        bz=jnp.zeros(HID), br=jnp.zeros(HID), bh=jnp.zeros(HID))
    return p


def seq_inputs(B):
    n, V = B["VM"].shape
    e = (B["E"] * B["M"][..., None]).reshape(n, V, -1)
    return jnp.concatenate([e, B["M"], B["X"], B["DT"][..., None]], -1)


def seq_forward(p, B):
    x = jax.nn.gelu(seq_inputs(B) @ p["inp"]["w"] + p["inp"]["b"])
    if "gru" in p:
        g = p["gru"]

        def cell(h, xt):
            hx = jnp.concatenate([h, xt], -1)
            z = jax.nn.sigmoid(hx @ g["Wz"] + g["bz"])
            r = jax.nn.sigmoid(hx @ g["Wr"] + g["br"])
            hh = jnp.tanh(jnp.concatenate([r * h, xt], -1) @ g["Wh"] + g["bh"])
            hn = (1 - z) * h + z * hh
            return hn, hn

        _, hs = jax.lax.scan(cell, jnp.zeros((x.shape[0], HID)), jnp.swapaxes(x, 0, 1))
        x = jnp.swapaxes(hs, 0, 1)
    return x @ p["out"]["w"] + p["out"]["b"]          # hazard logits (n, V, 12)


def seq_loss(p, B):
    logit = seq_forward(p, B)
    j = jnp.arange(N_INT)
    before = (j[None, None, :] < B["K"][..., None])
    at = (j[None, None, :] == B["K"][..., None])
    ll = (jax.nn.log_sigmoid(-logit) * before).sum(-1) + (jax.nn.log_sigmoid(logit) * at).sum(-1) * B["EV"]
    return -(ll * B["VM"]).sum() / B["VM"].sum()


def seq_risk(p, B, chunk=2048):
    f = jax.jit(lambda Bc: seq_forward(p, Bc))
    out = []
    for i in range(0, B["E"].shape[0], chunk):
        Bc = {k: B[k][i:i + chunk] for k in ("E", "M", "X", "DT", "VM")}
        h = jax.nn.sigmoid(f(Bc))
        cs = jnp.cumsum(jnp.log1p(-h), -1)
        out.append(np.asarray(1 - jnp.exp(cs[..., [3, 7, 11]])))
    return np.concatenate(out)


def _val(p, Bv):
    return sum(float(seq_loss(p, Bv[s])) * Bv[s]["VM"].sum() for s in Bv)


def train_seq(Btr, Bva, mode="fedavg", recurrent=True, seed=0, rounds=ROUNDS, local_steps=8, lr=3e-3, mu=0.05,
              lam_ditto=0.1):
    key = jax.random.PRNGKey(seed)
    p = init_seq(key, recurrent)
    opt = optax.adam(lr)
    sites = sorted(Btr)
    keys = ("E", "M", "X", "DT", "VM", "K", "EV")
    Btr = {s: {k: Btr[s][k] for k in keys} for s in sites}
    Bva = {s: {k: Bva[s][k] for k in keys} for s in Bva}

    def lf(pp, B, pg, prox):
        l = seq_loss(pp, B)
        reg = sum(((x - y) ** 2).sum() for x, y in zip(jax.tree_util.tree_leaves(pp), jax.tree_util.tree_leaves(pg)))
        return l + 0.5 * prox * reg

    @jax.jit
    def step(pp, st, B, pg, prox):
        l, g = jax.value_and_grad(lf)(pp, B, pg, prox)
        u, st = opt.update(g, st, pp)
        return optax.apply_updates(pp, u), st, l

    if mode == "central":
        Ball = {k: np.concatenate([Btr[s][k] for s in sites]) for k in keys}
        st = opt.init(p)
        best, bp = np.inf, p
        for rd in range(rounds):
            for _ in range(local_steps):
                p, st, l = step(p, st, Ball, p, 0.0)
            v = _val(p, Bva)
            if v < best:
                best, bp = v, p
        return dict(global_=bp)
    if mode == "local":
        res = {}
        for s in sites:
            ps, st = init_seq(key, recurrent), opt.init(p)
            best, bp = np.inf, ps
            for rd in range(rounds):
                for _ in range(local_steps):
                    ps, st, l = step(ps, st, Btr[s], ps, 0.0)
                v = _val(ps, {s: Bva[s]})
                if v < best:
                    best, bp = v, ps
            res[s] = bp
        return dict(local=res)
    n_s = np.array([Btr[s]["E"].shape[0] for s in sites], float)
    w = n_s / n_s.sum()
    pers = {s: p for s in sites}
    pst = {s: opt.init(p) for s in sites}
    best, bstate = np.inf, None
    for rd in range(rounds):
        loc = []
        for s in sites:
            pp, st = p, opt.init(p)
            prox = mu if mode == "fedprox" else 0.0
            for _ in range(local_steps):
                pp, st, l = step(pp, st, Btr[s], p, prox)
            loc.append(pp)
            if mode == "ditto":
                for _ in range(local_steps):
                    pers[s], pst[s], _ = step(pers[s], pst[s], Btr[s], p, lam_ditto)
        p = jax.tree_util.tree_map(lambda *xs: sum(wi * x for wi, x in zip(w, xs)), *loc)
        v = _val(p, Bva) if mode != "ditto" else sum(float(seq_loss(pers[s], Bva[s])) * Bva[s]["VM"].sum() for s in sites)
        if v < best:
            best, bstate = v, (p, dict(pers))
    out = dict(global_=bstate[0])
    if mode == "ditto":
        out["local"] = bstate[1]
    return out


# ------------------------------------------------------------------ classical baselines
def fit_classical(Xtr, Ttr, Etr, kind="cox", seed=0):
    if kind == "cox":
        from lifelines import CoxPHFitter
        import pandas as pd
        df = pd.DataFrame(Xtr, columns=[f"f{i}" for i in range(Xtr.shape[1])])
        df["T"], df["E"] = np.maximum(Ttr, 1e-3), Etr
        m = CoxPHFitter(penalizer=0.05)
        m.fit(df, "T", "E")
        return m
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    # event times are rounded up to a 0.05-unit grid to bound the memory of the leaf survival functions
    Tg = np.ceil(np.maximum(Ttr, 1e-3) / 0.05) * 0.05
    m = RandomSurvivalForest(n_estimators=100, min_samples_leaf=40, max_features="sqrt", n_jobs=2, random_state=seed)
    m.fit(Xtr, Surv.from_arrays(Etr.astype(bool), Tg))
    return m


def classical_risk(m, X, kind="cox"):
    hs = np.array([1.0, 2.0, 3.0])
    if kind == "cox":
        import pandas as pd
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        sf = m.predict_survival_function(df, times=hs).T.values
        return 1 - sf
    out = []
    for i in range(0, len(X), 2000):
        S = m.predict_survival_function(X[i:i + 2000], return_array=True)
        tt = m.unique_times_
        idx = [max(0, np.searchsorted(tt, h, side="right") - 1) for h in hs]
        out.append(1 - S[:, idx])
    return np.concatenate(out)
