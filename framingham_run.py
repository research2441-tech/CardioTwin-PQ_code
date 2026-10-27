"""Real-data study on the Framingham teaching cohort: federated self-supervised tabular encoders,
the state-space twin with institution adapters, baselines, and site-Mondrian conformal sets."""
import sys, pickle, time
import numpy as np
import jax, jax.numpy as jnp
import optax
import twin as T
from baselines import train_seq, seq_risk, fit_classical, classical_risk
from common import ROUNDS, RUNS, evaluate, interval_labels, boot_patients
from conformal import run_stream, summarise, label, label_known, scores
from framingham_prep import load, arrays, GROUPS
from ssl_encoder import infonce

GN = list(GROUPS)
INT_SITES = (0, 1, 2)
EXT_SITE = 3


def splits(V, seed=0):
    rng = np.random.default_rng(77)
    role = {}
    for u in np.unique(V["pid"]):
        s = V["site"][V["pid"] == u][0]
        r = rng.random()
        if s == EXT_SITE:
            role[u] = "pool" if r < 0.5 else "ext"
        else:
            role[u] = "train" if r < 0.6 else ("val" if r < 0.7 else ("cal" if r < 0.85 else "test"))
    return np.array([role[u] for u in V["pid"]])


def ssl_train(F, mask_tr, sites, seed=0, rounds=min(30, ROUNDS), local=20, bs=256):
    """Masked-feature reconstruction + cross-group InfoNCE, FedAvg over institutions."""
    X, M = {}, {}
    for g in GN:
        x = F[g]
        m = (~np.isnan(x)).astype(np.float32)
        mu, sd = np.nanmean(x[mask_tr], 0), np.nanstd(x[mask_tr], 0) + 1e-6
        X[g] = np.nan_to_num((x - mu) / sd).astype(np.float32)
        M[g] = m
    key = jax.random.PRNGKey(seed)
    ks = iter(jax.random.split(key, 40))
    dn = lambda i, o: dict(w=jax.random.normal(next(ks), (i, o)) * np.sqrt(2 / i), b=jnp.zeros(o))
    P = {g: dict(h=dn(2 * X[g].shape[1], 32), o=dn(32, T.DE), d=dn(T.DE, X[g].shape[1]), pr=dn(T.DE, 16)) for g in GN}

    def enc(pg, x, m):
        return jax.nn.gelu(jnp.concatenate([x * m, m], -1) @ pg["h"]["w"] + pg["h"]["b"]) @ pg["o"]["w"] + pg["o"]["b"]

    def loss(Pp, b, k):
        tot, H = 0.0, {}
        for i, g in enumerate(GN):
            x, m = b[g]
            drop = (jax.random.uniform(jax.random.fold_in(k, i), m.shape) < 0.3) * m
            h = enc(Pp[g], x, m - drop)
            rec = ((h @ Pp[g]["d"]["w"] + Pp[g]["d"]["b"] - x) ** 2 * drop).sum() / (drop.sum() + 1e-6)
            tot += rec
            H[g] = (enc(Pp[g], x, m) @ Pp[g]["pr"]["w"], (m.sum(1) > 0).astype(jnp.float32))
        pairs = [("vit", "metab"), ("vit", "hist"), ("metab", "lipid"), ("vit", "lipid"), ("metab", "hist")]
        tot += 0.5 * sum(infonce(H[a][0], H[b_][0], H[a][1] * H[b_][1]) for a, b_ in pairs) / len(pairs)
        return tot

    opt = optax.adam(1e-3)

    @jax.jit
    def step(Pp, st, b, k):
        l, g = jax.value_and_grad(loss)(Pp, b, k)
        u, st = opt.update(g, st, Pp)
        return optax.apply_updates(Pp, u), st, l

    rng = np.random.default_rng(seed)
    idx = {s: np.where(mask_tr & (sites == s))[0] for s in INT_SITES}
    w = np.array([len(idx[s]) for s in INT_SITES], float)
    w /= w.sum()
    for rd in range(rounds):
        loc = []
        for s in INT_SITES:
            pp, st = P, opt.init(P)
            for _ in range(local):
                bi = rng.choice(idx[s], min(bs, len(idx[s])), replace=False)
                key, sk = jax.random.split(key)
                pp, st, l = step(pp, st, {g: (X[g][bi], M[g][bi]) for g in GN}, sk)
            loc.append(pp)
        P = jax.tree_util.tree_map(lambda *xs: sum(wi * x for wi, x in zip(w, xs)), *loc)
    emb = np.stack([np.asarray(enc(P[g], X[g], M[g])) for g in GN], 1)
    avail = np.stack([(M[g].sum(1) > 0) for g in GN], 1).astype(np.float32)
    return emb, avail, float(l)


def pack(V, emb, avail, F, rows, stats=None):
    rows = rows[np.lexsort((V["vidx"][rows], V["pid"][rows]))]
    up, start = np.unique(V["pid"][rows], return_index=True)
    n = len(up)
    xs = np.stack([(V["age"] - 60) / 10, V["male"], V["smoke"], V["diab"], V["pmi"], V["htn"], V["statin"]], 1).astype(np.float32)
    yraw = np.stack([F["vit"][:, 0], F["metab"][:, 0], F["vit"][:, 2]], 1)
    if stats is None:
        mu = np.zeros((T.NM, T.DE)); sd = np.ones((T.NM, T.DE))
        for m in range(T.NM):
            a = emb[rows][avail[rows, m] > 0, m]
            mu[m], sd[m] = a.mean(0), a.std(0) + 1e-6
        stats = (mu, sd, np.nanmean(yraw[rows], 0), np.nanstd(yraw[rows], 0))
    mu, sd, ym, ys = stats
    kk, ee = interval_labels(V["T"], V["E"])
    B = dict(E=np.zeros((n, T.VMAX, T.NM, T.DE), np.float32), M=np.zeros((n, T.VMAX, T.NM), np.float32),
             X=np.zeros((n, T.VMAX, 7), np.float32), DT=np.zeros((n, T.VMAX), np.float32), VM=np.zeros((n, T.VMAX), np.float32),
             K=np.zeros((n, T.VMAX), np.int32), EV=np.zeros((n, T.VMAX), np.int32), Y=np.zeros((n, T.VMAX, T.NY), np.float32),
             ROW=-np.ones((n, T.VMAX), np.int64))
    ends = list(start[1:]) + [len(rows)]
    for i, (a, b) in enumerate(zip(start, ends)):
        r = rows[a:b]
        L = len(r)
        B["E"][i, :L] = (emb[r] - mu) / sd * avail[r][..., None]
        B["M"][i, :L] = avail[r]
        B["X"][i, :L] = xs[r]
        B["DT"][i, 1:L] = np.diff(V["t"][r])
        B["VM"][i, :L] = 1
        B["K"][i, :L], B["EV"][i, :L] = kk[r], ee[r]
        B["Y"][i, :L] = np.nan_to_num((yraw[r] - ym) / ys)
        B["ROW"][i, :L] = r
    B["site"] = V["site"][rows][start]
    return B, stats


def rows_of(B, val):
    m = B["VM"] > 0
    return B["ROW"][m], val[m]


def main(seed=0):
    t0 = time.time()
    df = load()
    V, F = arrays(df)
    role = splits(V)
    emb, avail, l = ssl_train(F, role == "train", V["site"], seed=seed)
    print("ssl", l, time.time() - t0, flush=True)
    tr_rows = np.where(role == "train")[0]
    _, stats = pack(V, emb, avail, F, tr_rows)
    Bs = {}
    for nm in ("train", "val", "cal", "test"):
        Bs[nm] = {s: pack(V, emb, avail, F, np.where((role == nm) & (V["site"] == s))[0], stats)[0] for s in INT_SITES}
    Bpool = pack(V, emb, avail, F, np.where(role == "pool")[0], stats)[0]
    Bext = pack(V, emb, avail, F, np.where(role == "ext")[0], stats)[0]
    out = {}
    # twin (federated + adapters), twin (central)
    p, ads, _ = T.train_federated(Bs["train"], Bs["val"], seed=seed, rounds=ROUNDS, local_steps=8, verbose=False)
    ext_ad = T.fit_adapter_labelfree(p, Bpool)
    out["twin_fed"] = [rows_of(Bs["test"][s], T.infer(p, ads[s], Bs["test"][s])["risk"]) for s in INT_SITES]
    out["twin_fed|ext"] = [rows_of(Bext, T.infer(p, ext_ad, Bext)["risk"])]
    out["twin_fed|ext0"] = [rows_of(Bext, T.infer(p, T.init_adapter(), Bext)["risk"])]
    out["twin_fed|cal"] = [rows_of(Bs["cal"][s], T.infer(p, ads[s], Bs["cal"][s])["risk"]) for s in INT_SITES]
    pc, adc, _ = T.train_federated(Bs["train"], Bs["val"], seed=seed, rounds=ROUNDS, local_steps=8, central=True, verbose=False)
    out["twin_central"] = [rows_of(Bs["test"][s], T.infer(pc, adc[0], Bs["test"][s])["risk"]) for s in INT_SITES]
    out["twin_central|ext"] = [rows_of(Bext, T.infer(pc, T.fit_adapter_labelfree(pc, Bpool, ad0=adc[0]), Bext)["risk"])]
    print("twin", time.time() - t0, flush=True)
    for name, mode in (("gru_fedavg", "fedavg"), ("gru_central", "central"), ("gru_ditto", "ditto")):
        m = train_seq(Bs["train"], Bs["val"], mode=mode, seed=seed, rounds=ROUNDS)
        out[name] = [rows_of(Bs["test"][s], seq_risk(m["local"][s] if mode == "ditto" else m["global_"], Bs["test"][s]))
                     for s in INT_SITES]
        out[name + "|ext"] = [rows_of(Bext, seq_risk(m["global_"], Bext))]
        out[name + "|cal"] = [rows_of(Bs["cal"][s], seq_risk(m["local"][s] if mode == "ditto" else m["global_"], Bs["cal"][s]))
                              for s in INT_SITES]
    print("gru", time.time() - t0, flush=True)
    if seed == 0:
        # classical Cox / RSF on the exam's own measurements (pooled development data)
        cols = sum([GROUPS[g] for g in GN], [])
        Xr = df[cols].values.astype(float)
        xs = np.stack([(V["age"] - 60) / 10, V["male"], V["smoke"], V["diab"], V["pmi"], V["htn"]], 1)
        tr = role == "train"
        med = np.nanmedian(Xr[tr], 0)
        Xi = np.where(np.isnan(Xr), med, Xr)
        Xs = np.concatenate([(Xi - Xi[tr].mean(0)) / (Xi[tr].std(0) + 1e-6), np.isnan(Xr[:, 6:7]).astype(float), xs], 1)
        te = np.where((role == "test"))[0]
        ex = np.where(role == "ext")[0]
        for kind in ("cox", "rsf"):
            m = fit_classical(Xs[tr], V["T"][tr], V["E"][tr], kind)
            out[kind] = [(te, classical_risk(m, Xs[te], kind))]
            out[kind + "|ext"] = [(ex, classical_risk(m, Xs[ex], kind))]
            cl = np.where(role == "cal")[0]
            out[kind + "|cal"] = [(cl, classical_risk(m, Xs[cl], kind))]
    res = {k: (np.concatenate([a for a, _ in v]), np.concatenate([b for _, b in v])) for k, v in out.items()}
    pickle.dump(dict(pred=res, V={k: V[k] for k in ("pid", "site", "T", "E", "t", "vidx")}, role=role),
                open(f"{RUNS}/fram_s{seed}.pkl", "wb"))
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
