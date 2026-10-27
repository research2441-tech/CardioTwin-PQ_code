"""Federated self-supervised multimodal encoder (masked temporal reconstruction + cross-modal InfoNCE).

Usage: python ssl_encoder.py [variant] [seed]
variant in {full, recon_only, contrast_only, supervised}
Writes runs/emb_<variant>_s<seed>.npz with 24-d embeddings for every visit.
"""
import sys, time, json
import numpy as np
import jax, jax.numpy as jnp
import optax
from common import load_sim, make_splits, visit_table, static_cov, interval_labels, RUNS, DATA, INTERNAL, N_INT

D_EMB = 24
PATCH = 20
NPATCH = 20


def detrend(x, w=100):
    k = np.ones(w) / w
    base = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, x)
    return x - base


def prep(V, ecg, ppg, train_idx):
    e = detrend(ecg)
    p = ppg.copy()
    has_e, has_p, has_l = V["has_e"], V["has_p"], V["has_l"]
    tr_e = train_idx[has_e[train_idx] == 1]
    tr_p = train_idx[has_p[train_idx] == 1]
    es, ps = e[tr_e].std(), p[tr_p].std()
    pm = p[tr_p].mean()
    e = (e / es).astype(np.float32)
    p = ((p - pm) / ps).astype(np.float32)
    p[has_p == 0] = 0
    vit = V["vit"]
    vm, vs = vit[train_idx].mean((0, 1)), vit[train_idx].std((0, 1))
    vit = ((vit - vm) / vs).astype(np.float32)
    lab = V["lab"]
    tl = train_idx[has_l[train_idx] == 1]
    lm, ls = lab[tl].mean(0), lab[tl].std(0)
    lab = ((lab - lm) / ls).astype(np.float32)
    lab[has_l == 0] = 0
    return e, p, vit, lab


# ------------------------------------------------------------------ network
def conv(x, w, b, stride):
    # x: (B, L, C) ; w: (K, Cin, Cout)
    y = jax.lax.conv_general_dilated(x, w, (stride,), "SAME", dimension_numbers=("NWC", "WIO", "NWC"))
    return y + b


def init_params(key):
    ks = iter(jax.random.split(key, 64))

    def dense(i, o):
        return dict(w=jax.random.normal(next(ks), (i, o)) * np.sqrt(2.0 / i), b=jnp.zeros(o))

    def cnn():
        return dict(c1=dict(w=jax.random.normal(next(ks), (7, 1, 16)) * np.sqrt(2 / 7), b=jnp.zeros(16)),
                    c2=dict(w=jax.random.normal(next(ks), (5, 16, 32)) * np.sqrt(2 / 80), b=jnp.zeros(32)),
                    c3=dict(w=jax.random.normal(next(ks), (5, 32, 32)) * np.sqrt(2 / 160), b=jnp.zeros(32)),
                    out=dense(64, D_EMB))

    enc = dict(ecg=cnn(), ppg=cnn(),
               vit=dict(h=dense(60 + 15, 64), out=dense(64, D_EMB)),
               lab=dict(h=dense(12, 48), out=dense(48, D_EMB)))
    dec = dict(ecg=dense(D_EMB, 100), ppg=dense(D_EMB, 100), vit=dense(D_EMB, 60), lab=dense(D_EMB, 6))
    proj = {m: dense(D_EMB, 32) for m in ("ecg", "ppg", "vit", "lab")}
    sup = dict(h=dense(4 * D_EMB + 4 + 7, 64), out=dense(64, N_INT))
    return dict(enc=enc, dec=dec, proj=proj, sup=sup)


def lin(p, x):
    return x @ p["w"] + p["b"]


def enc_wave(p, x):
    h = jax.nn.gelu(conv(x[..., None], p["c1"]["w"], p["c1"]["b"], 2))
    h = jax.nn.gelu(conv(h, p["c2"]["w"], p["c2"]["b"], 2))
    h = jax.nn.gelu(conv(h, p["c3"]["w"], p["c3"]["b"], 2))
    h = jnp.concatenate([h.mean(1), h.max(1)], -1)
    return lin(p["out"], h)


def enc_vit(p, v):
    tt = jnp.arange(12) - 5.5
    slope = (v * tt[None, :, None]).sum(1) / (tt ** 2).sum()
    f = jnp.concatenate([v.reshape(v.shape[0], -1), v.mean(1), jnp.sqrt(v.var(1) + 1e-4), slope], -1)
    return lin(p["out"], jax.nn.gelu(lin(p["h"], f)))


def enc_lab(p, l, m):
    return lin(p["out"], jax.nn.gelu(lin(p["h"], jnp.concatenate([l * m, m], -1))))


def encode(params, e, pp, v, l, lm):
    P = params["enc"]
    return dict(ecg=enc_wave(P["ecg"], e), ppg=enc_wave(P["ppg"], pp), vit=enc_vit(P["vit"], v),
                lab=enc_lab(P["lab"], l, lm))


def infonce(a, b, mask, tau=0.1):
    a = a * jax.lax.rsqrt((a ** 2).sum(-1, keepdims=True) + 1e-6)
    b = b * jax.lax.rsqrt((b ** 2).sum(-1, keepdims=True) + 1e-6)
    s = a @ b.T / tau
    s = jnp.where(mask[None, :] > 0, s, -1e4)
    l1 = -(jnp.diag(jax.nn.log_softmax(s, 1)))
    l2 = -(jnp.diag(jax.nn.log_softmax(s.T, 1)))
    return ((l1 + l2) / 2 * mask).sum() / (mask.sum() + 1e-6)


def ssl_loss(params, batch, key, w_rec, w_con):
    e, pp, v, l, me, mp, ml = batch
    k1, k2, k3, k4 = jax.random.split(key, 4)
    B = e.shape[0]
    # masked temporal reconstruction: 40% of 20-sample patches zeroed
    pm_e = (jax.random.uniform(k1, (B, NPATCH)) < 0.4).astype(jnp.float32)
    pm_p = (jax.random.uniform(k2, (B, NPATCH)) < 0.4).astype(jnp.float32)
    e_in = e * (1 - jnp.repeat(pm_e, PATCH, 1))
    p_in = pp * (1 - jnp.repeat(pm_p, PATCH, 1))
    vm = (jax.random.uniform(k3, v.shape[:2]) < 0.3).astype(jnp.float32)
    v_in = v * (1 - vm[..., None])
    lmask_drop = (jax.random.uniform(k4, l.shape) < 0.3).astype(jnp.float32)
    lmo = jnp.broadcast_to(ml[:, None], l.shape)
    l_vis = lmo * (1 - lmask_drop)
    h = encode(params, e_in, p_in, v_in, l, l_vis)
    D = params["dec"]
    tgt_e = e.reshape(B, 100, 4).mean(-1)
    tgt_p = pp.reshape(B, 100, 4).mean(-1)
    w_e = jnp.repeat(pm_e, 5, 1)
    w_p = jnp.repeat(pm_p, 5, 1)
    rec = (((lin(D["ecg"], h["ecg"]) - tgt_e) ** 2 * w_e).sum(1) / (w_e.sum(1) + 1e-6) * me).sum() / (me.sum() + 1e-6)
    rec += (((lin(D["ppg"], h["ppg"]) - tgt_p) ** 2 * w_p).sum(1) / (w_p.sum(1) + 1e-6) * mp).sum() / (mp.sum() + 1e-6)
    vr = (lin(D["vit"], h["vit"]).reshape(B, 12, 5) - v) ** 2
    rec += (vr * vm[..., None]).sum() / (vm.sum() * 5 + 1e-6)
    lr = (lin(D["lab"], h["lab"]) - l) ** 2 * lmo * lmask_drop
    rec += lr.sum() / ((lmo * lmask_drop).sum() + 1e-6)
    # cross-modal contrastive alignment on clean views
    hc = encode(params, e, pp, v, l, lmo)
    z = {m: lin(params["proj"][m], hc[m]) for m in hc}
    av = dict(ecg=me, ppg=mp, vit=jnp.ones_like(me), lab=ml)
    pairs = [("ecg", "ppg"), ("ecg", "vit"), ("ppg", "vit"), ("ecg", "lab"), ("ppg", "lab"), ("vit", "lab")]
    con = sum(infonce(z[a], z[b], av[a] * av[b]) for a, b in pairs) / len(pairs)
    return w_rec * rec + w_con * con, (rec, con)


def sup_loss(params, batch, key):
    """Supervised alternative: encoders trained only through a discrete-time hazard head."""
    e, pp, v, l, me, mp, ml, xs, k, ev = batch
    lmo = jnp.broadcast_to(ml[:, None], l.shape)
    h = encode(params, e, pp, v, l, lmo)
    f = jnp.concatenate([h["ecg"] * me[:, None], h["ppg"] * mp[:, None], h["vit"], h["lab"] * ml[:, None],
                         me[:, None], mp[:, None], ml[:, None], jnp.ones_like(me)[:, None], xs], -1)
    logit = lin(params["sup"]["out"], jax.nn.gelu(lin(params["sup"]["h"], f)))
    logp_ev = jax.nn.log_sigmoid(logit)
    logp_s = jax.nn.log_sigmoid(-logit)
    j = jnp.arange(N_INT)[None, :]
    surv = (logp_s * (j < k[:, None])).sum(1)
    evt = (logp_ev * (j == k[:, None])).sum(1) * ev
    return -(surv + evt).mean(), (0.0, 0.0)


def main(variant="full", seed=0, rounds=40, local_steps=15, bs=192):
    t0 = time.time()
    pats, visits = load_sim()
    V = visit_table(pats, visits)
    role = make_splits(pats)
    r = np.array([role[p] for p in V["pid"]])
    tr = np.where((r == "train") & (V["cal"] < 3.5))[0]
    ecg = np.load(f"{DATA}/ecg.npy")
    ppg = np.load(f"{DATA}/ppg.npy")
    e, p, vit, lab = prep(V, ecg, ppg, tr)
    del ecg, ppg
    me, mp, ml = (V["has_e"].astype(np.float32), V["has_p"].astype(np.float32), V["has_l"].astype(np.float32))
    xs = static_cov(V)
    kk, evv = interval_labels(V["Ttr"], V["Etr"])
    key = jax.random.PRNGKey(seed)
    params = init_params(key)
    opt = optax.adam(1e-3)
    w_rec, w_con = dict(full=(1.0, 0.5), recon_only=(1.0, 0.0), contrast_only=(0.0, 1.0), supervised=(0, 0))[variant]

    if variant == "supervised":
        def lossf(prm, b, k_):
            return sup_loss(prm, b, k_)
    else:
        def lossf(prm, b, k_):
            return ssl_loss(prm, b, k_, w_rec, w_con)

    @jax.jit
    def step(prm, st, b, k_):
        (lv, aux), g = jax.value_and_grad(lossf, has_aux=True)(prm, b, k_)
        upd, st = opt.update(g, st, prm)
        return optax.apply_updates(prm, upd), st, lv, aux

    site_idx = {s: tr[V["site"][tr] == s] for s in INTERNAL}
    sizes = np.array([len(site_idx[s]) for s in INTERNAL], float)
    wts = sizes / sizes.sum()
    rng = np.random.default_rng(seed)
    hist = []
    for rd in range(rounds):
        locals_ = []
        for s in INTERNAL:
            prm = params
            st = opt.init(prm)
            for _ in range(local_steps):
                b = rng.choice(site_idx[s], bs, replace=False)
                key, sk = jax.random.split(key)
                if variant == "supervised":
                    batch = (e[b], p[b], vit[b], lab[b], me[b], mp[b], ml[b], xs[b], kk[b], evv[b])
                else:
                    batch = (e[b], p[b], vit[b], lab[b], me[b], mp[b], ml[b])
                prm, st, lv, aux = step(prm, st, batch, sk)
            locals_.append(prm)
        params = jax.tree_util.tree_map(lambda *xs_: sum(w * x for w, x in zip(wts, xs_)), *locals_)
        hist.append(float(lv))
        if rd % 10 == 0 or rd == rounds - 1:
            print(variant, rd, float(lv), [float(a) for a in aux], round(time.time() - t0, 1), flush=True)

    @jax.jit
    def emb(prm, a, b_, c, d_, m_):
        return encode(prm, a, b_, c, d_, jnp.broadcast_to(m_[:, None], d_.shape))

    out = {m: [] for m in ("ecg", "ppg", "vit", "lab")}
    for i in range(0, len(V["pid"]), 2048):
        sl = slice(i, i + 2048)
        h = emb(params, e[sl], p[sl], vit[sl], lab[sl], ml[sl])
        for m in out:
            out[m].append(np.asarray(h[m]))
    out = {m: np.concatenate(v) for m, v in out.items()}
    np.savez(f"{RUNS}/emb_{variant}_s{seed}.npz", **out, loss=np.array(hist))
    print("done", variant, seed, time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full", int(sys.argv[2]) if len(sys.argv) > 2 else 0,
         rounds=int(sys.argv[3]) if len(sys.argv) > 3 else 40)
