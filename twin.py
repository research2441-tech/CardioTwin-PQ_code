"""CardioTwin-PQ patient-specific state-space twin with federated training and site adapters.

Continuous-time linear-Gaussian latent physiology, Van Loan discretisation for irregular visit gaps,
information-form Kalman update that skips unavailable modalities, moment-matched hazard read-out
for discrete-time multi-horizon risk, and label-free acquisition adapters per institution.
"""
import time
import numpy as np
import jax, jax.numpy as jnp
import optax
from jax.scipy.linalg import expm
from common import (load_sim, make_splits, visit_table, static_cov, interval_labels, N_INT, DELTA,
                    INTERNAL, EXTERNAL, RUNS, MODS)

K = 6            # latent state dimension
DE = 24          # embedding dimension per modality
NM = 4           # modalities
VMAX = 10
GRID = 0.025
NG = 200         # transition grid up to 5 years
TAU_IDX = np.array([int(round((j + 0.5) * DELTA / GRID)) for j in range(N_INT)])
NX = 7
NY = 3           # clinical read-outs: mean SBP, log NT-proBNP, mean HR
MOMENT = True    # ablation switch: second-order (moment-matched) hazard term
MEMORY = True    # ablation switch: carry the filtered state across visits


# --------------------------------------------------------------------------- data
def load_embeddings(variant="full", seed=0):
    d = np.load(f"{RUNS}/emb_{variant}_s{seed}.npz")
    return np.stack([d[m] for m in MODS], 1)  # (n_visits, 4, 24)


def pack(V, emb, idx_visits, label="train", emb_stats=None):
    """Pack visit rows (already filtered) into padded patient tensors."""
    pids = V["pid"][idx_visits]
    order = np.lexsort((V["vidx"][idx_visits], pids))
    iv = idx_visits[order]
    up, start = np.unique(V["pid"][iv], return_index=True)
    n = len(up)
    E = np.zeros((n, VMAX, NM, DE), np.float32)
    M = np.zeros((n, VMAX, NM), np.float32)
    X = np.zeros((n, VMAX, NX), np.float32)
    DT = np.zeros((n, VMAX), np.float32)
    VM = np.zeros((n, VMAX), np.float32)
    KK = np.zeros((n, VMAX), np.int32)
    EV = np.zeros((n, VMAX), np.int32)
    Y = np.zeros((n, VMAX, NY), np.float32)
    ROW = -np.ones((n, VMAX), np.int64)
    xs = static_cov(V)
    if emb_stats is None:
        mu = np.zeros((NM, DE)); sd = np.ones((NM, DE))
        avail = np.stack([V["has_e"], V["has_p"], np.ones_like(V["has_e"]), V["has_l"]], 1)[iv]
        for m in range(NM):
            a = emb[iv][avail[:, m] == 1, m]
            mu[m], sd[m] = a.mean(0), a.std(0) + 1e-6
        ymu = np.array([V["vit"][iv][:, :, 0].mean(), np.log(1) + V["lab"][iv][:, 4].mean(), V["vit"][iv][:, :, 2].mean()])
        ysd = np.array([V["vit"][iv][:, :, 0].mean(1).std(), V["lab"][iv][:, 4].std(), V["vit"][iv][:, :, 2].mean(1).std()])
        emb_stats = (mu, sd, ymu, ysd)
    mu, sd, ymu, ysd = emb_stats
    T, Ev = (V["Ttr"], V["Etr"]) if label == "train" else (V["T"], V["E"])
    kk, ee = interval_labels(T, Ev)
    ends = list(start[1:]) + [len(iv)]
    for i, (s0, s1) in enumerate(zip(start, ends)):
        rows = iv[s0:s1][:VMAX]
        L = len(rows)
        E[i, :L] = (emb[rows] - mu) / sd
        M[i, :L] = np.stack([V["has_e"][rows], V["has_p"][rows], np.ones(L), V["has_l"][rows]], 1)
        E[i, :L] *= M[i, :L, :, None]
        X[i, :L] = xs[rows]
        t = V["t"][rows]
        DT[i, 1:L] = np.diff(t)
        VM[i, :L] = 1
        KK[i, :L] = kk[rows]
        EV[i, :L] = ee[rows]
        Y[i, :L] = (np.stack([V["vit"][rows][:, :, 0].mean(1), V["lab"][rows][:, 4], V["vit"][rows][:, :, 2].mean(1)], 1)
                    - ymu) / ysd
        ROW[i, :L] = rows
    site = V["site"][iv][start]
    return dict(E=E, M=M, X=X, DT=DT, VM=VM, K=KK, EV=EV, Y=Y, ROW=ROW, site=site, pid=up), emb_stats


# --------------------------------------------------------------------------- parameters
def init_shared(key):
    ks = jax.random.split(key, 12)
    return dict(
        a_diag=jnp.full(K, -1.5), a_off=0.01 * jax.random.normal(ks[0], (K, K)),
        q_log=jnp.full(K, -2.0),
        Wu=0.05 * jax.random.normal(ks[1], (NX, K)), bu=jnp.zeros(K),
        W0=0.3 * jax.random.normal(ks[2], (NX, K)), b0=jnp.zeros(K), s0_log=jnp.zeros(K),
        H=0.3 * jax.random.normal(ks[3], (NM, DE, K)), b=jnp.zeros((NM, DE)), r_log=jnp.zeros((NM, DE)),
        theta=jnp.full(N_INT, -3.5), beta=0.1 * jax.random.normal(ks[4], (K,)), gamma=jnp.zeros(NX),
        Hy=0.3 * jax.random.normal(ks[5], (NY, K)), by=jnp.zeros(NY), ry_log=jnp.full(NY, -1.0),
        Wy=jnp.zeros((NX, NY)))


def init_adapter():
    return dict(a=jnp.zeros((NM, DE)), logd=jnp.zeros((NM, DE)))


def A_mat(p):
    off = p["a_off"] * (1 - jnp.eye(K))
    return -jnp.diag(jax.nn.softplus(p["a_diag"]) + 0.02) + off


def transition_grid(p):
    """Van Loan discretisation on a grid of gaps: F(d), G(d) = int_0^d e^{As} ds, Q(d)."""
    A = A_mat(p)
    Qc = jnp.diag(jax.nn.softplus(p["q_log"]))
    d = jnp.arange(NG) * GRID
    Z = jnp.zeros((K, K))
    M1 = jnp.block([[-A, Qc], [Z, A.T]])
    M2 = jnp.block([[A, jnp.eye(K)], [Z, Z]])
    C1 = jax.vmap(lambda t: expm(M1 * t))(d)
    C2 = jax.vmap(lambda t: expm(M2 * t))(d)
    F = jnp.transpose(C1[:, K:, K:], (0, 2, 1))
    Q = F @ C1[:, :K, K:]
    Q = 0.5 * (Q + jnp.transpose(Q, (0, 2, 1)))
    G = C2[:, :K, K:]
    return F, G, Q


def gap_index(dt):
    return jnp.clip(jnp.round(dt / GRID).astype(jnp.int32), 0, NG - 1)


# --------------------------------------------------------------------------- filter
def filter_patient(p, ad, F, G, Q, E, M, X, DT, VM, Y, with_mod=False):
    """Run the twin over one patient's visits. Returns per-visit outputs."""
    H = p["H"]
    Rinv = jnp.exp(-p["r_log"])                 # (NM, DE)
    beta = p["beta"]
    Ftau, Gtau, Qtau = F[TAU_IDX], G[TAU_IDX], Q[TAU_IDX]

    def step(carry, inp):
        mu, S, first = carry
        e, m, x, dt, vm, ynext, dtn, a_v, ld_v = inp
        D = jnp.exp(ld_v)
        Ht = D[:, :, None] * H                  # (NM, DE, K) acquisition-adapted loading
        off = D * p["b"] + a_v                  # (NM, DE)
        c = x @ p["Wu"] + p["bu"]
        gi = gap_index(dt)
        mu_t = F[gi] @ mu + G[gi] @ c
        S_t = F[gi] @ S @ F[gi].T + Q[gi]
        mu0 = x @ p["W0"] + p["b0"]
        S0 = jnp.diag(jnp.exp(p["s0_log"]))
        mu_pr = jnp.where(first | (not MEMORY), mu0, mu_t)
        S_pr = jnp.where(first | (not MEMORY), S0, S_t)
        nu = e - (jnp.einsum("mdk,k->md", Ht, mu_pr) + off)           # innovation
        wm = m[:, None] * Rinv
        Spr_inv = jnp.linalg.inv(S_pr + 1e-6 * jnp.eye(K))
        J = Spr_inv + jnp.einsum("mdk,md,mdl->kl", Ht, wm, Ht)
        S_po = jnp.linalg.inv(J)
        S_po = 0.5 * (S_po + S_po.T)
        g = jnp.einsum("mdk,md->k", Ht, wm * nu)
        mu_po = mu_pr + S_po @ g
        nis = (wm * nu ** 2).sum() - g @ S_po @ g
        nobs = DE * m.sum()
        ldS = (m[:, None] * p["r_log"]).sum() + jnp.linalg.slogdet(S_pr)[1] - jnp.linalg.slogdet(S_po)[1]
        obs_nll = 0.5 * (nis + ldS + nobs * jnp.log(2 * jnp.pi))
        # per-modality marginal NIS (drift localisation)
        def mod_nis(k_):
            Jk = Spr_inv + Ht[k_].T @ (Rinv[k_][:, None] * Ht[k_])
            gk = Ht[k_].T @ (Rinv[k_] * nu[k_])
            return (Rinv[k_] * nu[k_] ** 2).sum() - gk @ jnp.linalg.solve(Jk, gk)
        nis_m = jax.vmap(mod_nis)(jnp.arange(NM)) * m if with_mod else jnp.zeros(NM)
        # hazard read-out with moment matching over the forecast state
        mu_j = jnp.einsum("jkl,l->jk", Ftau, mu_po) + jnp.einsum("jkl,l->jk", Gtau, c)
        S_j = jnp.einsum("jkl,lm,jnm->jkn", Ftau, S_po, Ftau) + Qtau
        eta = p["theta"] + mu_j @ beta + (0.5 if MOMENT else 0.0) * jnp.einsum("k,jkl,l->j", beta, S_j, beta) + x @ p["gamma"]
        lam = jnp.exp(jnp.clip(eta, -12, 4)) * DELTA
        # next-visit clinical forecast
        gn = gap_index(dtn)
        mu_n = F[gn] @ mu_po + G[gn] @ c
        S_n = F[gn] @ S_po @ F[gn].T + Q[gn]
        ymu = p["Hy"] @ mu_n + p["by"] + x @ p["Wy"]
        yvar = jnp.einsum("yk,kl,yl->y", p["Hy"], S_n, p["Hy"]) + jnp.exp(p["ry_log"])
        mu_new = jnp.where(vm > 0, mu_po, mu)
        S_new = jnp.where(vm > 0, S_po, S)
        first_new = jnp.where(vm > 0, False, first)
        out = dict(lam=lam, obs_nll=obs_nll, nis=nis, nobs=nobs, nis_m=nis_m, ymu=ymu, yvar=yvar,
                   mu=mu_po, sd=jnp.sqrt(jnp.diag(S_po)))
        return (mu_new, S_new, first_new), out

    DTn = jnp.concatenate([DT[1:], jnp.zeros(1)])
    Yn = jnp.concatenate([Y[1:], jnp.zeros((1, NY))], 0)
    init = (jnp.zeros(K), jnp.eye(K), True)
    _, out = jax.lax.scan(step, init, (E, M, X, DT, VM, Yn, DTn, ad["a"], ad["logd"]))
    return out


def batch_outputs(p, ads, site_local, B, with_mod=False):
    """ads: stacked adapters indexed by site_local (n,), or per-visit adapters if B has 'ADa'."""
    F, G, Q = transition_grid(p)
    if "ADa" in B:
        adb = dict(a=B["ADa"], logd=B["ADd"])
    else:
        adb = jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a[site_local][:, None], (a[site_local].shape[0], VMAX) + a.shape[1:]), ads)
    return jax.vmap(lambda ad, E, M, X, DT, VM, Y: filter_patient(p, ad, F, G, Q, E, M, X, DT, VM, Y, with_mod))(
        adb, B["E"], B["M"], B["X"], B["DT"], B["VM"], B["Y"])


def surv_nll(lam, K_, EV, VM):
    j = jnp.arange(N_INT)
    before = (j[None, None, :] < K_[..., None]).astype(jnp.float32)
    at = (j[None, None, :] == K_[..., None]).astype(jnp.float32)
    log_surv = -(lam * before).sum(-1)
    log_ev = (jnp.log(-jnp.expm1(-lam) + 1e-12) * at).sum(-1)
    nll = -(log_surv + EV * log_ev)
    return (nll * VM).sum() / VM.sum()


def loss_fn(p, ads, site_local, B, w_obs=0.5, w_y=0.3, use_obs=True):
    o = batch_outputs(p, ads, site_local, B)
    ls = surv_nll(o["lam"], B["K"], B["EV"], B["VM"])
    lo = (o["obs_nll"] * B["VM"]).sum() / ((o["nobs"] * B["VM"]).sum() + 1e-6)
    # next-visit read-out likelihood (only where a next visit exists)
    has_next = jnp.concatenate([B["VM"][:, 1:], jnp.zeros((B["VM"].shape[0], 1))], 1)
    Yn = jnp.concatenate([B["Y"][:, 1:], jnp.zeros_like(B["Y"][:, :1])], 1)
    ly = 0.5 * (((Yn - o["ymu"]) ** 2 / o["yvar"] + jnp.log(o["yvar"])).sum(-1) * has_next).sum() / (has_next.sum() + 1e-6)
    tot = ls + (w_obs * lo if use_obs else 0.0) + w_y * ly
    return tot, (ls, lo, ly)


def risks_from_lam(lam):
    """cumulative risk at 1, 2, 3 years from per-interval integrated hazards."""
    cum = jnp.cumsum(lam, -1)
    return 1 - jnp.exp(-cum[..., [3, 7, 11]])


# --------------------------------------------------------------------------- training
def subset(B, idx):
    return {k: v[idx] for k, v in B.items()}


def train_federated(Btr_by_site, Bval_by_site, seed=0, rounds=60, local_steps=8, lr=3e-3, adapters=True,
                    prox_mu=0.0, use_obs=True, ditto=False, verbose=True, central=False, agg_fn=None):
    key = jax.random.PRNGKey(seed)
    p = init_shared(key)
    sites = sorted(Btr_by_site)
    ads = {s: init_adapter() for s in sites}
    opt = optax.adam(lr)
    n_s = np.array([Btr_by_site[s]["E"].shape[0] for s in sites], float)
    wts = n_s / n_s.sum()

    def lf(pp, ad, B, pg):
        a1 = jax.tree_util.tree_map(lambda a: a[None], ad)
        l, aux = loss_fn(pp, a1, jnp.zeros(B["E"].shape[0], jnp.int32), B, use_obs=use_obs)
        if prox_mu > 0:
            l = l + 0.5 * prox_mu * sum(((x - y) ** 2).sum() for x, y in
                                        zip(jax.tree_util.tree_leaves(pp), jax.tree_util.tree_leaves(pg)))
        return l, aux

    @jax.jit
    def local_step(pp, ad, st_p, st_a, B, pg):
        (l, aux), (gp, ga) = jax.value_and_grad(lf, argnums=(0, 1), has_aux=True)(pp, ad, B, pg)
        up, st_p = opt.update(gp, st_p, pp)
        pp = optax.apply_updates(pp, up)
        if adapters:
            ua, st_a = opt.update(ga, st_a, ad)
            ad = optax.apply_updates(ad, ua)
        return pp, ad, st_p, st_a, l, aux

    if central:
        Ball = {k: np.concatenate([Btr_by_site[s][k] for s in sites]) for k in Btr_by_site[sites[0]]}
        Btr_by_site = {0: Ball}
        ads = {0: init_adapter()}
        sites = [0]
        wts = np.array([1.0])
    st_a = {s: opt.init(ads[s]) for s in sites}
    vkeys = ("E", "M", "X", "DT", "VM", "Y", "K", "EV")
    Bval_j = {s: {k: jnp.asarray(Bval_by_site[s][k]) for k in vkeys} for s in Bval_by_site}
    Btr_by_site = {s: {k: jnp.asarray(Btr_by_site[s][k]) for k in vkeys} for s in Btr_by_site}

    @jax.jit
    def val_nll(pp, ad, B):
        a1 = jax.tree_util.tree_map(lambda a: a[None], ad)
        o = batch_outputs(pp, a1, jnp.zeros(B["E"].shape[0], jnp.int32), B)
        return surv_nll(o["lam"], B["K"], B["EV"], B["VM"])
    best, best_state, hist = np.inf, None, []
    t0 = time.time()
    for rd in range(rounds):
        loc = []
        for s in sites:
            pp, st_p = p, opt.init(p)
            for _ in range(local_steps):
                pp, ads[s], st_p, st_a[s], l, aux = local_step(pp, ads[s], st_p, st_a[s], Btr_by_site[s], p)
            loc.append(pp)
        if agg_fn is None:
            p = jax.tree_util.tree_map(lambda *xs: sum(w * x for w, x in zip(wts, xs)), *loc)
        else:
            p = agg_fn(loc, wts, rd)
        # validation (survival NLL) for model selection
        vl = 0.0
        for s in sorted(Bval_by_site):
            ad = ads[s] if (s in ads and adapters) else (ads[0] if central else init_adapter())
            vl += float(val_nll(p, ad, Bval_j[s])) * Bval_by_site[s]["VM"].sum()
        hist.append(vl)
        if vl < best:
            best, best_state = vl, (p, {s: ads[s] for s in ads})
        if verbose and (rd % 10 == 0 or rd == rounds - 1):
            print(f"round {rd} train {float(l):.4f} aux {[round(float(a), 3) for a in aux]} val {vl:.1f} "
                  f"{time.time() - t0:.0f}s", flush=True)
    return best_state[0], best_state[1], hist


def fit_adapter_labelfree(p, B, steps=150, lr=2e-2, ad0=None):
    """Re-estimate one institution's acquisition adapter by maximising the innovation likelihood only."""
    ad = ad0 if ad0 is not None else init_adapter()
    opt = optax.adam(lr)
    st = opt.init(ad)

    def lf(a):
        a1 = jax.tree_util.tree_map(lambda x: x[None], a)
        o = batch_outputs(p, a1, jnp.zeros(B["E"].shape[0], jnp.int32), B)
        return (o["obs_nll"] * B["VM"]).sum() / ((o["nobs"] * B["VM"]).sum() + 1e-6) + \
            1e-3 * sum((x ** 2).sum() for x in jax.tree_util.tree_leaves(a))

    @jax.jit
    def stp(a, st):
        l, g = jax.value_and_grad(lf)(a)
        u, st = opt.update(g, st, a)
        return optax.apply_updates(a, u), st, l

    for _ in range(steps):
        ad, st, l = stp(ad, st)
    return ad


def infer(p, ad, B, chunk=1024):
    outs = []
    a1 = jax.tree_util.tree_map(lambda a: a[None], ad)
    f = jax.jit(lambda Bc: batch_outputs(p, a1, jnp.zeros(Bc["E"].shape[0], jnp.int32), Bc, True))
    n = B["E"].shape[0]
    for i in range(0, n, chunk):
        Bc = {k: v[i:i + chunk] for k, v in B.items() if k in ("E", "M", "X", "DT", "VM", "Y", "ADa", "ADd")}
        o = f(Bc)
        outs.append({k: np.asarray(v) for k, v in o.items()})
    o = {k: np.concatenate([x[k] for x in outs]) for k in outs[0]}
    o["risk"] = np.asarray(risks_from_lam(o["lam"]))
    return o


def flatten(o, B):
    """Map padded outputs back to visit rows."""
    m = B["VM"] > 0
    rows = B["ROW"][m]
    res = {k: v[m] for k, v in o.items()}
    res["row"] = rows
    return res
