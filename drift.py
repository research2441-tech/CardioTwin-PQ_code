"""Online deployment study: innovation-CUSUM drift detection, label-free adapter re-estimation,
conformal risk sets with delayed labels, innovation-gated deferral, and next-visit trajectory intervals."""
import pickle, sys, time
import numpy as np
import jax, jax.numpy as jnp
import optax
import twin as T
from dataset import build
from common import RUNS, INTERNAL, EXTERNAL, evaluate
from conformal import (cusum, nis_z, run_stream, summarise, label, label_known, scores, conf_quantile, ALPHA)
from simulator import DRIFT

W_DET = 0.25     # years of post-alarm data gathered before the first re-estimation
REFITS = (0.25, 0.75)


def visit_adapters(B, versions):
    """versions: list of (t_valid_from, adapter). Each visit uses the latest version valid at its calendar time."""
    cal = B["CAL"]
    n = cal.shape[0]
    a = np.zeros((n, T.VMAX, T.NM, T.DE), np.float32)
    d = np.zeros_like(a)
    for tv, ad in sorted(versions, key=lambda x: x[0]):
        m = cal >= tv
        a[m] = np.asarray(ad["a"])
        d[m] = np.asarray(ad["logd"])
    return a, d


def add_cal(B, V):
    rows = B["ROW"]
    c = np.where(rows >= 0, V["cal"][np.maximum(rows, 0)], 1e9)
    B["CAL"] = c
    return B


def outputs(p, B, versions):
    a, d = visit_adapters(B, versions)
    Bv = dict(B)
    Bv["ADa"], Bv["ADd"] = a, d
    o = T.infer(p, T.init_adapter(), Bv)
    return T.flatten(o, B)


def refit(p, B, versions, t_from, t_to, steps=150, lr=2e-2):
    """Fit a new adapter on visits with calendar time in [t_from, t_to) keeping earlier visits on their versions."""
    a_old, d_old = visit_adapters(B, versions)
    new = ((B["CAL"] >= t_from) & (B["CAL"] < t_to) & (B["VM"] > 0)).astype(np.float32)
    keep = new.reshape(new.shape[0], -1).sum(1) > 0
    Bk = {k: jnp.asarray(v[keep]) for k, v in B.items() if k in ("E", "M", "X", "DT", "VM", "Y")}
    nk, ao, do = jnp.asarray(new[keep]), jnp.asarray(a_old[keep]), jnp.asarray(d_old[keep])
    ad = jax.tree_util.tree_map(jnp.asarray, versions[-1][1])
    opt = optax.adam(lr)
    st = opt.init(ad)

    def lf(adp):
        Bv = dict(Bk)
        Bv["ADa"] = jnp.where(nk[..., None, None] > 0, adp["a"][None, None], ao)
        Bv["ADd"] = jnp.where(nk[..., None, None] > 0, adp["logd"][None, None], do)
        o = T.batch_outputs(p, None, None, Bv)
        return (o["obs_nll"] * nk).sum() / ((o["nobs"] * nk).sum() + 1e-6)

    @jax.jit
    def stp(adp, st):
        l, g = jax.value_and_grad(lf)(adp)
        u, st = opt.update(g, st, adp)
        return optax.apply_updates(adp, u), st, l

    for _ in range(steps):
        ad, st, l = stp(ad, st)
    return jax.device_get(ad), int(new.sum())


def monitor_site(p, B, ad0, ref, t_start, enable=True, h=10.0, k=1.0, max_refits=4):
    versions = [(-np.inf, ad0)]
    alarms, n_used = [], []
    t_cur = t_start
    for _ in range(max_refits + 1):
        o = outputs(p, B, versions)
        z = (nis_z(o["nis"], o["nobs"]) - ref[0]) / ref[1]
        cal = B["CAL"][B["VM"] > 0]
        sel = cal >= t_cur
        ordr = np.argsort(cal[sel])
        zi, ci = z[sel][ordr], cal[sel][ordr]
        ia, path = cusum(zi, k=k, h=h)
        if ia < 0 or not enable:
            if ia >= 0:
                alarms.append(ci[ia])
            break
        ta = ci[ia]
        alarms.append(ta)
        for dt in REFITS:
            ad, nu = refit(p, B, versions, ta - W_DET, ta + dt)
            versions.append((ta + dt, ad))
            n_used.append(nu)
        t_cur = ta + REFITS[-1]
    o = outputs(p, B, versions)
    return o, versions, alarms, n_used


def main(seed=0):
    t0 = time.time()
    D = build("full", 0)
    V = D["V"]
    P = pickle.load(open(f"{RUNS}/twin_params_full_e0_s{seed}.pkl", "rb"))
    p = jax.tree_util.tree_map(jnp.asarray, P["p"])
    ads = P["ads"]
    res = dict(sites={}, conf={}, traj={}, series={})
    cal_scores, cal_labels, cal_site = [], [], []
    streams = {}
    for s in list(INTERNAL) + [EXTERNAL]:
        if s == EXTERNAL:
            Bte, ad0 = add_cal(D["ext"], V), P["ext_ad"]
            Bcal = add_cal(D["ext_pool"], V)
        else:
            Bte, ad0 = add_cal(D["te"][s], V), ads[s]
            Bcal = add_cal(D["cal"][s], V)
        oc = outputs(p, Bcal, [(-np.inf, ad0)])
        zc = nis_z(oc["nis"], oc["nobs"])
        ref = (float(zc.mean()), float(zc.std()) + 1e-6)
        ref_rob = (np.median(zc), 1.4826 * np.median(np.abs(zc - np.median(zc))) + 1e-6)
        zgate = np.quantile((zc - ref[0]) / ref[1], 0.99)
        t_start = 2.5 if s != EXTERNAL else 2.5
        o_static, _, al_static, _ = monitor_site(p, Bte, ad0, ref, t_start, enable=False)
        o_drift, versions, alarms, n_used = monitor_site(p, Bte, ad0, ref, t_start, enable=True)
        # CUSUM design sensitivity on the un-adapted stream (first alarm only)
        cal_s = Bte["CAL"][Bte["VM"] > 0]
        z_raw = nis_z(o_static["nis"], o_static["nobs"])
        sel = cal_s >= t_start
        ordr = np.argsort(cal_s[sel])
        grid = {}
        for std_name, rf in (("meanSD", ref), ("medianMAD", ref_rob)):
            for kk in (0.5, 1.0):
                for hh in (10.0, 20.0):
                    ia, _ = cusum((z_raw[sel][ordr] - rf[0]) / rf[1], k=kk, h=hh)
                    grid[f"{std_name}|k{kk}|h{hh}"] = float(cal_s[sel][ordr][ia]) if ia >= 0 else None
        rows = o_static["row"]
        assert (rows == o_drift["row"]).all()
        Tt, Et, cal = V["T"][rows], V["E"][rows], V["cal"][rows]
        dr = DRIFT.get(s)
        t_drift = dr["t"] if dr else None
        # oracle: refit triggered exactly at the true change time
        o_or = None
        if dr:
            vers = [(-np.inf, ad0)]
            for dt in REFITS:
                ad, _ = refit(p, Bte, vers, t_drift, t_drift + dt)
                vers.append((t_drift + dt, ad))
            o_or = outputs(p, Bte, vers)
        info = dict(alarms=alarms, n_refit_visits=n_used, t_drift=t_drift, n=len(rows), grid=grid, ref=ref, ref_rob=ref_rob)
        post = cal >= (t_drift if dr else 4.0)
        for name, o in (("static", o_static), ("drift", o_drift), ("oracle", o_or)):
            if o is None:
                continue
            info[name] = dict(all=evaluate(o["risk"], Tt, Et), post=evaluate(o["risk"][post], Tt[post], Et[post]),
                              pre=evaluate(o["risk"][~post], Tt[~post], Et[~post]))
        res["sites"][s] = info
        # innovation series for figure (quarterly mean standardised NIS, per modality)
        q = np.floor(cal * 4) / 4
        ser = {}
        for name, o in (("static", o_static), ("drift", o_drift)):
            z = (nis_z(o["nis"], o["nobs"]) - ref[0]) / ref[1]
            zm = o["nis_m"] / np.maximum(o["nis_m"] * 0 + 24, 1)
            ser[name] = {float(u): (float(z[q == u].mean()), [float(np.nanmean(np.where(zm[q == u, m] > 0, zm[q == u, m], np.nan))) for m in range(4)])
                         for u in np.unique(q)}
        res["series"][s] = ser
        # calibration scores (2-year horizon) on the site's own calibration patients
        Tc, Ec = V["Ttr"][oc["row"]], V["Etr"][oc["row"]]
        kc = label_known(Tc, Ec)
        yc = label(Tc, Ec)
        if s != EXTERNAL:
            cal_scores.append(scores(oc["risk"][kc, 1], yc[kc]))
            cal_labels.append(yc[kc])
            cal_site.append(np.full(kc.sum(), s))
        streams[s] = dict(o_static=o_static, o_drift=o_drift, z_static=(nis_z(o_static["nis"], o_static["nobs"]) - ref[0]) / ref[1],
                          z_drift=(nis_z(o_drift["nis"], o_drift["nobs"]) - ref[0]) / ref[1], zgate=zgate,
                          cal=(scores(oc["risk"][kc, 1], yc[kc]), yc[kc]), alarms=alarms, oc=oc, Bcal=Bcal, Bte=Bte)
        print("site", s, "alarms", [round(a, 2) for a in alarms], "drift", t_drift,
              {k: round(v["post"]["auc2"], 3) for k, v in info.items() if isinstance(v, dict) and "post" in v},
              {k: round(v["post"]["ece2"], 3) for k, v in info.items() if isinstance(v, dict) and "post" in v},
              round(time.time() - t0), flush=True)
    pooled_s, pooled_y = np.concatenate(cal_scores), np.concatenate(cal_labels)
    # ---------------------------------------------------------------- conformal streams
    for s, S in streams.items():
        rows = S["o_static"]["row"]
        Tt, Et, cal = V["T"][rows], V["E"][rows], V["cal"][rows]
        y, kn = label(Tt, Et), label_known(Tt, Et)
        t_rev = cal + np.where((Et == 1) & (Tt <= 2), Tt, 2.0)
        cs, cy = S["cal"] if s != EXTERNAL else (pooled_s, pooled_y)
        ta = S["alarms"][0] if S["alarms"] else None
        dr = DRIFT.get(s)
        post = cal >= (dr["t"] if dr else 4.0)
        ps, pd = S["o_static"]["risk"][:, 1], S["o_drift"]["risk"][:, 1]
        gate = S["z_drift"] > S["zgate"]
        R = {}
        R["split_mondrian"] = run_stream(ps, y, kn, cal, t_rev, cs, cy, method="static")
        R["pooled_fcp"] = run_stream(ps, y, kn, cal, t_rev, pooled_s, pooled_y, method="static")
        R["aci_delayed"] = run_stream(ps, y, kn, cal, t_rev, cs, cy, method="aci")
        R["aci_rescore"] = run_stream(ps, y, kn, cal, t_rev, cs, cy, method="aci", rescore=True)
        R["ours_norefit"] = run_stream(ps, y, kn, cal, t_rev, cs, cy, method="drift", alarm_t=ta, rescore=True,
                                       gate=S["z_static"] > S["zgate"])
        R["ours_nogate"] = run_stream(pd, y, kn, cal, t_rev, cs, cy, method="drift", alarm_t=ta, rescore=True)
        R["ours"] = run_stream(pd, y, kn, cal, t_rev, cs, cy, method="drift", alarm_t=ta, gate=gate, rescore=True)
        res.setdefault("visits", {})[s] = dict(row=rows, risk=pd, risk_static=ps, ymu=S["o_drift"]["ymu"], yvar=S["o_drift"]["yvar"],
                                               z=S["z_drift"], sets=R["ours"][0], defer=R["ours"][1], sd=S["o_drift"]["sd"])
        res["conf"][s] = {m: dict(all=summarise(st, df, y, kn), post=summarise(st, df, y, kn, post),
                                  pre=summarise(st, df, y, kn, ~post)) for m, (st, df) in R.items()}
        # quarterly event coverage for the figure
        q = np.floor(cal * 4) / 4
        res["conf"][s]["series"] = {m: {float(u): (float(st[(q == u) & kn, y[(q == u) & kn]].mean()) if ((q == u) & kn).any() else np.nan,
                                                   float(st[(q == u) & kn & (y == 1), 1].mean()) if ((q == u) & kn & (y == 1)).any() else np.nan,
                                                   float(df[(q == u) & kn].mean()) if ((q == u) & kn).any() else np.nan)
                                        for u in np.unique(q)} for m, (st, df) in R.items()}
        # trajectory intervals for next-visit SBP and log NT-proBNP (standardised units)
        oc, Bcal = S["oc"], S["Bcal"]
        Bte = S["Bte"]
        tr = {}
        for yi, nm in ((0, "sbp"), (1, "bnp")):
            def nxt(o, B):
                Yn = np.concatenate([B["Y"][:, 1:], np.zeros_like(B["Y"][:, :1])], 1)
                has = np.concatenate([B["VM"][:, 1:], np.zeros_like(B["VM"][:, :1])], 1) > 0
                mlab = np.concatenate([B["M"][:, 1:, 3], np.zeros_like(B["M"][:, :1, 3])], 1) > 0
                m = B["VM"] > 0
                ok = (has & (mlab if yi == 1 else True))[m]
                return Yn[m][:, yi], ok
            yc_, okc = nxt(oc, Bcal)
            yt_, okt = nxt(S["o_drift"], Bte)
            sc = np.abs(yc_ - oc["ymu"][:, yi]) / np.sqrt(oc["yvar"][:, yi])
            qn = conf_quantile(sc[okc], 0.10)
            od = S["o_drift"]
            sdt = np.sqrt(od["yvar"][:, yi])
            err = np.abs(yt_ - od["ymu"][:, yi])
            ysd = D["stats"][3][yi]
            tr[nm] = dict(gauss_cov=float((err <= 1.645 * sdt)[okt].mean()), conf_cov=float((err <= qn * sdt)[okt].mean()),
                          gauss_w=float((2 * 1.645 * sdt * ysd)[okt].mean()), conf_w=float((2 * qn * sdt * ysd)[okt].mean()),
                          q=float(qn), n=int(okt.sum()))
        res["traj"][s] = tr
        print("conf", s, {m: (round(v["all"]["cov"], 3), round(v["all"]["cov_ev"], 3), round(v["all"]["defer"], 3))
                          for m, v in res["conf"][s].items() if m != "series"}, flush=True)
    pickle.dump(res, open(f"{RUNS}/drift_s{seed}.pkl", "wb"))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
