"""CardioTwinSim: multi-institution longitudinal multimodal cardiovascular cohort simulator.

Latent physiology (5 states) evolves as a stochastic differential equation; each clinic
visit emits a 4-s single-lead ECG, a 4-s PPG, a short vital-sign series, a laboratory
panel and EHR covariates.  Sites differ in case mix, acquisition hardware, assay
calibration, missingness, visit frequency and treatment thresholds; two sites undergo
an abrupt acquisition/population change in calendar time.  The ground-truth event
process is a proportional hazard on the latent state, so oracle risks are available.
"""
import numpy as np

FS = 100          # Hz
WIN = 4.0         # seconds
NS = int(FS * WIN)
DT = 1.0 / 12.0   # monthly Euler step (years)
T_END_VISITS = 7.0
T_END_EVENTS = 10.5

SITES = [
    # name, n, age_mu, age_sd, diab, smoke, priorMI, visit_int, pECG, pPPG, pLab,
    # ecg_gain, ecg_noise, ecg_smooth, ppg_gain, ppg_noise, ppg_smooth, trop_off, ldl_off, sbp_thr, ldl_thr
    dict(name="S1", n=2000, age=(63, 10), diab=.25, smoke=.20, pmi=.12, vint=.5, pE=.95, pP=.80, pL=.90,
         eg=1.00, en=.03, es=1, pg=1.00, pn=.03, ps=1, trop=0.0, ldl=0.0, sbp_thr=135, ldl_thr=3.0),
    dict(name="S2", n=1800, age=(58, 11), diab=.18, smoke=.22, pmi=.08, vint=1.0, pE=.90, pP=.50, pL=.80,
         eg=0.85, en=.05, es=3, pg=0.90, pn=.05, ps=2, trop=0.18, ldl=0.0, sbp_thr=140, ldl_thr=3.4),
    dict(name="S3", n=1600, age=(66, 9), diab=.28, smoke=.18, pmi=.25, vint=.5, pE=.97, pP=.90, pL=.95,
         eg=1.15, en=.03, es=1, pg=1.10, pn=.03, ps=1, trop=-0.22, ldl=0.0, sbp_thr=130, ldl_thr=2.6),
    dict(name="S4", n=1400, age=(61, 12), diab=.24, smoke=.30, pmi=.10, vint=1.2, pE=.80, pP=.30, pL=.60,
         eg=1.00, en=.07, es=2, pg=1.00, pn=.07, ps=2, trop=0.0, ldl=0.15, sbp_thr=145, ldl_thr=3.6),
    dict(name="S5", n=1200, age=(55, 10), diab=.15, smoke=.16, pmi=.06, vint=.4, pE=.75, pP=.98, pL=.50,
         eg=0.95, en=.04, es=1, pg=0.70, pn=.06, ps=3, trop=0.10, ldl=0.0, sbp_thr=135, ldl_thr=3.2),
    dict(name="S6", n=1200, age=(64, 10), diab=.27, smoke=.21, pmi=.14, vint=.7, pE=.92, pP=.75, pL=.85,
         eg=1.30, en=.04, es=2, pg=1.25, pn=.04, ps=1, trop=0.34, ldl=-0.10, sbp_thr=130, ldl_thr=2.8),
]
# abrupt calendar-time changes (site index -> dict)
DRIFT = {
    2: dict(t=4.0, kind="device", eg=0.70, en=.08, es=5, pg=1.60, ps=3, ebase=0.15),
    3: dict(t=3.5, kind="assay+case-mix", trop=0.59, ldl=0.45, age_shift=6.0, diab=.38),
}
H0 = 0.012
BETA = np.array([0.45, 0.75, 0.85, 0.25, 0.45])   # hazard loadings on latent state


def _site_params(s, t):
    p = dict(SITES[s])
    d = DRIFT.get(s)
    if d is not None and t >= d["t"]:
        for k in ("eg", "en", "es", "pg", "ps", "trop", "ldl"):
            if k in d:
                p[k] = d[k]
        p["ebase"] = d.get("ebase", 0.0)
    else:
        p["ebase"] = 0.0
    return p


def _smooth(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def ecg_wave(rng, hr, hrv, af, qrs_w, st, t_amp, qt, gain, noise, smooth, ebase):
    """Sum-of-Gaussians single-lead ECG (McSharry-type morphology), 4 s at 100 Hz."""
    t = np.arange(NS) / FS
    sig = np.zeros(NS)
    rr_mu = 60.0 / hr
    beat_t = []
    tb = rng.uniform(0, rr_mu)
    while tb < WIN + 0.5:
        beat_t.append(tb)
        sd = 0.18 * rr_mu if af else hrv
        tb += max(0.3, rr_mu + rng.normal(0, sd))
    for b in beat_t:
        comps = []
        if not af:
            comps.append((-0.17, 0.12, 0.025))
        comps += [(-qrs_w * 0.35, -0.10, qrs_w / 7), (0.0, 1.0, qrs_w / 6), (qrs_w * 0.40, -0.22, qrs_w / 7),
                  (qt - 0.06, t_amp, 0.045)]
        for off, a, w in comps:
            sig += a * np.exp(-0.5 * ((t - b - off) / w) ** 2)
        # ST-segment deviation between S and T
        m = (t > b + qrs_w * 0.5) & (t < b + qt - 0.1)
        sig[m] += st
    if af:
        sig += 0.04 * np.sin(2 * np.pi * rng.uniform(5, 7) * t + rng.uniform(0, 6.28))
    sig += 0.08 * np.sin(2 * np.pi * rng.uniform(0.15, 0.4) * t + rng.uniform(0, 6.28)) + ebase
    sig = gain * _smooth(sig, smooth) + rng.normal(0, noise, NS)
    return sig.astype(np.float32), np.array(beat_t)


def ppg_wave(rng, beat_t, pat, refl_amp, refl_delay, amp, gain, noise, smooth):
    t = np.arange(NS) / FS
    sig = np.zeros(NS)
    for b in beat_t:
        o = b + pat
        sig += amp * (np.exp(-0.5 * ((t - o - 0.12) / 0.07) ** 2)
                      + refl_amp * np.exp(-0.5 * ((t - o - 0.12 - refl_delay) / 0.09) ** 2))
    # occasional motion artefact burst
    if rng.random() < 0.08:
        c = rng.uniform(0, WIN)
        sig += rng.normal(0, 0.25) * np.exp(-0.5 * ((t - c) / 0.3) ** 2) * np.sin(2 * np.pi * 2.5 * t)
    sig = gain * _smooth(sig, smooth) + rng.normal(0, noise, NS)
    return sig.astype(np.float32)


def simulate(seed=2026, scale=1.0):
    rng = np.random.default_rng(seed)
    pats, visits = [], []
    ecg_list, ppg_list = [], []
    pid = 0
    for s, sp in enumerate(SITES):
        n = int(sp["n"] * scale)
        for _ in range(n):
            enrol = rng.uniform(0, 4.5)
            dr = DRIFT.get(s)
            shifted = dr is not None and "age_shift" in dr and enrol >= dr["t"]
            age0 = rng.normal(sp["age"][0] + (dr["age_shift"] if shifted else 0), sp["age"][1])
            age0 = float(np.clip(age0, 35, 88))
            male = rng.random() < 0.55
            diab = rng.random() < (dr["diab"] if shifted else sp["diab"])
            smoke = rng.random() < sp["smoke"]
            pmi = rng.random() < sp["pmi"] + 0.004 * (age0 - 60)
            ldl0 = float(np.clip(rng.normal(3.3 + 0.2 * diab, 0.8), 1.2, 6.5))
            hdl = float(np.clip(rng.normal(1.3 - 0.15 * male - 0.1 * diab, 0.3), 0.6, 2.5))
            a = (age0 - 60) / 10
            x = np.array([0.40 * a + 0.30 * diab, 0.30 * a + 0.45 * smoke + 0.35 * diab + 0.55 * pmi + 0.35 * (ldl0 - 3.3),
                          0.20 * a + 0.80 * pmi, 0.30 * diab + 0.1 * smoke, 0.30 * a]) + rng.normal(0, 0.5, 5)
            u_off = rng.normal(0, 1, 6)  # patient-level measurement offsets
            # event time simulation on monthly grid from enrolment
            t = 0.0
            htn_tx = False
            statin = False
            ev_t, cens_t = None, None
            states = []
            n_steps = int((T_END_EVENTS - enrol) / DT)
            ldl = ldl0
            loghz_base = 0.32 * (age0 - 60) / 10 + 0.25 * male + 0.30 * diab
            for k in range(n_steps):
                states.append(x.copy())
                ldl_eff = ldl * (0.7 if statin else 1.0)
                drift = np.array([0.06 + 0.08 * diab - (0.08 if htn_tx else 0),
                                  0.05 + 0.10 * smoke + 0.10 * (ldl_eff - 3.3),
                                  0.03 + 0.06 * max(x[1], 0),
                                  0.02 + 0.04 * diab,
                                  0.03 + 0.04 * max(x[0], 0) + 0.05 * max(x[2], 0)])
                x = x + (drift - 0.12 * x) * DT + rng.normal(0, 0.22, 5) * np.sqrt(DT)
                hz = H0 * np.exp(BETA @ x + loghz_base)
                if rng.random() < 1 - np.exp(-hz * DT):
                    ev_t = (k + 1) * DT
                    break
                if rng.random() < 1 - np.exp(-0.035 * DT):
                    cens_t = (k + 1) * DT
                    break
                # treatment decisions happen at visits; approximated here monthly with the site interval
                if (k % max(1, int(sp["vint"] * 12)) == 0):
                    sbp_now = 118 + 14 * x[0] + 4 * x[1] - (8 if htn_tx else 0) + 5 * u_off[0]
                    if sbp_now > sp["sbp_thr"]:
                        htn_tx = True
                    if ldl > sp["ldl_thr"]:
                        statin = True
            follow = ev_t if ev_t is not None else (cens_t if cens_t is not None else n_steps * DT)
            event = ev_t is not None
            states = np.array(states)
            # visits
            vt = 0.0
            vtimes = [0.0]
            while True:
                vt += sp["vint"] * rng.gamma(4.0, 0.25)
                if vt >= follow or enrol + vt >= T_END_VISITS or len(vtimes) >= 10:
                    break
                vtimes.append(vt)
            pat = dict(pid=pid, site=s, enrol=enrol, age0=age0, male=int(male), diab=int(diab), smoke=int(smoke),
                       pmi=int(pmi), time=follow, event=int(event), shifted=int(shifted))
            pats.append(pat)
            htn_v, statin_v = False, False
            for j, vt in enumerate(vtimes):
                cal = enrol + vt
                p = _site_params(s, cal)
                k = min(int(vt / DT), len(states) - 1)
                z = states[k]
                age = age0 + vt
                sbp_t = 118 + 14 * z[0] + 4 * z[1] + 5 * u_off[0]
                if sbp_t > sp["sbp_thr"]:
                    htn_v = True
                if ldl0 > sp["ldl_thr"]:
                    statin_v = True if j > 0 else statin_v
                sbp_t -= 8 if htn_v else 0
                af_p = 1 / (1 + np.exp(-2.2 * (z[4] - 1.3)))
                af = rng.random() < af_p
                hr = float(np.clip(68 + 5 * z[3] + 4 * z[2] + (12 if af else 0) + 4 * u_off[1], 42, 140))
                hrv = 0.06 * np.exp(-0.45 * z[3])
                qrs = float(np.clip(0.088 + 0.016 * z[2], 0.07, 0.17))
                st = -0.05 * max(z[1], 0)
                tamp = 0.32 - 0.07 * z[1]
                qt = 0.38 + 0.015 * z[2] - 0.0008 * (hr - 70)
                has_e = rng.random() < p["pE"]
                has_p = rng.random() < p["pP"]
                has_l = rng.random() < p["pL"]
                e, beats = ecg_wave(rng, hr, hrv, af, qrs, st, tamp, qt, p["eg"], p["en"], p["es"], p["ebase"])
                pp = ppg_wave(rng, beats, 0.22 - 0.025 * z[0] + 0.01 * rng.normal(), 0.35 + 0.12 * z[0],
                              max(0.08, 0.30 - 0.04 * z[0]), max(0.3, 1 - 0.12 * z[2]), p["pg"], p["pn"], p["ps"])
                ecg_list.append(e if has_e else np.zeros(NS, np.float32))
                ppg_list.append(pp if has_p else np.zeros(NS, np.float32))
                # vital-sign series: 12 readings over a 1-h session
                tt = np.arange(12)
                vit = np.stack([
                    sbp_t + rng.normal(0, 6, 12) + 0.3 * tt,
                    74 + 6 * z[0] + 2 * z[1] + 3 * u_off[2] - (4 if htn_v else 0) + rng.normal(0, 4, 12),
                    hr + rng.normal(0, 3 if not af else 9, 12),
                    97.5 - 0.7 * z[2] + rng.normal(0, 0.6, 12),
                    15 + 1.2 * z[2] + rng.normal(0, 1.2, 12)], 1).astype(np.float32)
                lab = np.array([
                    ldl0 * (0.7 if statin_v else 1.0) + 0.2 * rng.normal() + p["ldl"],
                    hdl + 0.1 * rng.normal(),
                    5.4 + 1.6 * diab + 0.2 * z[3] + 0.25 * rng.normal(),
                    95 - 0.8 * (age - 40) - 5 * z[0] + 6 * rng.normal(),
                    4.2 + 0.9 * z[2] + 0.4 * z[4] + 0.2 * (age - 60) / 10 + 0.3 * rng.normal(),
                    1.2 + 0.35 * z[1] + 0.45 * z[2] + 0.3 * rng.normal() + p["trop"]], np.float32)
                visits.append(dict(pid=pid, site=s, vidx=j, t=vt, cal=cal, age=age, htn=int(htn_v), statin=int(statin_v),
                                   af=int(af), has_e=int(has_e), has_p=int(has_p), has_l=int(has_l),
                                   vit=vit, lab=lab, z=z.copy(), loghz=float(BETA @ z + loghz_base) + np.log(H0)))
            pid += 1
    return pats, visits, np.stack(ecg_list), np.stack(ppg_list)


if __name__ == "__main__":
    import time, pickle, sys
    t0 = time.time()
    pats, visits, E, P = simulate(scale=float(sys.argv[1]) if len(sys.argv) > 1 else 1.0)
    print(len(pats), len(visits), E.shape, time.time() - t0)
    ev = np.mean([p["event"] for p in pats])
    print("event rate", ev, "mean follow", np.mean([p["time"] for p in pats]))
    from common import DATA
    np.save(f"{DATA}/ecg.npy", E)
    np.save(f"{DATA}/ppg.npy", P)
    with open(f"{DATA}/sim.pkl", "wb") as f:
        pickle.dump(dict(pats=pats, visits=[{k: v for k, v in d.items()} for d in visits]), f)
