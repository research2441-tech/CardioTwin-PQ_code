"""Result figures (grayscale, no overlapping elements)."""
import json, pickle, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RUNS, load_sim, visit_table, td_auc

from common import FIGS as OUT
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
                     "font.size": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
NAMES = {"cox": "Cox", "rsf": "RSF", "mlp_fedavg": "MLP-FedAvg", "gru_local": "GRU-Local", "gru_fedavg": "GRU-FedAvg",
         "gru_fedprox": "GRU-FedProx", "gru_ditto": "GRU-Ditto", "gru_central": "GRU-Central", "twin_noad": "Twin, no adapter",
         "twin_central": "Twin-Central", "twin_fed": "CardioTwin-PQ"}
STY = {"cox": ("0.55", ":", "s"), "rsf": ("0.55", "--", "D"), "mlp_fedavg": ("0.40", "-.", "v"),
       "gru_fedavg": ("0.30", "--", "o"), "gru_ditto": ("0.30", ":", "^"), "gru_central": ("0.45", "-", "x"),
       "twin_noad": ("0.6", "-", "P"), "twin_central": ("0.2", "-.", "*"), "twin_fed": ("0.0", "-", "o")}


def fig_main(R):
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.35), gridspec_kw=dict(wspace=0.42))
    show = ["cox", "rsf", "mlp_fedavg", "gru_fedavg", "gru_ditto", "gru_central", "twin_fed"]
    hs = [1, 2, 3]
    for ax, split, title in ((axs[0], "main", "(a) Internal test, S1-S5"), (axs[1], "ext", "(b) External institution S6")):
        for m in show:
            if m not in R[split]:
                continue
            c, ls, mk = STY[m]
            mu = [R[split][m][f"auc{h}"][0] for h in hs]
            sd = [R[split][m][f"auc{h}"][1] for h in hs]
            ax.errorbar(hs, mu, yerr=sd, color=c, ls=ls, marker=mk, ms=3.6, lw=1.0 if m != "twin_fed" else 1.6,
                        capsize=2, label=NAMES[m], mfc="white" if m != "twin_fed" else "black")
        orc = [R["oracle"][split][f"auc{h}"] for h in hs]
        ax.plot(hs, orc, color="0.75", lw=2.6, alpha=0.9, zorder=0, label="Latent-hazard ceiling")
        ax.set_xticks(hs)
        ax.set_xticklabels(["1 y", "2 y", "3 y"])
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Time-dependent AUC")
        ax.set_title(title, fontsize=7.5, loc="left")
        ax.set_xlim(0.8, 3.2)
    lo = min(min(v["auc3"][0] for v in R[s].values()) for s in ("main", "ext")) - 0.02
    hi = max(R["oracle"]["main"]["auc1"], R["oracle"]["ext"]["auc1"]) + 0.01
    for ax in axs[:2]:
        ax.set_ylim(lo, hi)
    h_, l_ = axs[0].get_legend_handles_labels()
    # calibration panel
    ax = axs[2]
    P = pickle.load(open(f"{RUNS}/preds_full_e0_s0.pkl", "rb"))
    C = pickle.load(open(f"{RUNS}/preds_classical.pkl", "rb"))
    pats, visits = load_sim()
    V = visit_table(pats, visits)
    ax.plot([0, 0.6], [0, 0.6], color="0.7", lw=0.8, ls="-")
    for m, src in (("twin_fed", P), ("gru_fedavg", P), ("cox", C)):
        d = src[m]
        r = d["risk"][:, 1]
        T, E = V["T"][d["row"]], V["E"][d["row"]]
        known = ((T <= 2) & (E == 1)) | (T > 2)
        y = ((T <= 2) & (E == 1)).astype(float)
        r, y = r[known], y[known]
        q = np.quantile(r, np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(q, r, side="right") - 1, 0, 9)
        px = [r[b == k].mean() for k in range(10)]
        py = [y[b == k].mean() for k in range(10)]
        n = [np.sum(b == k) for k in range(10)]
        se = [1.96 * np.sqrt(max(p_ * (1 - p_), 1e-4) / k) for p_, k in zip(py, n)]
        c, ls, mk = STY[m]
        ax.errorbar(px, py, yerr=se, color=c, ls=ls, marker=mk, ms=3, lw=1.0 if m != "twin_fed" else 1.5, capsize=1.5,
                    mfc="white" if m != "twin_fed" else "black", label=NAMES[m])
    ax.set_xlabel("Predicted 2-year risk")
    ax.set_ylabel("Observed 2-year event rate")
    ax.set_title("(c) Calibration, internal", fontsize=7.5, loc="left")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    ax.legend(fontsize=6.0, loc="upper left", handlelength=2.4)
    fig.legend(h_, l_, loc="upper center", ncol=4, fontsize=6.2, bbox_to_anchor=(0.36, -0.10), handlelength=2.6,
               columnspacing=1.2)
    fig.savefig(f"{OUT}/fig4_main.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def fig_missing():
    M = pickle.load(open(f"{RUNS}/missing_s0.pkl", "rb"))
    sc = list(M.keys())
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.15), gridspec_kw=dict(wspace=0.55))
    x = np.arange(len(sc))
    w = 0.38
    for ax, k, lab in ((axs[0], "auc2", "2-year AUC"), (axs[1], "ece2", "2-year ECE")):
        a = [M[s]["twin"][k] for s in sc]
        b = [M[s]["gru"][k] for s in sc]
        ax.bar(x - w / 2, a, w, color="0.15", label="CardioTwin-PQ")
        ax.bar(x + w / 2, b, w, color="white", edgecolor="0.15", hatch="////", lw=0.6, label="GRU-FedAvg")
        ax.set_xticks(x)
        ax.set_xticklabels(["all present" if s_ == "none" else s_ for s_ in sc], rotation=55, ha="right", fontsize=6.2)
        ax.set_ylabel(lab)
        if k == "auc2":
            lo = min(min(a), min(b)) - 0.03
            ax.set_ylim(lo, max(max(a), max(b)) + 0.02)
    axs[0].legend(fontsize=6.0, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=1.4)
    fig.savefig(f"{OUT}/fig5_missing.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    R = json.load(open(f"{RUNS}/results.json"))
    fig_main(R)
    try:
        fig_missing()
    except FileNotFoundError:
        pass


def fig_drift():
    R = pickle.load(open(f"{RUNS}/drift_s0.pkl", "rb"))
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.2), gridspec_kw=dict(wspace=0.28))
    mods = ["ECG", "PPG", "Vital signs", "Laboratory"]
    sty = [("0.0", "-", "o"), ("0.0", "--", "s"), ("0.55", "-", "^"), ("0.35", ":", "D")]
    for ax, s, t0, title in ((axs[0], 2, 4.0, "(a) S3: ECG/PPG device replacement"),
                             (axs[1], 3, 3.5, "(b) S4: troponin/LDL assay change + case mix")):
        ser = R["series"][s]["static"]
        qs = sorted(ser)
        for k in range(4):
            ax.plot(qs, [ser[q][1][k] for q in qs], color=sty[k][0], ls=sty[k][1], marker=sty[k][2], ms=2.6, lw=1.0,
                    mfc="white", label=mods[k])
        ax.axvline(t0, color="0.5", lw=0.8, ls=(0, (4, 2)))
        ax.text(t0 + 0.06, 0.97, "change", transform=ax.get_xaxis_transform(), fontsize=6.2, va="top", ha="left")
        for a in R["sites"][s]["alarms"]:
            ax.annotate("alarm", xy=(a, ax.get_ylim()[0]), xytext=(a + 0.35, 0.80), textcoords=("data", "axes fraction"),
                        fontsize=6.2, arrowprops=dict(arrowstyle="-|>", lw=0.7, color="black"), ha="left")
        ax.set_xlabel("Calendar time (years)")
        ax.set_ylabel("NIS per dimension (un-adapted)")
        ax.set_title(title, fontsize=7.5, loc="left")
        ax.set_yscale("log")
        ax.set_ylim(0.4, 20)
        ax.set_xlim(2.4, 7.0)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=6.4, bbox_to_anchor=(0.5, -0.02), handlelength=2.4, columnspacing=1.6)
    fig.savefig(f"{OUT}/fig6_drift.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def fig_conformal():
    R = pickle.load(open(f"{RUNS}/drift_s0.pkl", "rb"))
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.15), gridspec_kw=dict(wspace=0.40))
    sty = {"split_mondrian": ("0.55", ":", "s", "Split-Mondrian"), "aci_delayed": ("0.30", "--", "v", "ACI, delayed labels"),
           "aci_rescore": ("0.45", "-.", "^", "ACI, re-scored"), "ours": ("0.0", "-", "o", "CardioTwin-PQ")}
    # (a) overall coverage over time at S1: ACI overshoot
    ax = axs[0]
    for m in ("split_mondrian", "aci_delayed", "aci_rescore", "ours"):
        ser = R["conf"][0]["series"][m]
        qs = [q for q in sorted(ser) if q < 6.99]
        yy = []
        for q in qs:
            yy.append(ser[q][0])
        c, ls, mk, lab = sty[m]
        ax.plot(qs, yy, color=c, ls=ls, marker=mk, ms=2.4, lw=1.0, mfc="white" if m != "ours" else c, label=lab)
    ax.axhline(0.9, color="0.7", lw=0.8)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Calendar time (years)")
    ax.set_ylabel("Coverage (all visits)")
    ax.set_title("(a) S1, no change", fontsize=7.5, loc="left")
    # (b) S3 event coverage in half-year windows
    ax = axs[1]
    for m in ("split_mondrian", "aci_rescore", "ours"):
        ser = R["conf"][2]["series"][m]
        qs = sorted(ser)
        halves = sorted(set(np.floor(np.array(qs) * 2) / 2))
        xx, yy = [], []
        for h in halves:
            v = [ser[q][1] for q in qs if np.floor(q * 2) / 2 == h and not np.isnan(ser[q][1])]
            if v:
                xx.append(h + 0.25); yy.append(np.mean(v))
        c, ls, mk, lab = sty[m]
        ax.plot(xx, yy, color=c, ls=ls, marker=mk, ms=2.6, lw=1.0, mfc="white" if m != "ours" else c, label=lab)
    ax.axhline(0.9, color="0.7", lw=0.8)
    ax.axvline(4.0, color="0.5", lw=0.8, ls=(0, (4, 2)))
    ax.set_ylim(0.55, 1.02)
    ax.set_xlabel("Calendar time (years)")
    ax.set_ylabel("Event coverage")
    ax.set_title("(b) S3, device change at 4.0", fontsize=7.5, loc="left")
    # (c) post-change summary bars at S3
    ax = axs[2]
    ms = ["split_mondrian", "pooled_fcp", "aci_rescore", "ours_nogate", "ours"]
    labs = ["Split", "Pooled", "ACI-r", "No gate", "Full"]
    ev = [R["conf"][2][m]["post"]["cov_ev"] for m in ms]
    miss = [R["conf"][2][m]["post"]["miss_ev"] for m in ms]
    dfr = [R["conf"][2][m]["post"]["defer"] for m in ms]
    x = np.arange(len(ms))
    ax.bar(x - 0.2, miss, 0.4, color="0.15", label="Missed events")
    ax.bar(x + 0.2, dfr, 0.4, color="white", edgecolor="0.15", hatch="////", lw=0.6, label="Deferral")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=6.2, rotation=30, ha="right")
    ax.set_ylabel("Rate after the change")
    ax.set_title("(c) S3 after the change", fontsize=7.5, loc="left")
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=6.0, loc="upper left", handlelength=1.4)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=6.2, bbox_to_anchor=(0.36, -0.06), handlelength=2.4, columnspacing=1.4)
    fig.savefig(f"{OUT}/fig7_conformal.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def fig_patient(site=0, pid=None):
    """One patient followed by the twin: 2-year risk with conformal decisions and filtered-state uncertainty."""
    R = pickle.load(open(f"{RUNS}/drift_s0.pkl", "rb"))
    pats, visits = load_sim()
    V = visit_table(pats, visits)
    S = R["visits"][site]
    rows = S["row"]
    if pid is None:
        best = None
        for u in np.unique(V["pid"][rows]):
            m = V["pid"][rows] == u
            r = rows[m]
            if m.sum() < 7 or V["E"][r][0] != 1:
                continue
            o = np.argsort(V["t"][r])
            rk = S["risk"][np.where(m)[0][o]]
            miss = ((V["has_l"][r] == 0) | (V["has_p"][r] == 0)).sum()
            if miss == 0 or miss > 3:
                continue
            score = rk[-1] - rk[0]
            if best is None or score > best[0]:
                best = (score, u)
        pid = best[1]
    m = V["pid"][rows] == pid
    r = rows[m]
    o = np.argsort(V["t"][r])
    r = r[o]
    idx = np.where(m)[0][o]
    t = V["t"][r]
    risk = S["risk"][idx]
    sets, defer = S["sets"][idx], S["defer"][idx]
    sd = S["sd"][idx].mean(1)
    ev_time = V["t"][r][0] + V["T"][r][0]
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 3.2), sharex=True, gridspec_kw=dict(hspace=0.16, height_ratios=[1.15, 1]))
    ax = axs[0]
    ax.plot(t, risk, color="0.0", lw=1.0, zorder=1)
    for ti, ri, st, df in zip(t, risk, sets, defer):
        fc, mk = ("0.6", "o") if df else (("black", "s") if st[1] else ("white", "o"))
        ax.scatter([ti], [ri], s=26, marker=mk, facecolor=fc, edgecolor="black", lw=0.6, zorder=3)
    ax.axvline(ev_time, color="0.0", lw=0.9, ls=":")
    ax.text(ev_time - 0.05, 0.95, "event", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=6.2)
    ax.set_ylabel("2-year risk")
    ax.set_ylim(0, max(0.5, risk.max() * 1.3))
    from matplotlib.lines import Line2D
    hl = [Line2D([], [], marker="s", ls="", mfc="black", mec="black", ms=4.5, label="set {1}"),
          Line2D([], [], marker="o", ls="", mfc="white", mec="black", ms=4.5, label="set {0}"),
          Line2D([], [], marker="o", ls="", mfc="0.6", mec="black", ms=4.5, label="deferred")]
    ax.legend(handles=hl, fontsize=6.0, loc="lower left", ncol=3, handletextpad=0.3, columnspacing=0.9, bbox_to_anchor=(0, 1.0))
    ax = axs[1]
    ax.plot(t, sd, color="0.0", lw=1.0, marker="o", ms=3.2, mfc="white")
    k_ann = 0
    for ti, si, rr in zip(t, sd, r):
        lab = []
        if V["has_l"][rr] == 0:
            lab.append("no lab")
        if V["has_p"][rr] == 0:
            lab.append("no PPG")
        if V["has_e"][rr] == 0:
            lab.append("no ECG")
        if lab:
            up = k_ann % 2 == 0
            ax.annotate("\n".join(lab), xy=(ti, si), xytext=(0, 8 if up else -8), textcoords="offset points", fontsize=5.8,
                        ha="center", va="bottom" if up else "top")
            k_ann += 1
    ax.axvline(ev_time, color="0.0", lw=0.9, ls=":")
    ax.set_ylabel("Posterior SD of $\\mathbf{z}$")
    ax.set_xlabel("Years since enrolment")
    ax.set_ylim(min(sd) * 0.75, max(sd) * 1.25)
    fig.savefig(f"{OUT}/fig_patient.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return pid


def fig_security():
    C = json.load(open(f"{RUNS}/crypto.json"))
    P = pickle.load(open(f"{RUNS}/preds_full_e0_s0.pkl", "rb"))
    round_train = P["timing"]["twin_fed"] / 40.0
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.35), gridspec_kw=dict(wspace=0.62, width_ratios=[1.0, 1.15]))
    ax = axs[0]
    items = [("ML-KEM-768\nencaps", C["prim"]["ml_kem_768"]["encaps"][0]), ("ML-KEM-768\ndecaps", C["prim"]["ml_kem_768"]["decaps"][0]),
             ("X25519\nexchange", C["prim"]["x25519"]["exchange"][0]), ("ML-DSA-65\nsign", C["prim"]["ml_dsa_65"]["sign"][0]),
             ("ML-DSA-65\nverify", C["prim"]["ml_dsa_65"]["verify"][0]), ("Ed25519\nsign", C["prim"]["ed25519"]["sign"][0]),
             ("XChaCha20\nrecord", C["record"]["enc"][0])]
    y = np.arange(len(items))[::-1]
    ax.barh(y, [v for _, v in items], color=["0.15", "0.15", "white", "0.15", "0.15", "white", "0.6"], edgecolor="0.15", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([k for k, _ in items], fontsize=5.6)
    ax.set_xscale("log")
    ax.set_xlabel("Median latency (ms)")
    ax.set_title("(a) Primitives", fontsize=7.2, loc="left")
    ax = axs[1]
    suites = ["classical", "hybrid", "pq"]
    labs = ["Classical", "Hybrid", "PQ"]
    models = [("twin", "Twin"), ("gru", "GRU"), ("encoder", "Encoder")]
    w = 0.26
    x = np.arange(len(models))
    hatches = ["", "////", ""]
    fills = ["white", "white", "0.15"]
    for k, s in enumerate(suites):
        v = [C["rounds"][f"{m}|{s}|drop0"]["t_keys_ms"] + C["rounds"][f"{m}|{s}|drop0"]["t_agg_ms"] for m, _ in models]
        ax.bar(x + (k - 1) * w, v, w, color=fills[k], hatch=hatches[k], edgecolor="0.15", lw=0.6, label=labs[k])
    ax.axhline(1e3 * round_train, color="0.4", lw=0.9, ls=(0, (4, 2)))
    ax.text(len(models) - 0.5, 1e3 * round_train * 1.15, "twin local\ntraining round", fontsize=5.6, ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in models], fontsize=6.4)
    ax.set_ylabel("Security time per round (ms)")
    ax.set_title("(b) Per round, 5 sites", fontsize=7.2, loc="left")
    ax.legend(fontsize=5.8, loc="upper left", handlelength=1.3, bbox_to_anchor=(0.0, 0.80))
    ax.set_ylim(1, 1e3 * round_train * 6)
    fig.savefig(f"{OUT}/fig8_security.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
