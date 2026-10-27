"""Test-time modality removal: twin (masked information update) versus recurrent baseline (mask inputs)."""
import pickle
import numpy as np
import jax, jax.numpy as jnp
import twin as T
from dataset import build
from baselines import train_seq, seq_risk
from common import ROUNDS, RUNS, INTERNAL, evaluate

SCEN = {"none": [], "no ECG": [0], "no PPG": [1], "no labs": [3], "no waveforms": [0, 1], "vitals+EHR only": [0, 1, 3]}


def drop(B, mods):
    B = dict(B)
    M = B["M"].copy()
    E = B["E"].copy()
    for m in mods:
        M[:, :, m] = 0
        E[:, :, m] = 0
    B["M"], B["E"] = M, E
    return B


def main(seed=0):
    D = build("full", 0)
    V = D["V"]
    P = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(f"{RUNS}/twin_params_full_e0_s{seed}.pkl", "rb")))
    p = P["p"]
    g = train_seq(D["tr"], D["va"], mode="fedavg", seed=seed)["global_"]
    res = {}
    for name, mods in SCEN.items():
        rt, rg, rows = [], [], []
        for s in INTERNAL:
            B = drop(D["te"][s], mods)
            o = T.flatten(T.infer(p, P["ads"][s], B), B)
            m = B["VM"] > 0
            rt.append(o["risk"]); rg.append(seq_risk(g, B)[m]); rows.append(o["row"])
        rows = np.concatenate(rows)
        res[name] = dict(twin=evaluate(np.concatenate(rt), V["T"][rows], V["E"][rows]),
                         gru=evaluate(np.concatenate(rg), V["T"][rows], V["E"][rows]),
                         twin_risk=np.concatenate(rt), gru_risk=np.concatenate(rg), rows=rows)
        print(name, {k: round(v["auc2"], 3) for k, v in res[name].items() if isinstance(v, dict)},
              {k: round(v["ece2"], 3) for k, v in res[name].items() if isinstance(v, dict)}, flush=True)
    pickle.dump(res, open(f"{RUNS}/missing_s{seed}.pkl", "wb"))


if __name__ == "__main__":
    main()
