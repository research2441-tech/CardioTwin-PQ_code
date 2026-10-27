"""Monitoring validity at the external institution with and without the label-free adapter."""
import pickle, json
import numpy as np, jax, jax.numpy as jnp
import twin as T, drift as Dr
from dataset import build
from conformal import nis_z, cusum
from common import RUNS

D = build("full", 0)
V = D["V"]
P = pickle.load(open(f"{RUNS}/twin_params_full_e0_s0.pkl", "rb"))
P = jax.tree_util.tree_map(jnp.asarray, P)
p = P["p"]
out = {}
Bte, Bpool = Dr.add_cal(D["ext"], V), Dr.add_cal(D["ext_pool"], V)
for name, ad in (("fitted", P["ext_ad"]), ("zero", T.init_adapter())):
    oc = Dr.outputs(p, Bpool, [(-np.inf, ad)])
    zc = nis_z(oc["nis"], oc["nobs"])
    o = Dr.outputs(p, Bte, [(-np.inf, ad)])
    z = nis_z(o["nis"], o["nobs"])
    cal = V["cal"][o["row"]]
    srt = np.argsort(cal)
    # reference taken from the in-distribution calibration visits of the training institutions (pooled)
    refs = []
    for s in range(5):
        Bc = Dr.add_cal(D["cal"][s], V)
        oo = Dr.outputs(p, Bc, [(-np.inf, P["ads"][s])])
        refs.append(nis_z(oo["nis"], oo["nobs"]))
    rz = np.concatenate(refs)
    ref = (rz.mean(), rz.std())
    zs = (z - ref[0]) / ref[1]
    ia, _ = cusum(zs[srt], k=1.0, h=10.0)
    gate_thr = np.quantile((rz - ref[0]) / ref[1], 0.99)
    out[name] = dict(first_alarm=float(cal[srt][ia]) if ia >= 0 else None, n_before_alarm=int(ia),
                     gate_rate=float((zs > gate_thr).mean()), mean_z=float(zs.mean()),
                     nis_dim=[float((o["nis_m"][:, k][o["nis_m"][:, k] > 0] / 24).mean()) for k in range(4)])
    print(name, out[name], flush=True)
json.dump(out, open(f"{RUNS}/adapter_monitor.json", "w"), indent=1)
