"""Train the twin with every FedAvg step replaced by the post-quantum secure aggregation protocol and compare
with plaintext aggregation (same seed)."""
import pickle, time
import numpy as np
import jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import twin as T
from dataset import build
from security import Institution, establish_pairwise, secure_aggregate, Ledger
from common import ROUNDS, RUNS, INTERNAL, evaluate

LOG = dict(t_agg=[], bytes=[], err=[])


def make_agg(n_sites):
    led = Ledger()

    def agg(loc, wts, rd):
        flat = [ravel_pytree(p)[0] for p in loc]
        unravel = ravel_pytree(loc[0])[1]
        insts = [Institution(i, n_sites) for i in range(n_sites)]
        t0 = time.perf_counter()
        kb = establish_pairwise(insts, rd)
        vec, st = secure_aggregate(insts, [np.asarray(f, np.float64) for f in flat], np.asarray(wts), rd, ledger=led)
        LOG["t_agg"].append(time.perf_counter() - t0)
        LOG["bytes"].append(st["bytes"] + kb)
        ref = sum(w * np.asarray(f, np.float64) for w, f in zip(wts, flat))
        LOG["err"].append(float(np.abs(vec - ref).max()))
        return unravel(jnp.asarray(vec, jnp.float32))
    return agg, led


def main():
    D = build("full", 0)
    V = D["V"]
    agg, led = make_agg(len(INTERNAL))
    p, ads, _ = T.train_federated(D["tr"], D["va"], seed=0, rounds=ROUNDS, agg_fn=agg, verbose=False)
    P0 = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(f"{RUNS}/twin_params_full_e0_s0.pkl", "rb")))
    out = {}
    for tag, pp, aa in (("secure", p, ads), ("plain", P0["p"], P0["ads"])):
        rs, rows = [], []
        for s in INTERNAL:
            o = T.flatten(T.infer(pp, aa[s], D["te"][s]), D["te"][s])
            rs.append(o["risk"]); rows.append(o["row"])
        rows = np.concatenate(rows)
        out[tag] = evaluate(np.concatenate(rs), V["T"][rows], V["E"][rows])
        out[tag + "_risk"] = np.concatenate(rs)
    out["max_abs_risk_diff"] = float(np.abs(out["secure_risk"] - out["plain_risk"]).max())
    out["log"] = LOG
    out["ledger_ok"] = led.verify()
    out["ledger_n"] = len(led.entries)
    print({k: v for k, v in out.items() if not k.endswith("_risk") and k != "log"},
          "agg ms", 1e3 * np.median(LOG["t_agg"]), "max quant err", max(LOG["err"]))
    pickle.dump(out, open(f"{RUNS}/secagg_equiv.pkl", "wb"))


if __name__ == "__main__":
    main()
