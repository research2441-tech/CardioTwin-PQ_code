"""Builds the development, calibration, internal-test and external tensors once per encoder variant."""
import numpy as np
from common import load_sim, make_splits, visit_table, INTERNAL, EXTERNAL, T_TRAIN_END, ENROL_SPLIT
from twin import load_embeddings, pack


def build(variant="full", eseed=0):
    pats, visits = load_sim()
    V = visit_table(pats, visits)
    role = make_splits(pats)
    r = np.array([role[p] for p in V["pid"]])
    emb = load_embeddings(variant, eseed)
    dev = V["cal"] < T_TRAIN_END
    tr_rows = np.where((r == "train") & dev)[0]
    _, stats = pack(V, emb, tr_rows, "train")
    D = dict(V=V, role=r, emb=emb, stats=stats, tr={}, va={}, cal={}, te={})
    for s in INTERNAL:
        site = V["site"] == s
        D["tr"][s], _ = pack(V, emb, np.where((r == "train") & dev & site)[0], "train", stats)
        D["va"][s], _ = pack(V, emb, np.where((r == "val") & dev & site)[0], "train", stats)
        D["cal"][s], _ = pack(V, emb, np.where((r == "cal") & dev & site)[0], "train", stats)
        D["te"][s], _ = pack(V, emb, np.where((r == "test") & site)[0], "full", stats)
    ext = V["site"] == EXTERNAL
    enrol = np.array([0.0] * len(V["pid"]))
    # adaptation pool: early external patients, inputs only (labels never touched)
    early = V["cal"] - V["t"] < ENROL_SPLIT
    D["ext_pool"], _ = pack(V, emb, np.where(ext & early & dev)[0], "train", stats)
    D["ext"], _ = pack(V, emb, np.where(ext & ~early)[0], "full", stats)
    return D
