"""Measured cost of the CardioTwin-PQ security layer (primitives, records, rounds, ledger)."""
import time, json, hashlib, statistics, sys
import numpy as np
from security import (kem, sig, aead_encrypt, aead_decrypt, hkdf, Institution, establish_pairwise,
                      secure_aggregate, Ledger, shamir_split, shamir_recover)
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from common import RUNS

REPS = 300


def bench(fn, reps=REPS):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return 1e3 * statistics.median(ts), 1e3 * np.percentile(ts, 95)


def main(n_params_twin, n_params_gru):
    R = dict(prim={}, sizes={}, record={}, rounds={}, ledger={}, shamir={})
    for name in ("ml_kem_512", "ml_kem_768", "ml_kem_1024"):
        K = kem(name)
        pk, sk = K.keygen()
        ct, ss = K.encaps(pk)
        R["prim"][name] = dict(keygen=bench(K.keygen), encaps=bench(lambda: K.encaps(pk)),
                               decaps=bench(lambda: K.decaps(sk, ct)))
        R["sizes"][name] = dict(pk=K.PUBLIC_KEY_SIZE, sk=K.SECRET_KEY_SIZE, ct=K.CIPHERTEXT_SIZE)
    msg = hashlib.sha3_256(b"x").digest()
    for name in ("ml_dsa_44", "ml_dsa_65", "ml_dsa_87"):
        S = sig(name)
        pk, sk = S.keygen()
        s = S.sign(sk, msg)
        R["prim"][name] = dict(keygen=bench(S.keygen, 100), sign=bench(lambda: S.sign(sk, msg)),
                               verify=bench(lambda: S.verify(pk, msg, s)))
        R["sizes"][name] = dict(pk=S.PUBLIC_KEY_SIZE, sk=S.SECRET_KEY_SIZE, sig=S.SIGNATURE_SIZE)
    a, b = X25519PrivateKey.generate(), X25519PrivateKey.generate()
    bp = b.public_key()
    R["prim"]["x25519"] = dict(keygen=bench(X25519PrivateKey.generate), exchange=bench(lambda: a.exchange(bp)))
    e = Ed25519PrivateKey.generate()
    es = e.sign(msg)
    ep = e.public_key()
    R["prim"]["ed25519"] = dict(sign=bench(lambda: e.sign(msg)), verify=bench(lambda: ep.verify(es, msg)))
    R["sizes"]["x25519"] = dict(pk=32, ct=32)
    R["sizes"]["ed25519"] = dict(pk=32, sig=64)
    # one encounter record: ECG + PPG (2 x 400 float32), 12x5 vitals, 6 labs, 7 EHR fields, metadata
    rec = np.random.randn(800 + 60 + 6 + 7).astype(np.float32).tobytes() + b"{" + b"m" * 120 + b"}"
    key = hkdf(b"k" * 32, b"rec")
    blob = aead_encrypt(key, rec, b"pid|visit")
    R["record"] = dict(bytes=len(rec), overhead=len(blob) - len(rec),
                       enc=bench(lambda: aead_encrypt(key, rec, b"pid|visit"), 2000),
                       dec=bench(lambda: aead_decrypt(key, blob, b"pid|visit"), 2000))
    big = np.random.bytes(10_000_000)
    t0 = time.perf_counter()
    aead_encrypt(key, big)
    R["record"]["mb_per_s"] = 10 / (time.perf_counter() - t0)
    # Shamir 3-of-5 recovery of a 256-bit key
    sec = int.from_bytes(np.random.bytes(32), "big")
    sh = shamir_split(sec, 5, 3)
    R["shamir"] = dict(split=bench(lambda: shamir_split(sec, 5, 3)), recover=bench(lambda: shamir_recover(sh[:3])))
    # rounds: 5 institutions, fresh pairwise keys per round
    for model, npar in (("twin", n_params_twin), ("gru", n_params_gru), ("encoder", 44630)):
        for suite in ("classical", "pq", "hybrid"):
            for drop in ((), (3,)):
                times, stats_all = [], []
                for r in range(5):
                    I = [Institution(i, 5, suite=suite) for i in range(5)]
                    vecs = [np.random.randn(npar) * 0.01 for _ in range(5)]
                    w = np.full(5, 0.2)
                    L = Ledger()
                    t0 = time.perf_counter()
                    kb = establish_pairwise(I, r)
                    tk = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    srv = Institution(99, 5, suite=suite)
                    agg, st = secure_aggregate(I, vecs, w, r, drop=drop, ledger=L, server=srv)
                    ta = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    ok = L.verify()
                    tl = time.perf_counter() - t0
                    ref = sum(w[i] * vecs[i] for i in range(5) if i not in drop)
                    err = float(np.abs(agg - ref).max())
                    times.append((tk, ta, tl))
                    st.update(kem_bytes=kb, err=err, ledger_ok=ok)
                    stats_all.append(st)
                tk, ta, tl = np.median(np.array(times), 0)
                R["rounds"][f"{model}|{suite}|drop{len(drop)}"] = dict(
                    n_params=npar, t_keys_ms=1e3 * tk, t_agg_ms=1e3 * ta, t_ledger_verify_ms=1e3 * tl,
                    bytes_total=stats_all[0]["bytes"] + stats_all[0]["kem_bytes"], plain_bytes=5 * 4 * npar,
                    max_err=max(s["err"] for s in stats_all), ledger_ok=all(s["ledger_ok"] for s in stats_all),
                    t_sign_ms=1e3 * np.median([s["t_sign"] for s in stats_all]),
                    t_verify_ms=1e3 * np.median([s["t_verify"] for s in stats_all]),
                    t_mask_ms=1e3 * np.median([s["t_mask"] for s in stats_all]))
                print(model, suite, drop, R["rounds"][f"{model}|{suite}|drop{len(drop)}"], flush=True)
    # ledger tamper test
    I = [Institution(i, 5) for i in range(5)]
    L = Ledger()
    for k in range(200):
        L.append(I[k % 5], "access", hashlib.sha3_256(str(k).encode()).hexdigest(), k)
    t0 = time.perf_counter()
    ok = L.verify()
    tv = time.perf_counter() - t0
    rec, h, s, a = L.entries[57]
    L.entries[57] = (rec.replace(b'"access"', b'"export"'), h, s, a)
    R["ledger"] = dict(n=200, verify_ms=1e3 * tv, ok_before=ok, ok_after_tamper=L.verify(),
                       entry_bytes=len(rec) + 32 + len(s))
    json.dump(R, open(f"{RUNS}/crypto.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
