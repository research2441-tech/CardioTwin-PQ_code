"""Post-quantum security layer of CardioTwin-PQ.

* ML-KEM (FIPS 203) pairwise key establishment between institutions (ciphertexts relayed by the server)
* HKDF-SHA256 derivation of pairwise mask seeds and record-wrapping keys
* XChaCha20-Poly1305 authenticated encryption of patient records and share envelopes
* Secure aggregation with pairwise cancelling masks + self masks, Shamir t-of-n recovery for dropouts
* ML-DSA (FIPS 204) signatures on masked updates, the aggregated model, and audit-ledger entries
* SHA3-256 hash-chained append-only audit ledger
Classical (X25519/Ed25519) and hybrid suites are provided for cost comparison.
"""
import hashlib, hmac, os, secrets, struct, time, json
import numpy as np
from nacl.bindings import (crypto_aead_xchacha20poly1305_ietf_encrypt as xc_enc,
                           crypto_aead_xchacha20poly1305_ietf_decrypt as xc_dec)
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
import importlib

PRIME = 2 ** 521 - 1          # Mersenne prime field for Shamir sharing
QBITS = 32                    # modulus 2^32 for masked fixed-point arithmetic
SCALE = 2 ** 16


def kem(name="ml_kem_768"):
    return importlib.import_module(f"pqcrypto.kem.{name}")


def sig(name="ml_dsa_65"):
    return importlib.import_module(f"pqcrypto.sign.{name}")


# ------------------------------------------------------------------ primitives
def hkdf(ikm, info, length=32, salt=b"CardioTwin-PQ"):
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    out, t, i = b"", b"", 1
    while len(out) < length:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        out += t
        i += 1
    return out[:length]


def aead_encrypt(key, plaintext, aad=b""):
    nonce = secrets.token_bytes(24)
    return nonce + xc_enc(plaintext, aad, nonce, key)


def aead_decrypt(key, blob, aad=b""):
    return xc_dec(blob[24:], aad, blob[:24], key)


def prg_mask(seed, n):
    """Expand a 32-byte seed into n uint32 words (SHAKE-256 as the PRG)."""
    return np.frombuffer(hashlib.shake_256(seed).digest(4 * n), dtype=np.uint32).copy()


def shamir_split(secret_int, n, t):
    coef = [secret_int] + [secrets.randbelow(PRIME) for _ in range(t - 1)]
    return [(x, sum(c * pow(x, k, PRIME) for k, c in enumerate(coef)) % PRIME) for x in range(1, n + 1)]


def shamir_recover(shares):
    s = 0
    for j, (xj, yj) in enumerate(shares):
        num, den = 1, 1
        for m, (xm, _) in enumerate(shares):
            if m != j:
                num = num * (-xm) % PRIME
                den = den * (xj - xm) % PRIME
        s = (s + yj * num * pow(den, -1, PRIME)) % PRIME
    return s


# ------------------------------------------------------------------ fixed-point encoding
def encode(vec, clip=64.0):
    q = np.round(np.clip(vec, -clip, clip) * SCALE).astype(np.int64)
    return (q % (2 ** QBITS)).astype(np.uint32)


def decode(u32, count=1):
    v = u32.astype(np.int64)
    v = np.where(v >= 2 ** (QBITS - 1), v - 2 ** QBITS, v)
    return v.astype(np.float64) / SCALE


# ------------------------------------------------------------------ institutions
class Institution:
    def __init__(self, idx, n_inst, kem_name="ml_kem_768", sig_name="ml_dsa_65", suite="pq"):
        self.idx, self.n = idx, n_inst
        self.suite = suite
        self.K, self.S = kem(kem_name), sig(sig_name)
        if suite in ("pq", "hybrid"):
            self.kem_pk, self.kem_sk = self.K.keygen()
            self.sig_pk, self.sig_sk = self.S.keygen()
        if suite in ("classical", "hybrid"):
            self.x_sk = X25519PrivateKey.generate()
            self.x_pk = self.x_sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            self.ed_sk = Ed25519PrivateKey.generate()
            self.ed_pk = self.ed_sk.public_key()
        self.pair = {}

    def sign(self, msg):
        out = b""
        if self.suite in ("pq", "hybrid"):
            out += self.S.sign(self.sig_sk, msg)
        if self.suite in ("classical", "hybrid"):
            out += self.ed_sk.sign(msg)
        return out

    def verify(self, msg, s):
        if self.suite in ("pq", "hybrid"):
            self.S.verify(self.sig_pk, msg, s[:self.S.SIGNATURE_SIZE])
            s = s[self.S.SIGNATURE_SIZE:]
        if self.suite in ("classical", "hybrid"):
            self.ed_pk.verify(s, msg)
        return True


def establish_pairwise(insts, round_id):
    """Every unordered pair (i<j): i encapsulates to j (PQ) and/or performs X25519; both derive seed_ij."""
    bytes_up = 0
    for i in insts:
        for j in insts:
            if i.idx >= j.idx:
                continue
            ikm = b""
            if i.suite in ("pq", "hybrid"):
                ct, ss_i = i.K.encaps(j.kem_pk)
                ss_j = j.K.decaps(j.kem_sk, ct)
                assert ss_i == ss_j
                ikm += ss_i
                bytes_up += len(ct)
            if i.suite in ("classical", "hybrid"):
                from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey
                ss = i.x_sk.exchange(X25519PublicKey.from_public_bytes(j.x_pk))
                ikm += ss
            info = f"pair|{round_id}|{i.idx}|{j.idx}".encode()
            seed = hkdf(ikm, info)
            i.pair[j.idx] = seed
            j.pair[i.idx] = seed
    return bytes_up


def masked_update(inst, vec_u32, round_id):
    n = len(vec_u32)
    y = vec_u32.astype(np.uint64)
    for j, seed in inst.pair.items():
        m = prg_mask(seed + struct.pack(">I", round_id), n).astype(np.uint64)
        y = (y + m) if inst.idx < j else (y - m)
    inst.self_seed = secrets.token_bytes(32)
    y = y + prg_mask(inst.self_seed, n).astype(np.uint64)
    return (y % (2 ** QBITS)).astype(np.uint32)


def secure_aggregate(insts, vecs, weights, round_id=0, t=3, drop=(), ledger=None, server=None):
    """Weighted secure aggregation. vecs: list of float arrays; weights sum to 1.
    Institutions in `drop` fail after key agreement; their pairwise masks are rebuilt from Shamir shares."""
    stats = dict(bytes=0, t_mask=0.0, t_sign=0.0, t_verify=0.0, t_unmask=0.0)
    n = len(vecs[0])
    alive = [i for i in insts if i.idx not in drop]
    # each institution Shamir-shares its self-mask seed and pairwise seeds (envelopes encrypted with pairwise keys)
    msgs = {}
    shares_self, shares_pair = {}, {}
    for inst in insts:
        u = encode(vecs[inst.idx] * weights[inst.idx])
        t0 = time.perf_counter()
        y = masked_update(inst, u, round_id)
        stats["t_mask"] += time.perf_counter() - t0
        shares_self[inst.idx] = shamir_split(int.from_bytes(inst.self_seed, "big"), len(insts), t)
        shares_pair[inst.idx] = {j: shamir_split(int.from_bytes(s, "big"), len(insts), t) for j, s in inst.pair.items()}
        # share envelopes: one AEAD blob per recipient
        for k in insts:
            if k.idx != inst.idx:
                payload = json.dumps([str(shares_self[inst.idx][k.idx][1])]).encode()
                stats["bytes"] += len(aead_encrypt(inst.pair[k.idx], payload))
        if inst.idx in drop:
            continue
        body = y.tobytes()
        t0 = time.perf_counter()
        s = inst.sign(hashlib.sha3_256(body + struct.pack(">I", round_id)).digest())
        stats["t_sign"] += time.perf_counter() - t0
        msgs[inst.idx] = (y, s)
        stats["bytes"] += len(body) + len(s)
    # server verifies signatures
    for idx, (y, s) in msgs.items():
        t0 = time.perf_counter()
        insts[idx].verify(hashlib.sha3_256(y.tobytes() + struct.pack(">I", round_id)).digest(), s)
        stats["t_verify"] += time.perf_counter() - t0
        if ledger is not None:
            ledger.append(insts[idx], "masked_update", hashlib.sha3_256(y.tobytes()).hexdigest(), round_id)
    t0 = time.perf_counter()
    agg = np.zeros(n, np.uint64)
    for y, _ in msgs.values():
        agg = (agg + y.astype(np.uint64)) % (2 ** QBITS)
    # remove self masks of survivors (t shares from survivors)
    for inst in alive:
        sh = [shares_self[inst.idx][k.idx] for k in alive[:t]]
        seed = shamir_recover(sh).to_bytes(32, "big")
        agg = (agg - prg_mask(seed, n).astype(np.uint64)) % (2 ** QBITS)
    # cancel dangling pairwise masks of dropped institutions
    for d in drop:
        for inst in alive:
            sh = [shares_pair[d][inst.idx][k.idx] for k in alive[:t]]
            seed = shamir_recover(sh).to_bytes(32, "big")
            m = prg_mask(seed + struct.pack(">I", round_id), n).astype(np.uint64)
            # survivor inst added +m if inst<d else -m; remove that contribution
            agg = (agg - m) % (2 ** QBITS) if inst.idx < d else (agg + m) % (2 ** QBITS)
    stats["t_unmask"] = time.perf_counter() - t0
    if server is not None:
        # the server signs the aggregate it broadcasts and logs it
        t0 = time.perf_counter()
        digest = hashlib.sha3_256(agg.astype(np.uint32).tobytes() + struct.pack(">I", round_id)).digest()
        stats["server_sig"] = server.sign(digest)
        stats["t_server_sign"] = time.perf_counter() - t0
        if ledger is not None:
            ledger.append(server, "aggregate", digest.hex(), round_id)
    return decode(agg.astype(np.uint32)), stats


# ------------------------------------------------------------------ audit ledger
class Ledger:
    def __init__(self):
        self.entries = []
        self.head = b"\x00" * 32

    def append(self, actor, action, obj_hash, round_id):
        rec = json.dumps(dict(prev=self.head.hex(), actor=actor.idx, action=action, obj=obj_hash, round=round_id,
                              ts=time.time()), sort_keys=True).encode()
        h = hashlib.sha3_256(rec).digest()
        s = actor.sign(h)
        self.entries.append((rec, h, s, actor))
        self.head = h

    def verify(self):
        prev = b"\x00" * 32
        for rec, h, s, actor in self.entries:
            d = json.loads(rec)
            if bytes.fromhex(d["prev"]) != prev or hashlib.sha3_256(rec).digest() != h:
                return False
            actor.verify(h, s)
            prev = h
        return True
