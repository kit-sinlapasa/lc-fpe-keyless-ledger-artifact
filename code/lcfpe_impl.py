#!/usr/bin/env python3
"""
LC-FPE code layer + the Paillier COMPARISON BASELINE for the amount layer.

Everything here is real crypto -- nothing is simulated:
  * FF1      NIST SP 800-38G, 10-round Feistel, AES-CBC-MAC PRF
  * FPAE     exact l1-optimal alias allocation + HMAC-SHA256 selection
  * Paillier additively homomorphic amounts, with the two standard
             deployment optimisations: g = 1+n  (so g^m = 1+mn mod n^2)
             and an offline pool of precomputed r^n values
Baseline: AES-256-GCM record encryption.

IMPORTANT -- what this file is and is not.

  The amount layer measured here is PAILLIER, which the manuscript reports as
  the REJECTED alternative: 512 B per amount, and the balance check needs the
  private key.  The DEPLOYED design uses Pedersen commitments with an offset
  encoding, implemented and measured in pedersen_bench.py and offset_bench.py,
  which is where the 515 us/row and 78 B/row figures in the paper come from.
  Keep this file for the Paillier-vs-Pedersen comparison table; do not read it
  as the reference implementation of the scheme.
"""
import os
import time
import math
import hmac
import hashlib
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from Crypto.Util.number import getPrime, inverse

from paperb_protocol import encode_row_id


# --------------------------------------------------------------- AES helpers
def aes_ecb(key, block):
    enc = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    return enc.update(block) + enc.finalize()


def cbc_mac(key, data):
    y = b"\x00" * 16
    for i in range(0, len(data), 16):
        blk = data[i:i + 16]
        y = aes_ecb(key, bytes(a ^ b for a, b in zip(y, blk)))
    return y


# --------------------------------------------------------------- FF1
class FF1:
    """NIST SP 800-38G FF1.  Numerals are lists of ints in [0, radix)."""

    def __init__(self, key, radix=10):
        self.key = key
        self.radix = radix

    def _setup(self, x, tweak):
        r = self.radix
        n = len(x)
        u = n // 2
        v = n - u
        b = (math.ceil(math.ceil(v * math.log2(r)) / 8))
        d = 4 * math.ceil(b / 4) + 4
        t = len(tweak)
        P = (bytes([1, 2, 1]) + r.to_bytes(3, "big") + bytes([10, u % 256])
             + n.to_bytes(4, "big") + t.to_bytes(4, "big"))
        return u, v, b, d, t, P

    def _num(self, half):
        n = 0
        for dig in half:
            n = n * self.radix + dig
        return n

    def _str(self, c, m):
        C = []
        for _ in range(m):
            C.append(c % self.radix)
            c //= self.radix
        C.reverse()
        return C

    def _y(self, P, tweak, i, half, b, d, t):
        """The round function of SP 800-38G, steps 6.i-6.iv."""
        pad = (-t - b - 1) % 16
        Q = tweak + b"\x00" * pad + bytes([i]) + self._num(half).to_bytes(b, "big")
        R = cbc_mac(self.key, P + Q)
        S = bytearray(R)
        j = 1
        while len(S) < d:
            blk = bytes(p ^ q for p, q in zip(R, j.to_bytes(16, "big")))
            S += aes_ecb(self.key, blk)
            j += 1
        return int.from_bytes(bytes(S[:d]), "big")

    def encrypt(self, x, tweak=b""):
        u, v, b, d, t, P = self._setup(x, tweak)
        A, B = x[:u], x[u:]
        for i in range(10):
            y = self._y(P, tweak, i, B, b, d, t)
            m = u if i % 2 == 0 else v
            C = self._str((self._num(A) + y) % (self.radix ** m), m)
            A, B = B, C
        return A + B

    def decrypt(self, x, tweak=b""):
        """SP 800-38G Algorithm 8: the rounds run backwards and the halves
           exchange roles, so B is recovered by subtracting the round value."""
        u, v, b, d, t, P = self._setup(x, tweak)
        A, B = x[:u], x[u:]
        for i in range(9, -1, -1):
            y = self._y(P, tweak, i, A, b, d, t)
            m = u if i % 2 == 0 else v
            C = self._str((self._num(B) - y) % (self.radix ** m), m)
            B = A
            A = C
        return A + B


# --------------------------------------------------------------- Paillier
class Paillier:
    def __init__(self, bits=2048, pool=4096):
        p = getPrime(bits // 2)
        q = getPrime(bits // 2)
        self.n = p * q
        self.n2 = self.n * self.n
        self.lam = (p - 1) * (q - 1)
        self.mu = inverse(self.lam, self.n)
        self.nbytes = (self.n2.bit_length() + 7) // 8
        t0 = time.time()
        self.pool = [pow(int.from_bytes(os.urandom(bits // 8), "big") % self.n,
                         self.n, self.n2) for _ in range(pool)]
        self.pool_time = time.time() - t0
        self.pi = 0

    def enc(self, m):
        r = self.pool[self.pi % len(self.pool)]
        self.pi += 1
        return ((1 + (m % self.n) * self.n) * r) % self.n2

    def add(self, a, b):
        return (a * b) % self.n2

    def dec(self, c):
        x = pow(c, self.lam, self.n2)
        m = ((x - 1) // self.n * self.mu) % self.n
        return m - self.n if m > self.n // 2 else m


# --------------------------------------------------------------- FPAE
def alloc_hamilton(f, K):
    """Delegates to harness.alloc_opt, the exact L1 minimiser; the
    largest-remainder body below is unreachable and kept for provenance."""
    from harness import alloc_opt
    return alloc_opt(f, K)


def _alloc_hamilton_legacy(f, K):
    """Largest-remainder apportionment with a floor of one alias per account,
       balanced to budget in both directions so that sum(k) == K exactly.
       The clawback branch matters: without it K' != K introduces a constant
       scale term correlated with account frequency."""
    m = len(f)
    if K < m:
        raise ValueError("infeasible: K={} < m={}".format(K, m))
    q = f * K
    base = np.maximum(np.floor(q).astype(np.int64), 1)
    short = int(K - base.sum())
    if short > 0:
        base[np.argsort(-(q - np.floor(q)))[:short]] += 1
    while short < 0:
        # repeat until the budget is met: one pass can be insufficient when
        # the surplus exceeds the number of accounts holding more than one
        cand = np.nonzero(base > 1)[0]
        if cand.size == 0:
            raise RuntimeError("clawback exhausted")
        take = min(-short, cand.size)
        worst = cand[np.argsort((q - base)[cand])[:take]]
        base[worst] -= 1
        short += take
    assert int(base.sum()) == K, (int(base.sum()), K)
    return base


ALPHA = 10_000      # format space of the four-digit account-code field
N_CC_FMT = 1_000    # three digits
N_DT_FMT = 100      # two digits
ALIAS_SECURITY_BITS = 256
MAX_ALIAS_BLOCKS = (ALIAS_SECURITY_BITS + 17) // 18


def lift(alias, cc, dt, alpha=ALPHA):
    """Encode <alias, cost centre, document type> as one 9-digit numeral.

    Setup in the paper aborts unless m <= K <= alpha, and the embedding step
    requires K <= alpha because the alias has to fit the account-code field.
    An earlier version of this function wrote `alias % 10000`, which enforces
    nothing: at K' > alpha two aliases silently share a ciphertext class, the
    scheme stops being decryptable and the leakage analysis no longer applies.
    A deployer following the paper's own advice to make K large is exactly who
    would hit it, so the precondition is checked rather than wrapped.
    """
    if not 0 <= alias < alpha:
        raise ValueError(
            "alias {} outside the account-code field of size {}; the budget K "
            "must satisfy K <= alpha (Claim 1)".format(alias, alpha))
    if not 0 <= cc < N_CC_FMT:
        raise ValueError("cost centre {} does not fit three digits".format(cc))
    if not 0 <= dt < N_DT_FMT:
        raise ValueError("document type {} does not fit two digits".format(dt))
    return "{:04d}{:03d}{:02d}".format(alias, cc, dt)


class DuplicateRowError(ValueError):
    """Two rows produced the same RowID. Encryption must abort, not continue:
    a repeated RowID repeats the alias index and, worse, the Pedersen blinding,
    and two commitments with the same blinding reveal the difference of their
    amounts to anyone."""


def row_id(entity, period, partition, ledger, doc, line):
    """Canonical fixed-width RowID shared by both papers.

        id = entity | period | partition | ledger | document | line

    The partition index is its own field and is not folded into the period.
    It has to be: the FF1 tweak is (entity, period, partition), the occupancy
    rule is stated per partition, and the AEAD key is derived per context, so
    two rows of the same period but different partitions are in different
    cryptographic contexts and must have different RowIDs. An earlier version
    of this function took five fields and callers encoded the partition
    inside the period string, which happens to be unique but does not match
    the specification the papers state or let a reader parse the identifier
    back into its fields.

    The byte layout is ``u32|u32|u16|u16|u64|u32`` in network order.  Keeping
    this function as the entry point preserves existing callers while making
    Paper A and Paper B hash exactly the same identifier bytes."""
    return encode_row_id(entity, period, partition, ledger, doc, line)


class RowIDRegistry:
    """Enforces RowID uniqueness at ingestion; the paper's Setup aborts on a
    duplicate and so does this."""

    def __init__(self):
        self._seen = set()

    def check(self, rid):
        if rid in self._seen:
            raise DuplicateRowError("duplicate RowID {!r}".format(rid))
        self._seen.add(rid)
        return rid


def _prf(key, tag, msg):
    """HMAC-SHA256 with a domain-separation tag, so that the alias index, the
    blinding scalar and the AEAD nonce are never derived from the same input."""
    return hmac.new(key, tag + b"\x00" + msg, hashlib.sha256).digest()


def derive_key(master_key, label, context=b""):
    """HKDF-SHA-256 key hierarchy used by the manuscript's KeyGen."""
    if not isinstance(label, bytes) or not isinstance(context, bytes):
        raise TypeError("HKDF label and context must be bytes")
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=b"LC-FPE/v1/" + label + b"/" + context).derive(master_key)


def alias_index(key, doc, line, ka, rid=None):
    """Unbiased index in [0, ka) by rejection sampling on a counter-extended
    PRF. `rid` is the canonical RowID; the (doc, line) pair is kept for
    callers that predate it and is used only when `rid` is None. The old
    implementation reduced a 64-bit output modulo ka, which the proof had to
    carry as an N*k_max*2^-64 term; this removes the term rather than bounding
    it."""
    if ka <= 0:
        raise ValueError("k(a) must be positive")
    if ka == 1:
        return 0
    if rid is None:
        rid = str(doc).encode() + b":" + str(line).encode()
    width = ka.bit_length()
    mask = (1 << width) - 1
    for ctr in range(MAX_ALIAS_BLOCKS):
        h = _prf(key, b"alias", rid + b"#" + ctr.to_bytes(4, "big"))
        # take independent `width`-bit windows from the digest; each is a
        # uniform candidate and is accepted iff it falls below ka
        bits = int.from_bytes(h, "big")
        for _ in range(256 // width):
            cand = bits & mask
            if cand < ka:
                return cand
            bits >>= width
    raise RuntimeError("alias rejection sampler exhausted its security cap")


def blinding_scalar(key, rid, order):
    """Hash-to-scalar for rho in Z_q: 512 bits of PRF output reduced mod q,
    which is within 2^-256 of uniform, under its own domain tag."""
    h = _prf(key, b"blind/0", rid) + _prf(key, b"blind/1", rid)
    return int.from_bytes(h, "big") % order


NONCE_BOUNDS = dict(field=1, ledger=7, doc=64, line=24)   # bits; 1+7+64+24 = 96


def aead_nonce(field, ledger, doc, line):
    """96-bit AES-GCM nonce as an INJECTIVE encoding of (field, ledger, doc,
    line): field flag (0 = amount, 1 = text) | ledger index (7 bits) |
    document number (64 bits) | line number (24 bits). Two rows of one
    (entity, period, partition) context -- which selects the AEAD key -- get
    distinct nonces exactly when their (ledger, doc, line) differ, which the
    RowID registry enforces; the two fields of one row differ in the flag.
    Uniqueness therefore follows from the encoding, not from a PRF: an earlier
    version truncated a PRF output to 96 bits and claimed uniqueness from the
    uniqueness of its input, which a truncation does not give."""
    if not (0 <= field < 2 and 0 <= ledger < 2 ** 7 and 0 <= doc < 2 ** 64
            and 0 <= line < 2 ** 24):
        raise ValueError("nonce field out of its declared range")
    n = (field << 95) | (ledger << 88) | (doc << 24) | line
    return n.to_bytes(12, "big")


# --------------------------------------------------------------- benchmark
def bench(N=20000, m=200, seed=7):
    rng = np.random.default_rng(seed)
    K = 10000 - m
    w = 1.0 / np.arange(1, m + 1)
    f = w / w.sum()
    k = alloc_hamilton(f, K)
    Kp = int(k.sum())
    start = np.concatenate([[0], np.cumsum(k)[:-1]])
    acct = rng.choice(m, size=N, p=f)
    cc = rng.integers(0, 100, N)
    dt = rng.integers(0, 10, N)
    amt = rng.integers(-10 ** 7, 10 ** 7, N)

    key_ff1 = os.urandom(32)
    key_alias = os.urandom(32)
    key_gcm = os.urandom(32)
    ff1 = FF1(key_ff1, radix=10)
    print("  rows = {:,}   m = {}   K prime = {}".format(N, m, Kp), flush=True)

    # correctness check on FF1 (format preservation + determinism)
    probe = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    out = ff1.encrypt(probe, tweak=b"ENT1_2026M09")
    ok_fmt = (len(out) == len(probe)
              and all(0 <= z < 10 for z in out)
              and out == ff1.encrypt(probe, tweak=b"ENT1_2026M09"))
    print("  FF1 format-preserving + deterministic : {}".format(ok_fmt), flush=True)

    # --- FPAE alias selection
    t0 = time.time()
    alias = np.empty(N, dtype=np.int64)
    for i in range(N):
        a = int(acct[i])
        alias[i] = start[a] + alias_index(key_alias, i, 0, int(k[a]))
    t_alias = (time.time() - t0) / N
    print("  FPAE alias select : {:8.1f} us/row  {:>12,.0f} rows/s"
          .format(t_alias * 1e6, 1 / t_alias), flush=True)

    # --- FF1 on the lifted domain: 4 + 3 + 2 = 9 decimal digits = 10^9
    S = 2000
    t0 = time.time()
    for i in range(S):
        s = lift(int(alias[i]), int(cc[i]), int(dt[i]))
        ff1.encrypt([int(ch) for ch in s], tweak=b"ENT1_2026M09")
    t_ff1 = (time.time() - t0) / S
    print("  FF1 encrypt       : {:8.1f} us/row  {:>12,.0f} rows/s"
          .format(t_ff1 * 1e6, 1 / t_ff1), flush=True)

    # --- Paillier
    print("  building Paillier 2048-bit key and r^n pool ...", flush=True)
    pai = Paillier(2048, pool=4096)
    print("    offline pool of 4096 r^n : {:.1f}s (one-off)"
          .format(pai.pool_time), flush=True)
    S2 = 5000
    t0 = time.time()
    ct = [pai.enc(int(amt[i])) for i in range(S2)]
    t_pai = (time.time() - t0) / S2
    print("  Paillier encrypt  : {:8.1f} us/row  {:>12,.0f} rows/s"
          .format(t_pai * 1e6, 1 / t_pai), flush=True)

    t0 = time.time()
    acc = ct[0]
    for c in ct[1:]:
        acc = pai.add(acc, c)
    t_add = (time.time() - t0) / (S2 - 1)
    print("  Paillier hom. add : {:8.1f} us      {:>12,.0f} adds/s"
          .format(t_add * 1e6, 1 / t_add), flush=True)

    t0 = time.time()
    got = pai.dec(acc)
    t_dec = time.time() - t0
    exp = int(sum(int(amt[i]) for i in range(S2)))
    print("  Paillier decrypt sum : {:.2f} ms   correct = {}"
          .format(t_dec * 1e3, got == exp), flush=True)

    # --- AES-256-GCM baseline
    t0 = time.time()
    for i in range(S2):
        nonce = os.urandom(12)
        e = Cipher(algorithms.AES(key_gcm), modes.GCM(nonce)).encryptor()
        e.update("{}|{}|{}|{}".format(int(acct[i]), int(cc[i]),
                                      int(dt[i]), int(amt[i])).encode())
        e.finalize()
    t_gcm = (time.time() - t0) / S2
    print("  AES-256-GCM       : {:8.1f} us/row  {:>12,.0f} rows/s   <-- baseline"
          .format(t_gcm * 1e6, 1 / t_gcm), flush=True)

    # --- storage
    plain = 4 + 3 + 2 + 8
    lcfpe = 4 + 3 + 2 + pai.nbytes
    print("", flush=True)
    print("  STORAGE per journal line", flush=True)
    print("    plaintext codes+int64 amount : {:>6} bytes".format(plain), flush=True)
    print("    LC-FPE codes (format kept)   : {:>6} bytes".format(4 + 3 + 2),
          flush=True)
    print("    LC-FPE Paillier amount       : {:>6} bytes  <-- {:.0f}x an int64"
          .format(pai.nbytes, pai.nbytes / 8), flush=True)
    print("    total {} -> {} bytes = {:.1f}x expansion"
          .format(plain, lcfpe, lcfpe / plain), flush=True)

    per_row = t_ff1 + t_pai + t_alias
    print("", flush=True)
    print("  END-TO-END", flush=True)
    print("    LC-FPE  {:.0f} us/row -> {:,.0f} rows/s  ({:.1f}s for {:,} rows)"
          .format(per_row * 1e6, 1 / per_row, N * per_row, N), flush=True)
    print("    AES-GCM {:.0f} us/row -> {:,.0f} rows/s"
          .format(t_gcm * 1e6, 1 / t_gcm), flush=True)
    print("    slowdown vs AES-GCM : {:,.0f}x".format(per_row / t_gcm), flush=True)
    print("    dominant cost       : {}".format(
        "Paillier" if t_pai > t_ff1 else "FF1"), flush=True)


if __name__ == "__main__":
    print("=" * 78, flush=True)
    print("LC-FPE REFERENCE IMPLEMENTATION -- PERFORMANCE", flush=True)
    print("=" * 78, flush=True)
    bench()
