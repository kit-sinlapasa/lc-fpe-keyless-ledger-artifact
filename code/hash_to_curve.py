#!/usr/bin/env python3
"""
RFC 9380 hash-to-curve for NIST P-256, suite P256_XMD:SHA-256_SSWU_RO_.

Why this exists. The Bulletproofs implementation derived its generators by
try-and-increment hashing: hash a label, read the result as an x-coordinate,
and retry until one lands on the curve. That is sound for the purpose --
generators need only have unknown relative discrete logarithm, which a fixed
label provides -- but it is not the standard construction, and Paper B listed
it as one of two places where the proof system departs from what anyone would
deploy. This module removes that departure by implementing the suite RFC 9380
specifies for P-256, and it is checked against the RFC's own test vectors
rather than against our expectations of it.

The curve parameters are written out from FIPS 186-4 / SEC 2 rather than read
from a private attribute of the arithmetic library, which was the other
departure. `self_test` re-derives them: it checks the generator satisfies the
curve equation and has the stated order, so the constants are verified here
and not merely copied.

Reference: RFC 9380, "Hashing to Elliptic Curves", sections 5, 6.6.2 and 8.2,
test vectors in Appendix J.1.1.
"""
import hashlib

from Crypto.PublicKey import ECC

# --- P-256 domain parameters, FIPS 186-4 D.1.2.3 / SEC 2 2.4.2 --------------
P = 0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff
N = 0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551
A = P - 3
B = 0x5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b
GX = 0x6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296
GY = 0x4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5

# --- suite parameters, RFC 9380 section 8.2 ---------------------------------
Z = P - 10          # Z = -10
L = 48              # ceil((ceil(log2(p)) + k) / 8) with k = 128
M = 1               # extension degree
H_BLOCK = 64        # SHA-256 input block, s_in_bytes
H_OUT = 32          # SHA-256 output, b_in_bytes

_SQRT_EXP = (P + 1) // 4      # p = 3 mod 4
_SQR_EXP = (P - 1) // 2


def _sha256(b):
    return hashlib.sha256(b).digest()


def expand_message_xmd(msg: bytes, dst: bytes, len_in_bytes: int) -> bytes:
    """RFC 9380 section 5.3.1."""
    if len(dst) > 255:
        raise ValueError("DST longer than 255 bytes needs the hashed form")
    ell = -(-len_in_bytes // H_OUT)
    if ell > 255 or len_in_bytes > 65535:
        raise ValueError("len_in_bytes out of range")
    dst_prime = dst + bytes([len(dst)])
    msg_prime = (bytes(H_BLOCK) + msg + len_in_bytes.to_bytes(2, "big")
                 + b"\x00" + dst_prime)
    b_0 = _sha256(msg_prime)
    b_i = _sha256(b_0 + b"\x01" + dst_prime)
    out = bytearray(b_i)
    for i in range(2, ell + 1):
        x = bytes(a ^ b for a, b in zip(b_0, b_i))
        b_i = _sha256(x + bytes([i]) + dst_prime)
        out += b_i
    return bytes(out[:len_in_bytes])


def hash_to_field(msg: bytes, count: int, dst: bytes):
    """RFC 9380 section 5.2, for m = 1."""
    ub = expand_message_xmd(msg, dst, count * M * L)
    return [int.from_bytes(ub[i * L:(i + 1) * L], "big") % P
            for i in range(count)]


def _inv0(x):
    return pow(x, P - 2, P)


def _is_square(x):
    return x == 0 or pow(x, _SQR_EXP, P) == 1


def _sqrt(x):
    return pow(x, _SQRT_EXP, P)


def _sgn0(x):
    return x % 2


def map_to_curve_simple_swu(u):
    """RFC 9380 section 6.6.2, the plain form (A != 0, B != 0)."""
    u2 = u * u % P
    tv1 = _inv0((Z * Z % P) * (u2 * u2 % P) % P + Z * u2 % P)
    x1 = (P - B) * _inv0(A) % P * ((1 + tv1) % P) % P
    if tv1 == 0:
        x1 = B * _inv0(Z * A % P) % P
    gx1 = (pow(x1, 3, P) + A * x1 + B) % P
    x2 = Z * u2 % P * x1 % P
    gx2 = (pow(x2, 3, P) + A * x2 + B) % P
    if _is_square(gx1):
        x, y = x1, _sqrt(gx1)
    else:
        x, y = x2, _sqrt(gx2)
    if _sgn0(u) != _sgn0(y):
        y = P - y
    return x, y


def hash_to_curve(msg: bytes, dst: bytes):
    """RFC 9380 section 3, the random-oracle (RO) variant. Cofactor is 1 for
    P-256, so clear_cofactor is the identity and is omitted."""
    u = hash_to_field(msg, 2, dst)
    x0, y0 = map_to_curve_simple_swu(u[0])
    x1, y1 = map_to_curve_simple_swu(u[1])
    r = (ECC.EccPoint(x0, y0, curve="p256")
         + ECC.EccPoint(x1, y1, curve="p256"))
    return r


# ---------------------------------------------------------------- self-test
DST_RFC = b"QUUX-V01-CS02-with-P256_XMD:SHA-256_SSWU_RO_"

# RFC 9380 Appendix J.1.1, read from the RFC text itself
VECTORS = [
    (b"",
     "2c15230b26dbc6fc9a37051158c95b79656e17a1a920b11394ca91c44247d3e4",
     "8a7a74985cc5c776cdfe4b1f19884970453912e9d31528c060be9ab5c43e8415",
     "ad5342c66a6dd0ff080df1da0ea1c04b96e0330dd89406465eeba11582515009",
     "8c0f1d43204bd6f6ea70ae8013070a1518b43873bcd850aafa0a9e220e2eea5a"),
    (b"abc",
     "0bb8b87485551aa43ed54f009230450b492fead5f1cc91658775dac4a3388a0f",
     "5c41b3d0731a27a7b14bc0bf0ccded2d8751f83493404c84a88e71ffd424212e",
     "afe47f2ea2b10465cc26ac403194dfb68b7f5ee865cda61e9f3e07a537220af1",
     "379a27833b0bfe6f7bdca08e1e83c760bf9a338ab335542704edcd69ce9e46e0"),
    (b"abcdef0123456789",
     "65038ac8f2b1def042a5df0b33b1f4eca6bff7cb0f9c6c1526811864e544ed80",
     "cad44d40a656e7aff4002a8de287abc8ae0482b5ae825822bb870d6df9b56ca3",
     "0fad9d125a9477d55cf9357105b0eb3a5c4259809bf87180aa01d651f53d312c",
     "b68597377392cd3419d8fcc7d7660948c8403b19ea78bbca4b133c9d2196c0fb"),
]


def self_test(verbose=True):
    ok = True

    # the constants are checked, not trusted
    lhs = (GY * GY) % P
    rhs = (pow(GX, 3, P) + A * GX + B) % P
    curve_ok = lhs == rhs
    g = ECC.EccPoint(GX, GY, curve="p256")
    order_ok = (g * N).is_point_at_infinity()
    if verbose:
        print("  generator on the curve                : %s" % curve_ok)
        print("  n*G is the point at infinity          : %s" % order_ok)
    ok &= curve_ok and order_ok

    for msg, px, py, u0, u1 in VECTORS:
        u = hash_to_field(msg, 2, DST_RFC)
        pt = hash_to_curve(msg, DST_RFC)
        gu = u[0] == int(u0, 16) and u[1] == int(u1, 16)
        gp = int(pt.x) == int(px, 16) and int(pt.y) == int(py, 16)
        ok &= gu and gp
        if verbose:
            print("  msg=%-18r u ok %-5s  P ok %-5s"
                  % (msg.decode() or "", gu, gp))
    return ok


if __name__ == "__main__":
    print("=" * 70)
    print("RFC 9380 P256_XMD:SHA-256_SSWU_RO_ against the RFC's own vectors")
    print("=" * 70)
    good = self_test()
    print()
    print("  ALL VECTORS MATCH" if good else "  MISMATCH")
    raise SystemExit(0 if good else 1)
