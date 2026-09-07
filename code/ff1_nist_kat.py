#!/usr/bin/env python3
"""
Known-answer test of our FF1 against the official NIST sample values.

Source: NIST, "Block Cipher Modes of Operation: FF1 Method for
Format-Preserving Encryption", sample values accompanying SP 800-38G
(FF1samples.pdf, csrc.nist.gov).  All nine samples are reproduced here: three
key sizes (AES-128/192/256) x three settings (radix 10 empty tweak, radix 10
with tweak, radix 36 with tweak).

Both directions are checked.  Nothing in either paper depends on FF1 being
correct in the leakage sense -- FF1 under a fixed tweak is a bijection, so the
clustering results hold for any bijection -- but the implementation section
claims a working FF1, and that claim should rest on the standard's own vectors
rather than on a round-trip test against ourselves.
"""
from lcfpe_impl import FF1

ALPH = "0123456789abcdefghijklmnopqrstuvwxyz"

K128 = "2B7E151628AED2A6ABF7158809CF4F3C"
K192 = K128 + "EF4359D8D580AA4F"
K256 = K192 + "7F036D6F04FC6A94"

T0 = ""
T1 = "39383736353433323130"
T2 = "3737373770717273373737"

# (key hex, radix, tweak hex, plaintext, expected ciphertext)
SAMPLES = [
    (K128, 10, T0, "0123456789", "2433477484"),
    (K128, 10, T1, "0123456789", "6124200773"),
    (K128, 36, T2, "0123456789abcdefghi", "a9tv40mll9kdu509eum"),
    (K192, 10, T0, "0123456789", "2830668132"),
    (K192, 10, T1, "0123456789", "2496655549"),
    (K192, 36, T2, "0123456789abcdefghi", "xbj3kv35jrawxv32ysr"),
    (K256, 10, T0, "0123456789", "6657667009"),
    (K256, 10, T1, "0123456789", "1001623463"),
    (K256, 36, T2, "0123456789abcdefghi", "xs8a0azh2avyalyzuwd"),
]


def nums(s, radix):
    return [ALPH.index(ch) for ch in s]


def text(ns):
    return "".join(ALPH[n] for n in ns)


print("=" * 78, flush=True)
print("FF1 KNOWN-ANSWER TEST against the NIST SP 800-38G sample values",
      flush=True)
print("=" * 78, flush=True)
print("  {:<3} {:<10} {:>5} {:<22} {:<10} {}".format(
    "#", "key", "radix", "ciphertext", "encrypt", "decrypt"), flush=True)

fails = 0
for i, (kh, radix, th, pt, ct) in enumerate(SAMPLES, 1):
    key = bytes.fromhex(kh)
    tweak = bytes.fromhex(th)
    f = FF1(key, radix=radix)
    got = text(f.encrypt(nums(pt, radix), tweak=tweak))
    enc_ok = got == ct
    try:
        back = text(f.decrypt(nums(got, radix), tweak=tweak))
        dec_ok = back == pt
    except AttributeError:
        dec_ok = None
    if not enc_ok or dec_ok is False:
        fails += 1
    print("  {:<3} AES-{:<6} {:>5} {:<22} {:<10} {}".format(
        i, len(key) * 8, radix, got,
        "ok" if enc_ok else "MISMATCH",
        "ok" if dec_ok else ("no decrypt()" if dec_ok is None else "MISMATCH")),
        flush=True)
    if not enc_ok:
        print("        expected {}".format(ct), flush=True)

print(flush=True)
print("  {} of {} samples reproduced".format(len(SAMPLES) - fails,
                                             len(SAMPLES)), flush=True)
raise SystemExit(1 if fails else 0)
