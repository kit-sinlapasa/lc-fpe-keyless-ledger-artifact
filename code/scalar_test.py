import time, os
from Crypto.PublicKey import ECC
from Crypto.Math.Numbers import Integer
c = ECC._curves["p256"]
G = ECC.EccPoint(int(c.Gx), int(c.Gy), curve="p256")
ORDER = int(c.order)
S = 400

def timeit(vals, label):
    t0 = time.time()
    for v in vals:
        G * Integer(v)
    dt = (time.time() - t0) / len(vals)
    print("  {:<34} {:8.1f} us".format(label, dt * 1e6), flush=True)
    return dt

print("SCALAR SIZE vs EC SCALAR-MULT COST (P-256)", flush=True)
small = [abs(int.from_bytes(os.urandom(5), "big")) % 10**7 for _ in range(S)]
neg   = [(-(x)) % ORDER for x in small]
offs  = [x + 2**40 for x in small]
full  = [int.from_bytes(os.urandom(32), "big") % ORDER for _ in range(S)]
a = timeit(small, "small positive (~24 bit)")
b = timeit(neg,   "negative -> x mod n (~256 bit)")
d = timeit(offs,  "offset-encoded (~41 bit)")
e = timeit(full,  "full random 256-bit")
print("", flush=True)
print("  negative / small      = {:.1f}x".format(b / a), flush=True)
print("  offset-encoded / small= {:.1f}x".format(d / a), flush=True)
print("  full / small          = {:.1f}x".format(e / a), flush=True)
