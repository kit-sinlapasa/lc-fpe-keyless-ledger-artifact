#!/usr/bin/env python3
"""Small end-to-end fixture for Paper B's anchored sampling path.

This is deliberately separate from the long benchmark.  It checks that one
canonical record list is used for the Merkle root, chain head, signed anchor,
beacon-derived sample, interval proof, inclusion path, and row openings.
It also checks the explicit future-round and close-time guards in the protocol
specification.
"""
import bisect
import hashlib
import struct

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from bulletproofs import BP, Q, enc, mul


H = hashlib.sha256
OMEGA = 1 << 40
N = 16
S = 4
CURRENT_ROUND = 100
NOW = 150


def node(left, right):
    return H(b"\x01" + left + right).digest()


def leaf(record):
    return H(b"\x00" + record).digest()


def split_point(n):
    return 1 << ((n - 1).bit_length() - 1)


def merkle_root(records):
    leaves = [leaf(r) for r in records]
    if len(leaves) == 1:
        return leaves[0]
    k = split_point(len(leaves))
    return node(merkle_root(records[:k]), merkle_root(records[k:]))


def chain_head(records):
    h = b"\x00" * 32
    for record in records:
        h = H(h + record).digest()
    return h


def merkle_path(records, index):
    if len(records) == 1:
        return []
    k = split_point(len(records))
    if index < k:
        return merkle_path(records[:k], index) + [
            (merkle_root(records[k:]), True)]
    return merkle_path(records[k:], index - k) + [
        (merkle_root(records[:k]), False)]


def verify_path(record, index, path, root):
    current = leaf(record)
    for sibling, sibling_on_right in path:
        current = (node(current, sibling) if sibling_on_right
                   else node(sibling, current))
    return current == root


class Beacon:
    """Deterministic stand-in for an externally authenticated beacon."""

    def value(self, round_number):
        return H(b"paperb-e2e-beacon" + struct.pack(">I", round_number)).digest()

    def opens_at(self, round_number):
        return round_number + 100


def sample_below(x, bound):
    limit = (1 << 256) // bound * bound
    return None if x >= limit else x % bound


def sample_distinct(seed, bound, count):
    if count < 1 or count > bound:
        raise ValueError("invalid sample size")
    out, seen, counter = [], set(), 0
    while len(out) < count:
        x = int.from_bytes(H(seed + struct.pack(">I", counter)).digest(),
                           "big")
        counter += 1
        value = sample_below(x, bound)
        if value is not None and value not in seen:
            seen.add(value)
            out.append(value)
    return sorted(out)


def record_bytes(index, amount, account, C, A, D, K, side_link):
    return (struct.pack(">IqI", index, amount, account) + enc(C) + enc(A)
            + enc(D) + enc(K) + int(side_link).to_bytes(32, "big"))


def make_fixture(tau=100, round_number=200, understate=False):
    bp = BP(32, 2 * S)
    amounts = [3, -2, 5, -6, 4, -4, 7, -7,
               2, -3, 6, -5, 1, -2, 8, -7]
    accounts = [1000 + i for i in range(N)]
    rho = [11 + i for i in range(N)]
    sigma = [101 + i for i in range(N)]
    eta = [201 + i for i in range(N)]
    zeta = [301 + i for i in range(N)]
    d = [max(v, 0) for v in amounts]
    k = [max(-v, 0) for v in amounts]
    if understate:
        d[0] -= 1
        k[1] -= 1
    T = sum(d)
    assert sum(amounts) == 0 and sum(k) == T

    C = [bp.commit(amounts[i] + OMEGA, rho[i]) for i in range(N)]
    A = [bp.commit(accounts[i], sigma[i]) for i in range(N)]
    D = [bp.commit(d[i], eta[i]) for i in range(N)]
    K = [bp.commit(k[i], zeta[i]) for i in range(N)]
    side_link = [(eta[i] - zeta[i] - rho[i]) % Q for i in range(N)]
    records = [record_bytes(i, amounts[i], accounts[i], C[i], A[i], D[i], K[i],
                            side_link[i])
               for i in range(N)]

    root = merkle_root(records)
    head = chain_head(records)
    env = struct.pack(">IIQI", N, T, tau, round_number) + root + head
    sk = Ed25519PrivateKey.from_private_bytes(H(b"paperb-e2e-firm").digest())
    anchor = env + sk.sign(env)
    return dict(bp=bp, amounts=amounts, accounts=accounts, rho=rho,
                sigma=sigma, eta=eta, zeta=zeta, d=d, k=k, T=T, C=C, A=A,
                D=D, K=K, side_link=side_link, records=records, root=root,
                head=head, env=env,
                anchor=anchor, public_key=sk.public_key(), tau=tau,
                round=round_number)


def parse_anchor(fixture, anchor, current_round=CURRENT_ROUND):
    if len(anchor) < 64 + 20 + 64:
        return False
    env, signature = anchor[:-64], anchor[-64:]
    try:
        fixture["public_key"].verify(signature, env)
    except Exception:
        return False
    n_rows, total, tau, round_number = struct.unpack(">IIQI", env[:20])
    root, head = env[20:52], env[52:84]
    if n_rows != N or total != fixture["T"]:
        return False
    if round_number <= current_round:
        return False
    if tau >= Beacon().opens_at(round_number):
        return False
    return (root == fixture["root"] and head == fixture["head"]
            and tau == fixture["tau"] and round_number == fixture["round"])


def verify_period(fixture, records=None, anchor=None, current_round=CURRENT_ROUND):
    records = fixture["records"] if records is None else records
    anchor = fixture["anchor"] if anchor is None else anchor
    if not parse_anchor(fixture, anchor, current_round):
        return False
    if merkle_root(records) != fixture["root"] or chain_head(records) != fixture["head"]:
        return False
    bp = fixture["bp"]
    return all(
        fixture["D"][i] + mul(fixture["K"][i], Q - 1)
        + mul(fixture["C"][i], Q - 1) + mul(bp.G, OMEGA)
        == mul(bp.H, fixture["side_link"][i]) for i in range(N))


def prepare_sample(fixture):
    bp = fixture["bp"]
    beacon = Beacon()
    seed = fixture["anchor"] + beacon.value(fixture["round"]) + b"mus"
    units = sample_distinct(seed, fixture["T"], S)
    cumulative_values, cumulative_points = [], []
    value_total, point_total = 0, mul(bp.G, 0)
    for point, value in zip(fixture["D"], fixture["d"]):
        point_total = point_total + point
        value_total += value
        cumulative_points.append(point_total)
        cumulative_values.append(value_total)

    indices, interval_values, interval_blinds = [], [], []
    for unit in units:
        j = bisect.bisect_right(cumulative_values, unit)
        previous_value = cumulative_values[j - 1] if j else 0
        previous_point = cumulative_points[j - 1] if j else mul(bp.G, 0)
        indices.append(j)
        interval_values.extend([unit - previous_value,
                                cumulative_values[j] - unit - 1])
        previous_blind = sum(fixture["eta"][:j]) % Q
        current_blind = sum(fixture["eta"][:j + 1]) % Q
        interval_blinds.extend([(-previous_blind) % Q, current_blind])

    commitments = [bp.commit(interval_values[i], interval_blinds[i])
                   for i in range(len(interval_values))]
    proof = bp.prove(interval_values, interval_blinds)
    return dict(beacon=beacon, units=units, indices=indices,
                cumulative_values=cumulative_values,
                cumulative_points=cumulative_points, commitments=commitments,
                proof=proof)


def verify_sample(fixture, sample, claimed_indices=None):
    if not verify_period(fixture):
        return False
    bp = fixture["bp"]
    if bp.verify(sample["commitments"], sample["proof"])[0] is not True:
        return False
    indices = sample["indices"] if claimed_indices is None else claimed_indices
    if len(indices) != S:
        return False
    for n, (unit, j) in enumerate(zip(sample["units"], indices)):
        if not 0 <= j < N:
            return False
        previous_value = sample["cumulative_values"][j - 1] if j else 0
        current_value = sample["cumulative_values"][j]
        if not previous_value <= unit < current_value:
            return False
        previous_point = (sample["cumulative_points"][j - 1]
                          if j else mul(bp.G, 0))
        current_point = sample["cumulative_points"][j]
        D1 = mul(bp.G, unit) + mul(previous_point, Q - 1)
        D2 = current_point + mul(bp.G, (-(unit + 1)) % Q)
        if D1 != sample["commitments"][2 * n] or D2 != sample["commitments"][2 * n + 1]:
            return False
        if not verify_path(fixture["records"][j], j,
                           merkle_path(fixture["records"], j), fixture["root"]):
            return False
        if bp.commit(fixture["amounts"][j] + OMEGA, fixture["rho"][j]) != fixture["C"][j]:
            return False
        if bp.commit(fixture["accounts"][j], fixture["sigma"][j]) != fixture["A"][j]:
            return False
        if bp.commit(fixture["d"][j], fixture["eta"][j]) != fixture["D"][j]:
            return False
        if bp.commit(fixture["k"][j], fixture["zeta"][j]) != fixture["K"][j]:
            return False
    return True


def main():
    fixture = make_fixture()
    sample = prepare_sample(fixture)
    wrong_indices = list(sample["indices"])
    wrong_indices[0] = ((wrong_indices[0] + 1) % N)
    tampered_records = list(fixture["records"])
    tampered_records[0] += b"x"
    late_fixture = make_fixture(tau=350)
    understated_fixture = make_fixture(understate=True)

    checks = {
        "honest end-to-end path accepts": verify_sample(fixture, sample),
        "one record list reproduces anchor root and chain": verify_period(fixture),
        "tampered record is rejected": not verify_period(fixture, tampered_records),
        "past beacon round is rejected": not verify_period(fixture, current_round=200),
        "anchor after beacon opening time is rejected": not verify_period(late_fixture),
        "wrong sampled row is rejected": not verify_sample(fixture, sample, wrong_indices),
        "anchored understated side is rejected": not verify_period(understated_fixture),
    }
    for label, result in checks.items():
        print("  {:52} : {}".format(label, result))
    print("  all checks pass                                     : {}".format(
        all(checks.values())))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
