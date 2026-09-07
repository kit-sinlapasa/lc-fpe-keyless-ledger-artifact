#!/usr/bin/env python3
"""Role-separated, small end-to-end fixture for Paper B.

The writer closes one canonical record set, a custodian timestamps that exact
anchor before the chosen beacon round opens, and a later keyless verifier
checks the period and a monetary-unit sample. Plaintext witnesses are kept
out of the verifier-facing period object.
"""
import bisect
import copy
import hashlib
import struct

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from bulletproofs import BP, Q, enc, mul
from paperb_protocol import (
    P256_RECORD_BYTES,
    ROW_ID_BYTES,
    decode_row_id,
    encode_anchor_envelope,
    encode_record,
    encode_row_id,
    make_anchor,
    make_receipt,
    verify_anchor,
    verify_receipt,
)


H = hashlib.sha256
OMEGA = 1 << 40
N = 16
S = 4
ENTITY = 1
PERIOD = 202609
PARTITION = 0
LEDGER = 0
PREVIOUS_ANCHOR_HASH = H(b"paperb-e2e/genesis").digest()
CLOSE_TIME = 100
RECEIPT_TIME = 120
BEACON_ROUND = 200
AUDIT_TIME = 350


def node(left, right):
    return H(b"\x01" + left + right).digest()


def leaf(record):
    return H(b"\x00" + record).digest()


def split_point(n):
    return 1 << ((n - 1).bit_length() - 1)


def merkle_root(records):
    leaves = [leaf(record) for record in records]
    if len(leaves) == 1:
        return leaves[0]
    k = split_point(len(leaves))
    return node(merkle_root(records[:k]), merkle_root(records[k:]))


def chain_head(records):
    head = b"\x00" * 32
    for record in records:
        head = H(head + record).digest()
    return head


def merkle_path(records, index):
    if len(records) == 1:
        return []
    k = split_point(len(records))
    if index < k:
        return merkle_path(records[:k], index) + [
            (merkle_root(records[k:]), True)]
    return merkle_path(records[k:], index - k) + [
        (merkle_root(records[:k]), False)]


def verify_path(record, path, root):
    current = leaf(record)
    for sibling, sibling_on_right in path:
        current = (node(current, sibling) if sibling_on_right
                   else node(sibling, current))
    return current == root


def sum_points(points, generator):
    total = mul(generator, 0)
    for point in points:
        total = total + point
    return total


class Beacon:
    """Authenticated deterministic stand-in; no live network is claimed."""

    DOMAIN = b"PAPERB-E2E-BEACON/1"

    def __init__(self):
        self._private_key = Ed25519PrivateKey.from_private_bytes(
            H(b"paperb-e2e-beacon-key").digest())
        self.public_key = self._private_key.public_key()

    @staticmethod
    def opens_at(round_number):
        return round_number + 100

    def publish(self, round_number):
        opening = self.opens_at(round_number)
        value = H(self.DOMAIN + struct.pack(">I", round_number)).digest()
        message = (self.DOMAIN + struct.pack(">IQ", round_number, opening)
                   + value)
        return dict(round=round_number, opening=opening, value=value,
                    signature=self._private_key.sign(message))

    def verify(self, statement, expected_round, audit_time):
        try:
            if statement["round"] != expected_round:
                return False
            if statement["opening"] != self.opens_at(expected_round):
                return False
            if audit_time < statement["opening"]:
                return False
            message = (self.DOMAIN
                       + struct.pack(">IQ", statement["round"],
                                     statement["opening"])
                       + statement["value"])
            self.public_key.verify(statement["signature"], message)
            return True
        except (KeyError, TypeError, ValueError):
            return False
        except Exception:
            return False


def sample_below(value, bound):
    limit = (1 << 256) // bound * bound
    return None if value >= limit else value % bound


def sample_distinct(seed, bound, count):
    if count < 1 or count > bound:
        raise ValueError("invalid sample size")
    out, seen, counter = [], set(), 0
    while len(out) < count:
        candidate = int.from_bytes(
            H(seed + struct.pack(">I", counter)).digest(), "big")
        counter += 1
        value = sample_below(candidate, bound)
        if value is not None and value not in seen:
            seen.add(value)
            out.append(value)
    return sorted(out)


def parameters_valid(row_count, total, sample_size):
    return (row_count > 0
            and row_count * (1 << 64) < Q
            and (row_count + 1) * (1 << 32) + OMEGA < Q
            and (1 << 64) + (1 << 33) + OMEGA < Q
            and 0 < total < row_count * (1 << 32)
            and total < (1 << 256)
            and 1 <= sample_size <= total)


def encrypted_fields(row_id, amount, account, key):
    nonce = H(b"paperb-e2e/nonce" + row_id).digest()[:12]
    payload = struct.pack(">qI", amount, account)
    value = nonce + AESGCM(key).encrypt(nonce, payload, row_id)
    if len(value) != 40:
        raise AssertionError("encrypted-field width changed")
    return value


def make_fixture():
    amount_bp = BP(64, N)
    side_bp = BP(32, 2 * N)
    sample_bp = BP(32, 2 * S)
    amounts = [3, -2, 5, -6, 4, -4, 7, -7,
               2, -3, 6, -5, 1, -2, 8, -7]
    accounts = [1000 + i % 4 for i in range(N)]
    rho = [11 + i for i in range(N)]
    sigma = [101 + i for i in range(N)]
    eta = [201 + i for i in range(N)]
    zeta = [301 + i for i in range(N)]
    debit = [max(value, 0) for value in amounts]
    credit = [max(-value, 0) for value in amounts]
    total = sum(debit)
    assert sum(amounts) == 0 and sum(credit) == total

    shifted = [value + OMEGA for value in amounts]
    C = [amount_bp.commit(shifted[i], rho[i]) for i in range(N)]
    A = [amount_bp.commit(accounts[i], sigma[i]) for i in range(N)]
    D = [amount_bp.commit(debit[i], eta[i]) for i in range(N)]
    K = [amount_bp.commit(credit[i], zeta[i]) for i in range(N)]
    side_link = [(eta[i] - zeta[i] - rho[i]) % Q for i in range(N)]
    data_key = H(b"paperb-e2e/data-key").digest()
    row_ids = [encode_row_id(ENTITY, PERIOD, PARTITION, LEDGER,
                             10000 + i // 2, i % 2)
               for i in range(N)]
    encrypted = [encrypted_fields(row_ids[i], amounts[i], accounts[i],
                                  data_key) for i in range(N)]
    records = [encode_record(row_ids[i], encrypted[i], enc(C[i]), enc(A[i]),
                             enc(D[i]), enc(K[i]),
                             side_link[i].to_bytes(32, "big"))
               for i in range(N)]
    assert all(len(record) == P256_RECORD_BYTES for record in records)

    root, head = merkle_root(records), chain_head(records)
    firm_key = Ed25519PrivateKey.from_private_bytes(
        H(b"paperb-e2e-firm-key").digest())
    custodian_key = Ed25519PrivateKey.from_private_bytes(
        H(b"paperb-e2e-custodian-key").digest())
    env = encode_anchor_envelope(
        ENTITY, PERIOD, N, root, head, PREVIOUS_ANCHOR_HASH,
        CLOSE_TIME, BEACON_ROUND)
    anchor = make_anchor(env, firm_key)
    receipt = make_receipt(anchor, RECEIPT_TIME, custodian_key)

    public = dict(
        entity=ENTITY, period=PERIOD, row_count=N, sample_size=S,
        previous_anchor_hash=PREVIOUS_ANCHOR_HASH, records=records,
        row_ids=row_ids, encrypted=encrypted, C=C, A=A, D=D, K=K,
        side_link=side_link, total=total, sum_rho=sum(rho) % Q,
        sum_eta_minus_rho=(sum(eta) - sum(rho)) % Q,
        sum_zeta_plus_rho=(sum(zeta) + sum(rho)) % Q,
        amount_proof=amount_bp.prove(shifted, rho),
        side_proof=side_bp.prove(debit + credit, eta + zeta),
        amount_bp=amount_bp, side_bp=side_bp, sample_bp=sample_bp,
        anchor=anchor, receipt=receipt,
        firm_public_key=firm_key.public_key(),
        custodian_public_key=custodian_key.public_key(),
    )
    witness = dict(amounts=amounts, accounts=accounts, rho=rho, sigma=sigma,
                   eta=eta, zeta=zeta, debit=debit, credit=credit)
    signing = dict(firm=firm_key, custodian=custodian_key)
    return public, witness, signing


def record_matches_public(period, index, record):
    try:
        if len(record) != P256_RECORD_BYTES:
            return False
        row_id = record[:ROW_ID_BYTES]
        entity, period_code, partition, ledger, _, _ = decode_row_id(row_id)
        if (entity != period["entity"] or period_code != period["period"]
                or partition != PARTITION or ledger != LEDGER):
            return False
        expected = encode_record(
            row_id, period["encrypted"][index], enc(period["C"][index]),
            enc(period["A"][index]), enc(period["D"][index]),
            enc(period["K"][index]),
            period["side_link"][index].to_bytes(32, "big"))
        return record == expected
    except (KeyError, ValueError, IndexError):
        return False


def verify_side_links(period):
    bp = period["amount_bp"]
    return all(
        period["D"][i] + mul(period["K"][i], Q - 1)
        + mul(period["C"][i], Q - 1) + mul(bp.G, OMEGA)
        == mul(bp.H, period["side_link"][i])
        for i in range(period["row_count"]))


def verify_period(period, records=None, anchor=None, receipt=None,
                  expected_previous_anchor_hash=PREVIOUS_ANCHOR_HASH):
    records = period["records"] if records is None else records
    anchor = period["anchor"] if anchor is None else anchor
    receipt = period["receipt"] if receipt is None else receipt
    try:
        header = verify_anchor(anchor, period["firm_public_key"])
        receipt_time = verify_receipt(
            receipt, anchor, period["custodian_public_key"])
    except Exception:
        return False
    if (header["entity"] != period["entity"]
            or header["period"] != period["period"]
            or header["row_count"] != period["row_count"]
            or header["previous_anchor_hash"] != expected_previous_anchor_hash
            or not header["close_time"] < receipt_time
            < Beacon.opens_at(header["beacon_round"])):
        return False
    if not parameters_valid(period["row_count"], period["total"],
                            period["sample_size"]):
        return False
    if len(records) != period["row_count"]:
        return False
    row_ids = [record[:ROW_ID_BYTES] for record in records]
    if len(set(row_ids)) != len(row_ids):
        return False
    if not all(record_matches_public(period, i, record)
               for i, record in enumerate(records)):
        return False
    if (merkle_root(records) != header["root"]
            or chain_head(records) != header["chain_head"]):
        return False
    if not verify_side_links(period):
        return False

    amount_bp, side_bp = period["amount_bp"], period["side_bp"]
    if amount_bp.verify(period["C"], period["amount_proof"])[0] is not True:
        return False
    if side_bp.verify(period["D"] + period["K"],
                      period["side_proof"])[0] is not True:
        return False
    if (sum_points(period["C"], amount_bp.G)
            != mul(amount_bp.G, period["row_count"] * OMEGA)
            + mul(amount_bp.H, period["sum_rho"])):
        return False
    sum_c = sum_points(period["C"], amount_bp.G)
    sum_d = sum_points(period["D"], amount_bp.G)
    sum_k = sum_points(period["K"], amount_bp.G)
    if (sum_d + mul(sum_c, Q - 1)
            + mul(amount_bp.G, period["row_count"] * OMEGA - period["total"])
            != mul(amount_bp.H, period["sum_eta_minus_rho"])):
        return False
    if (sum_k + sum_c
            + mul(amount_bp.G,
                  -(period["row_count"] * OMEGA + period["total"]))
            != mul(amount_bp.H, period["sum_zeta_plus_rho"])):
        return False
    return True


def prepare_account_balance(period, witness, account):
    indices = [i for i, value in enumerate(witness["accounts"])
               if value == account]
    return dict(account=account, indices=indices,
                balance=sum(witness["amounts"][i] for i in indices),
                amount_blind=sum(witness["rho"][i] for i in indices) % Q,
                account_blinds=[witness["sigma"][i] for i in indices])


def verify_account_balance(period, presentation):
    if not verify_period(period):
        return False
    try:
        indices = presentation["indices"]
        if (not indices or len(indices) != len(set(indices))
                or len(indices) != len(presentation["account_blinds"])):
            return False
        if any(index < 0 or index >= period["row_count"] for index in indices):
            return False
        bp = period["amount_bp"]
        for index, blind in zip(indices, presentation["account_blinds"]):
            if (period["A"][index]
                    != mul(bp.G, presentation["account"])
                    + mul(bp.H, blind)):
                return False
        balance = presentation["balance"]
        if not (-len(indices) * OMEGA <= balance
                < len(indices) * ((1 << 64) - OMEGA)):
            return False
        return (sum_points([period["C"][i] for i in indices], bp.G)
                == mul(bp.G, balance + len(indices) * OMEGA)
                + mul(bp.H, presentation["amount_blind"]))
    except (KeyError, TypeError, ValueError):
        return False


def prepare_sample(period, witness, beacon, side="debit"):
    if side not in ("debit", "credit"):
        raise ValueError("sample side must be debit or credit")
    statement = beacon.publish(BEACON_ROUND)
    units = sample_distinct(period["anchor"] + statement["value"] + b"mus",
                            period["total"], period["sample_size"])
    cumulative_values = []
    value_total = 0
    for value in witness[side]:
        value_total += value
        cumulative_values.append(value_total)

    blindings = witness["eta"] if side == "debit" else witness["zeta"]
    indices, values, blinds, openings, paths = [], [], [], [], []
    for unit in units:
        index = bisect.bisect_right(cumulative_values, unit)
        previous_value = cumulative_values[index - 1] if index else 0
        indices.append(index)
        values.extend([unit - previous_value,
                       cumulative_values[index] - unit - 1])
        previous_blind = sum(blindings[:index]) % Q
        current_blind = sum(blindings[:index + 1]) % Q
        blinds.extend([(-previous_blind) % Q, current_blind])
        openings.append(dict(
            amount=witness["amounts"][index],
            account=witness["accounts"][index],
            rho=witness["rho"][index], sigma=witness["sigma"][index],
            debit=witness["debit"][index], eta=witness["eta"][index],
            credit=witness["credit"][index], zeta=witness["zeta"][index]))
        paths.append(merkle_path(period["records"], index))

    commitments = [period["sample_bp"].commit(values[i], blinds[i])
                   for i in range(len(values))]
    return dict(side=side, beacon=statement, indices=indices,
                commitments=commitments,
                proof=period["sample_bp"].prove(values, blinds),
                openings=openings, paths=paths)


def verify_sample(period, presentation, beacon, audit_time=AUDIT_TIME):
    if not verify_period(period):
        return False
    try:
        header = verify_anchor(period["anchor"], period["firm_public_key"])
        if not beacon.verify(presentation["beacon"],
                             header["beacon_round"], audit_time):
            return False
        side = presentation["side"]
        if side not in ("debit", "credit"):
            return False
        units = sample_distinct(
            period["anchor"] + presentation["beacon"]["value"] + b"mus",
            period["total"], period["sample_size"])
        if (len(presentation["indices"]) != period["sample_size"]
                or len(presentation["openings"]) != period["sample_size"]
                or len(presentation["paths"]) != period["sample_size"]):
            return False
        if period["sample_bp"].verify(
                presentation["commitments"], presentation["proof"])[0] is not True:
            return False

        cumulative_points = []
        point_total = mul(period["amount_bp"].G, 0)
        side_points = period["D"] if side == "debit" else period["K"]
        for point in side_points:
            point_total = point_total + point
            cumulative_points.append(point_total)
        zero = mul(period["amount_bp"].G, 0)
        bp = period["amount_bp"]
        for n, (unit, index, opening, path) in enumerate(zip(
                units, presentation["indices"], presentation["openings"],
                presentation["paths"])):
            if not 0 <= index < period["row_count"]:
                return False
            previous_point = cumulative_points[index - 1] if index else zero
            current_point = cumulative_points[index]
            D1 = mul(bp.G, unit) + mul(previous_point, Q - 1)
            D2 = current_point + mul(bp.G, (-(unit + 1)) % Q)
            if (D1 != presentation["commitments"][2 * n]
                    or D2 != presentation["commitments"][2 * n + 1]):
                return False
            if not verify_path(period["records"][index], path,
                               header["root"]):
                return False
            if (bp.commit(opening["amount"] + OMEGA, opening["rho"])
                    != period["C"][index]
                    or bp.commit(opening["account"], opening["sigma"])
                    != period["A"][index]
                    or bp.commit(opening["debit"], opening["eta"])
                    != period["D"][index]
                    or bp.commit(opening["credit"], opening["zeta"])
                    != period["K"][index]
                    or opening["debit"] != max(opening["amount"], 0)
                    or opening["credit"] != max(-opening["amount"], 0)):
                return False
        return True
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def anchored_understatement(period, witness, signing):
    """Build the old equal-reduction attack into a freshly signed record set."""
    changed = copy.copy(period)
    changed["D"] = list(period["D"])
    changed["K"] = list(period["K"])
    changed["records"] = list(period["records"])
    changed_debit = list(witness["debit"])
    changed_credit = list(witness["credit"])
    changed_debit[0] -= 1
    changed_credit[1] -= 1
    bp = period["amount_bp"]
    changed["D"][0] = bp.commit(changed_debit[0], witness["eta"][0])
    changed["K"][1] = bp.commit(changed_credit[1], witness["zeta"][1])
    for index in (0, 1):
        changed["records"][index] = encode_record(
            period["row_ids"][index], period["encrypted"][index],
            enc(period["C"][index]), enc(period["A"][index]),
            enc(changed["D"][index]), enc(changed["K"][index]),
            period["side_link"][index].to_bytes(32, "big"))
    root, head = merkle_root(changed["records"]), chain_head(changed["records"])
    env = encode_anchor_envelope(
        ENTITY, PERIOD, N, root, head, PREVIOUS_ANCHOR_HASH,
        CLOSE_TIME, BEACON_ROUND)
    changed["anchor"] = make_anchor(env, signing["firm"])
    changed["receipt"] = make_receipt(
        changed["anchor"], RECEIPT_TIME, signing["custodian"])
    return changed


def main():
    period, witness, signing = make_fixture()
    beacon = Beacon()
    sample = prepare_sample(period, witness, beacon)
    credit_sample = prepare_sample(period, witness, beacon, side="credit")
    account = prepare_account_balance(period, witness, 1000)

    wrong_sample = dict(sample)
    wrong_sample["indices"] = list(sample["indices"])
    wrong_sample["indices"][0] = (wrong_sample["indices"][0] + 1) % N
    forged_beacon = dict(sample)
    forged_beacon["beacon"] = dict(sample["beacon"])
    forged_beacon["beacon"]["value"] = b"\xff" * 32
    foreign_account = dict(account)
    foreign_account["indices"] = list(account["indices"])
    foreign_account["indices"][0] = 1
    tampered_records = list(period["records"])
    tampered_records[0] += b"x"
    late_receipt = make_receipt(
        period["anchor"], Beacon.opens_at(BEACON_ROUND), signing["custodian"])
    understated = anchored_understatement(period, witness, signing)

    checks = {
        "honest period accepts after beacon opens": verify_period(period),
        "honest debit MUS presentation accepts": verify_sample(
            period, sample, beacon),
        "honest credit MUS presentation accepts": verify_sample(
            period, credit_sample, beacon),
        "honest account-balance presentation accepts": verify_account_balance(
            period, account),
        "audit before beacon opening is rejected": not verify_sample(
            period, sample, beacon, audit_time=250),
        "forged beacon statement is rejected": not verify_sample(
            period, forged_beacon, beacon),
        "tampered record and length are rejected": not verify_period(
            period, records=tampered_records),
        "receipt at beacon opening is rejected": not verify_period(
            period, receipt=late_receipt),
        "wrong sampled row is rejected": not verify_sample(
            period, wrong_sample, beacon),
        "foreign account row is rejected": not verify_account_balance(
            period, foreign_account),
        "changed previous anchor is rejected": not verify_period(
            period, expected_previous_anchor_hash=b"\x00" * 32),
        "anchored equal understatement is rejected": not verify_period(
            understated),
        "understatement violates per-row side link": not verify_side_links(
            understated),
        "canonical records are exactly 228 bytes": all(
            len(record) == P256_RECORD_BYTES for record in period["records"]),
    }
    for label, result in checks.items():
        print("  {:52} : {}".format(label, result))
    print("  all checks pass                                     : {}".format(
        all(checks.values())))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
