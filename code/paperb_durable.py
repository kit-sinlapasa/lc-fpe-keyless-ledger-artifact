#!/usr/bin/env python3
"""Durable end-to-end research prototype for Paper B.

The small fixture in paperb_e2e.py checks protocol consistency in memory. This
prototype adds the system properties that fixture deliberately omits: SQLite
transactions with WAL/FULL durability, encrypted row payloads, an atomic
period close, a one-receipt-per-period custodian, an authenticated future
beacon, persisted full-ledger range proofs, restart recovery, and fault tests.

It is a research prototype, not a production ERP or an external audit.
"""
import hashlib
import hmac
import json
import sqlite3
import struct
import tempfile
from pathlib import Path

from Crypto.PublicKey import ECC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from bulletproofs import BP, P, Q, B, enc, mul
from paperb_e2e import (chain_head, merkle_path, merkle_root, sample_distinct,
                        verify_path)


H = hashlib.sha256
OMEGA = 1 << 40
PERIOD = "entity-1/2026-09"
ROUND = 200
CLOSE_TIME = 100
BEACON_OPEN_TIME = 300
SAMPLE_SIZE = 4


def scalar(key, tag, row_id):
    return int.from_bytes(hmac.new(key, tag + row_id, hashlib.sha256).digest(),
                          "big") % Q


def dec(data):
    if len(data) != 33 or data[0] not in (2, 3):
        raise ValueError("non-canonical compressed P-256 point")
    x = int.from_bytes(data[1:], "big")
    if x >= P:
        raise ValueError("x coordinate outside field")
    y = pow((pow(x, 3, P) - 3 * x + B) % P, (P + 1) // 4, P)
    if (y & 1) != (data[0] & 1):
        y = P - y
    point = ECC.EccPoint(x, y, curve="p256")
    if enc(point) != data:
        raise ValueError("point encoding did not round trip")
    return point


def proof_bytes(proof):
    points = [proof[name] for name in ("A", "S", "T1", "T2")]
    scalars = [proof[name] for name in ("taux", "mu", "that", "a", "b")]
    rounds = len(proof["L"])
    if rounds != len(proof["R"]) or rounds > 0xFFFF:
        raise ValueError("invalid inner-product proof shape")
    return (b"BP1" + struct.pack(">H", rounds)
            + b"".join(enc(point) for point in points)
            + b"".join(int(value).to_bytes(32, "big") for value in scalars)
            + b"".join(enc(point) for point in proof["L"])
            + b"".join(enc(point) for point in proof["R"]))


def parse_proof(blob, expected_rounds):
    expected = 5 + 4 * 33 + 5 * 32 + 2 * expected_rounds * 33
    if len(blob) != expected or blob[:3] != b"BP1":
        raise ValueError("proof length/header rejected before allocation")
    rounds = struct.unpack(">H", blob[3:5])[0]
    if rounds != expected_rounds:
        raise ValueError("unexpected proof rounds")
    pos = 5
    pts = []
    for _ in range(4):
        pts.append(dec(blob[pos:pos + 33])); pos += 33
    scalars = []
    for _ in range(5):
        value = int.from_bytes(blob[pos:pos + 32], "big"); pos += 32
        if value >= Q:
            raise ValueError("non-canonical scalar")
        scalars.append(value)
    left, right = [], []
    for target in (left, right):
        for _ in range(rounds):
            target.append(dec(blob[pos:pos + 33])); pos += 33
    return dict(zip(("A", "S", "T1", "T2"), pts),
                **dict(zip(("taux", "mu", "that", "a", "b"), scalars)),
                L=left, R=right)


def row_record(row):
    return (row["row_id"] + row["nonce"] + row["ciphertext"] + row["C"]
            + row["A"] + row["D"] + row["K"] + row["side_link"])


def anchor_envelope(period, n_rows, total, root, head, close_time, round_number,
                    sum_rho, sum_eta_minus_rho, sum_zeta_plus_rho):
    body = dict(version=1, period=period, n_rows=n_rows, total=total,
                root=root.hex(), head=head.hex(), close_time=close_time,
                beacon_round=round_number, sum_rho=str(sum_rho),
                sum_eta_minus_rho=str(sum_eta_minus_rho),
                sum_zeta_plus_rho=str(sum_zeta_plus_rho))
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


class LedgerService:
    def __init__(self, path, data_key, blind_key, firm_key):
        self.path = str(path)
        self.data_key = data_key
        self.blind_key = blind_key
        self.firm_key = firm_key
        self.db = sqlite3.connect(self.path, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS periods(
          period TEXT PRIMARY KEY, state TEXT NOT NULL CHECK(state IN ('OPEN','CLOSED')),
          anchor BLOB, amount_proof BLOB, side_proof BLOB);
        CREATE TABLE IF NOT EXISTS rows(
          period TEXT NOT NULL REFERENCES periods(period), seq INTEGER NOT NULL,
          row_id BLOB NOT NULL UNIQUE, nonce BLOB NOT NULL, ciphertext BLOB NOT NULL,
          C BLOB NOT NULL, A BLOB NOT NULL, D BLOB NOT NULL, K BLOB NOT NULL,
          side_link BLOB NOT NULL, PRIMARY KEY(period,seq));
        CREATE TABLE IF NOT EXISTS receipts(
          period TEXT PRIMARY KEY REFERENCES periods(period), receipt_time INTEGER NOT NULL,
          receipt BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS beacons(
          round INTEGER PRIMARY KEY, open_time INTEGER NOT NULL, value BLOB NOT NULL,
          signature BLOB NOT NULL);
        """)
        self.bp = BP(32, 2 * SAMPLE_SIZE)

    def close(self):
        self.db.close()

    def start_period(self, period=PERIOD):
        self.db.execute("INSERT INTO periods(period,state) VALUES(?, 'OPEN')", (period,))

    def ingest(self, seq, amount, account, period=PERIOD):
        if not -(1 << 32) < amount < (1 << 32):
            raise ValueError("amount outside declared side range")
        row_id = H((period + "/journal-1/doc-{}/line-1".format(seq)).encode()).digest()
        rho = scalar(self.blind_key, b"amount/", row_id)
        sigma = scalar(self.blind_key, b"account/", row_id)
        eta = scalar(self.blind_key, b"debit/", row_id)
        zeta = scalar(self.blind_key, b"credit/", row_id)
        debit, credit = max(amount, 0), max(-amount, 0)
        C = self.bp.commit(amount + OMEGA, rho)
        A = self.bp.commit(account, sigma)
        D = self.bp.commit(debit, eta)
        K = self.bp.commit(credit, zeta)
        link = (eta - zeta - rho) % Q
        if D + mul(K, Q - 1) + mul(C, Q - 1) + mul(self.bp.G, OMEGA) != mul(self.bp.H, link):
            raise AssertionError("side-link construction failed")
        nonce = hmac.new(self.data_key, b"nonce/" + row_id, hashlib.sha256).digest()[:12]
        ciphertext = AESGCM(self.data_key).encrypt(
            nonce, struct.pack(">qI", amount, account), row_id)
        state = self.db.execute("SELECT state FROM periods WHERE period=?", (period,)).fetchone()
        if not state or state[0] != "OPEN":
            raise ValueError("period is not open")
        self.db.execute(
            "INSERT INTO rows VALUES(?,?,?,?,?,?,?,?,?,?)",
            (period, seq, row_id, nonce, ciphertext, enc(C), enc(A), enc(D), enc(K),
             link.to_bytes(32, "big")))

    def _rows(self, period=PERIOD):
        return [dict(row) for row in self.db.execute(
            "SELECT * FROM rows WHERE period=? ORDER BY seq", (period,))]

    def opening(self, row):
        amount, account = struct.unpack(">qI", AESGCM(self.data_key).decrypt(
            row["nonce"], row["ciphertext"], row["row_id"]))
        rid = row["row_id"]
        return dict(amount=amount, account=account,
                    rho=scalar(self.blind_key, b"amount/", rid),
                    sigma=scalar(self.blind_key, b"account/", rid),
                    eta=scalar(self.blind_key, b"debit/", rid),
                    zeta=scalar(self.blind_key, b"credit/", rid))

    def close_period(self, fail_before_commit=False, period=PERIOD):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            state = self.db.execute("SELECT state FROM periods WHERE period=?", (period,)).fetchone()
            if not state or state[0] != "OPEN":
                raise ValueError("period already closed")
            rows = self._rows(period)
            openings = [self.opening(row) for row in rows]
            amounts = [item["amount"] for item in openings]
            if not rows or sum(amounts) != 0:
                raise ValueError("period does not balance")
            debits = [max(value, 0) for value in amounts]
            credits = [max(-value, 0) for value in amounts]
            rho = [item["rho"] for item in openings]
            eta = [item["eta"] for item in openings]
            zeta = [item["zeta"] for item in openings]
            n = len(rows)
            if n & (n - 1):
                raise ValueError("prototype uses a power-of-two row count")
            amount_bp = BP(64, n)
            side_bp = BP(32, 2 * n)
            amount_proof = proof_bytes(amount_bp.prove(
                [value + OMEGA for value in amounts], rho))
            side_proof = proof_bytes(side_bp.prove(debits + credits, eta + zeta))
            records = [row_record(row) for row in rows]
            total = sum(debits)
            env = anchor_envelope(period, n, total, merkle_root(records),
                                  chain_head(records), CLOSE_TIME, ROUND,
                                  sum(rho) % Q, (sum(eta) - sum(rho)) % Q,
                                  (sum(zeta) + sum(rho)) % Q)
            anchor = struct.pack(">I", len(env)) + env + self.firm_key.sign(env)
            if fail_before_commit:
                raise RuntimeError("injected crash before COMMIT")
            self.db.execute(
                "UPDATE periods SET state='CLOSED',anchor=?,amount_proof=?,side_proof=? "
                "WHERE period=?", (anchor, amount_proof, side_proof, period))
            self.db.execute("COMMIT")
            return anchor
        except Exception:
            self.db.execute("ROLLBACK")
            raise


def decode_anchor(anchor, firm_public):
    if len(anchor) < 4 + 64:
        raise ValueError("short anchor")
    size = struct.unpack(">I", anchor[:4])[0]
    if len(anchor) != 4 + size + 64:
        raise ValueError("anchor length mismatch")
    env, signature = anchor[4:4 + size], anchor[4 + size:]
    firm_public.verify(signature, env)
    return json.loads(env)


def countersign(db, period, anchor, receipt_time, custodian_key):
    message = H(anchor).digest() + period.encode() + struct.pack(">Q", receipt_time)
    receipt = message + custodian_key.sign(message)
    db.execute("INSERT INTO receipts VALUES(?,?,?)", (period, receipt_time, receipt))
    return receipt


def publish_beacon(db, round_number, open_time, value, beacon_key):
    message = struct.pack(">IQ", round_number, open_time) + value
    signature = beacon_key.sign(message)
    db.execute("INSERT INTO beacons VALUES(?,?,?,?)",
               (round_number, open_time, value, signature))
    return signature


def add_points(points):
    total = points[0]
    for point in points[1:]:
        total = total + point
    return total


def verify_period(db, firm_public, custodian_public, current_round=100, period=PERIOD):
    p = db.execute("SELECT * FROM periods WHERE period=?", (period,)).fetchone()
    receipt_row = db.execute("SELECT * FROM receipts WHERE period=?", (period,)).fetchone()
    if not p or p["state"] != "CLOSED" or not receipt_row:
        return False
    try:
        meta = decode_anchor(p["anchor"], firm_public)
        receipt = receipt_row["receipt"]
        custodian_public.verify(receipt[-64:], receipt[:-64])
        if receipt[:-64] != (H(p["anchor"]).digest() + period.encode()
                             + struct.pack(">Q", receipt_row["receipt_time"])):
            return False
        rows = [dict(row) for row in db.execute(
            "SELECT * FROM rows WHERE period=? ORDER BY seq", (period,))]
        records = [row_record(row) for row in rows]
        if (meta["period"] != period
                or len(rows) != meta["n_rows"]
                or len({row["row_id"] for row in rows}) != len(rows)
                or merkle_root(records).hex() != meta["root"]
                or chain_head(records).hex() != meta["head"]
                or meta["beacon_round"] <= current_round
                or meta["close_time"] >= receipt_row["receipt_time"]
                or receipt_row["receipt_time"] >= BEACON_OPEN_TIME):
            return False
        bp = BP(32, 2 * SAMPLE_SIZE)
        C, D, K = ([dec(row[name]) for row in rows] for name in ("C", "D", "K"))
        links = [int.from_bytes(row["side_link"], "big") for row in rows]
        if not all(D[i] + mul(K[i], Q - 1) + mul(C[i], Q - 1) + mul(bp.G, OMEGA)
                   == mul(bp.H, links[i]) for i in range(len(rows))):
            return False
        n = len(rows)
        amount_bp, side_bp = BP(64, n), BP(32, 2 * n)
        amount_pf = parse_proof(p["amount_proof"], (64 * n).bit_length() - 1)
        side_pf = parse_proof(p["side_proof"], (64 * n).bit_length() - 1)
        if not amount_bp.verify(C, amount_pf)[0] or not side_bp.verify(D + K, side_pf)[0]:
            return False
        sumC, sumD, sumK = add_points(C), add_points(D), add_points(K)
        total, count = int(meta["total"]), len(rows)
        if sumC != mul(bp.G, count * OMEGA) + mul(bp.H, int(meta["sum_rho"])):
            return False
        lhs_d = sumD + mul(sumC, Q - 1) + mul(bp.G, count * OMEGA - total)
        lhs_k = sumK + sumC + mul(bp.G, -(count * OMEGA + total))
        return (lhs_d == mul(bp.H, int(meta["sum_eta_minus_rho"]))
                and lhs_k == mul(bp.H, int(meta["sum_zeta_plus_rho"])))
    except Exception:
        return False


def prepare_and_verify_sample(service, firm_public, custodian_public,
                              beacon_public, period=PERIOD):
    db = service.db
    if not verify_period(db, firm_public, custodian_public, period=period):
        return False
    period_row = db.execute(
        "SELECT anchor FROM periods WHERE period=?", (period,)).fetchone()
    receipt = db.execute(
        "SELECT receipt_time FROM receipts WHERE period=?", (period,)).fetchone()
    meta = decode_anchor(period_row["anchor"], firm_public)
    beacon_round = int(meta["beacon_round"])
    beacon = db.execute("SELECT * FROM beacons WHERE round=?", (beacon_round,)).fetchone()
    if not beacon:
        return False
    message = struct.pack(">IQ", beacon_round, beacon["open_time"]) + beacon["value"]
    try:
        beacon_public.verify(beacon["signature"], message)
    except Exception:
        return False
    if (beacon["open_time"] != BEACON_OPEN_TIME
            or beacon["open_time"] <= receipt["receipt_time"]):
        return False
    rows = service._rows(period)
    openings = [service.opening(row) for row in rows]
    debit = [max(item["amount"], 0) for item in openings]
    eta = [item["eta"] for item in openings]
    cumulative, running = [], 0
    for value in debit:
        running += value; cumulative.append(running)
    anchor = period_row["anchor"]
    units = sample_distinct(anchor + beacon["value"] + b"mus", running, SAMPLE_SIZE)
    indices, values, blinds = [], [], []
    for unit in units:
        j = next(i for i, total in enumerate(cumulative) if unit < total)
        previous = cumulative[j - 1] if j else 0
        indices.append(j)
        values += [unit - previous, cumulative[j] - unit - 1]
        before_blind = sum(eta[:j]) % Q
        through_blind = sum(eta[:j + 1]) % Q
        blinds += [(-before_blind) % Q, through_blind]
    bp = service.bp
    interval_commitments = [bp.commit(v, r) for v, r in zip(values, blinds)]
    interval_proof = bp.prove(values, blinds)
    if not bp.verify(interval_commitments, interval_proof)[0]:
        return False
    records = [row_record(row) for row in rows]
    D = [dec(row["D"]) for row in rows]
    prefix, point = [], mul(bp.G, 0)
    for commitment in D:
        point = point + commitment; prefix.append(point)
    for pos, (unit, j) in enumerate(zip(units, indices)):
        previous_point = prefix[j - 1] if j else mul(bp.G, 0)
        if (mul(bp.G, unit) + mul(previous_point, Q - 1) != interval_commitments[2 * pos]
                or prefix[j] + mul(bp.G, -(unit + 1)) != interval_commitments[2 * pos + 1]
                or not verify_path(records[j], j, merkle_path(records, j), merkle_root(records))):
            return False
        item, row = openings[j], rows[j]
        amount, account = item["amount"], item["account"]
        if (bp.commit(amount + OMEGA, item["rho"]) != dec(row["C"])
                or bp.commit(account, item["sigma"]) != dec(row["A"])
                or bp.commit(max(amount, 0), item["eta"]) != dec(row["D"])
                or bp.commit(max(-amount, 0), item["zeta"]) != dec(row["K"])):
            return False
    return True


def main():
    firm = Ed25519PrivateKey.from_private_bytes(H(b"paperb/durable/firm").digest())
    custodian = Ed25519PrivateKey.from_private_bytes(H(b"paperb/durable/custodian").digest())
    beacon_key = Ed25519PrivateKey.from_private_bytes(H(b"paperb/durable/beacon").digest())
    data_key, blind_key = H(b"paperb/durable/data").digest(), H(b"paperb/durable/blind").digest()
    amounts = [3, -2, 5, -6, 4, -4, 7, -7, 2, -3, 6, -5, 1, -2, 8, -7]

    with tempfile.TemporaryDirectory(prefix="paperb-durable-") as directory:
        path = Path(directory) / "ledger.db"
        service = LedgerService(path, data_key, blind_key, firm)
        service.start_period()
        for i, amount in enumerate(amounts):
            service.ingest(i, amount, 1000 + i)
        duplicate_rejected = False
        try:
            service.ingest(99, 0, 9999)
            service.ingest(99, 0, 9999)
        except sqlite3.IntegrityError:
            duplicate_rejected = True
        # Remove the first insertion if the duplicate check reached it.
        service.db.execute("DELETE FROM rows WHERE seq=99")

        crash_rolled_back = False
        try:
            service.close_period(fail_before_commit=True)
        except RuntimeError:
            state = service.db.execute("SELECT state,anchor FROM periods").fetchone()
            crash_rolled_back = state["state"] == "OPEN" and state["anchor"] is None

        anchor = service.close_period()
        countersign(service.db, PERIOD, anchor, 150, custodian)
        second_close_rejected = False
        try:
            service.close_period()
        except ValueError:
            second_close_rejected = True
        conflicting_receipt_rejected = False
        try:
            countersign(service.db, PERIOD, anchor, 151, custodian)
        except sqlite3.IntegrityError:
            conflicting_receipt_rejected = True
        publish_beacon(service.db, ROUND, BEACON_OPEN_TIME,
                       H(b"paperb/durable/beacon/value").digest(), beacon_key)
        service.close()

        # Restart from the durable database, then run the verifier and sample.
        restarted = LedgerService(path, data_key, blind_key, firm)
        restart_verifies = verify_period(
            restarted.db, firm.public_key(), custodian.public_key())
        sample_verifies = prepare_and_verify_sample(
            restarted, firm.public_key(), custodian.public_key(), beacon_key.public_key())

        original = restarted.db.execute(
            "SELECT ciphertext FROM rows WHERE period=? AND seq=0", (PERIOD,)).fetchone()[0]
        tampered = bytes([original[0] ^ 1]) + original[1:]
        restarted.db.execute("UPDATE rows SET ciphertext=? WHERE period=? AND seq=0",
                             (tampered, PERIOD))
        tamper_rejected = not verify_period(
            restarted.db, firm.public_key(), custodian.public_key())
        restarted.db.execute("UPDATE rows SET ciphertext=? WHERE period=? AND seq=0",
                             (original, PERIOD))

        beacon = restarted.db.execute("SELECT signature FROM beacons WHERE round=?",
                                      (ROUND,)).fetchone()[0]
        bad = bytes([beacon[0] ^ 1]) + beacon[1:]
        restarted.db.execute("UPDATE beacons SET signature=? WHERE round=?", (bad, ROUND))
        forged_beacon_rejected = not prepare_and_verify_sample(
            restarted, firm.public_key(), custodian.public_key(), beacon_key.public_key())
        restarted.close()

    checks = {
        "duplicate RowID rejected": duplicate_rejected,
        "crash before COMMIT rolls close back": crash_rolled_back,
        "second period close rejected": second_close_rejected,
        "conflicting custodian receipt rejected": conflicting_receipt_rejected,
        "restart verifies persisted anchor and full proofs": restart_verifies,
        "authenticated-beacon sample verifies end to end": sample_verifies,
        "post-close row tamper rejected": tamper_rejected,
        "forged beacon rejected": forged_beacon_rejected,
    }
    print("=" * 78)
    print("PAPER B DURABLE END-TO-END PROTOTYPE")
    print("=" * 78)
    print("  SQLite journal_mode=WAL, synchronous=FULL")
    for label, result in checks.items():
        print("  {:55} : {}".format(label, result))
    print("  all checks pass                                      : {}".format(all(checks.values())))
    print("  scope: durable research prototype, not production ERP")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
