#!/usr/bin/env python3
"""Durable, role-separated research prototype for Paper B.

The writer alone receives the master key. The restarted verifier opens the
SQLite database directly and checks persisted proofs, signatures, anchors,
sampling presentations, and account presentations using public inputs only.
This remains a research prototype, not a production ERP or an external audit.
"""
import hashlib
import hmac
import sqlite3
import struct
import tempfile
from pathlib import Path

from Crypto.PublicKey import ECC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from bulletproofs import BP, P, Q, B, enc, mul
from paperb_e2e import (chain_head, merkle_path, merkle_root, sample_distinct,
                        verify_path)
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
PERIOD_KEY = "entity-1/2026-09"
ENTITY = 1
PERIOD_CODE = 202609
PARTITION = 0
LEDGER = 0
ROUND = 200
CLOSE_TIME = 100
RECEIPT_TIME = 150
BEACON_OPEN_TIME = 300
AUDIT_TIME = 350
SAMPLE_SIZE = 4
PREVIOUS_ANCHOR_HASH = H(b"paperb-durable/genesis").digest()
BEACON_DOMAIN = b"PAPERB-DURABLE-BEACON/1"


def derive_key(master_key, label):
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=b"paperb-durable/" + label).derive(master_key)


def scalar(key, tag, row_id):
    return int.from_bytes(hmac.new(key, tag + row_id, H).digest(), "big") % Q


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
    points = []
    for _ in range(4):
        points.append(dec(blob[pos:pos + 33]))
        pos += 33
    scalars = []
    for _ in range(5):
        value = int.from_bytes(blob[pos:pos + 32], "big")
        pos += 32
        if value >= Q:
            raise ValueError("non-canonical scalar")
        scalars.append(value)
    left, right = [], []
    for target in (left, right):
        for _ in range(rounds):
            target.append(dec(blob[pos:pos + 33]))
            pos += 33
    return dict(zip(("A", "S", "T1", "T2"), points),
                **dict(zip(("taux", "mu", "that", "a", "b"), scalars)),
                L=left, R=right)


def add_points(points, generator):
    total = mul(generator, 0)
    for point in points:
        total = total + point
    return total


def row_record(row):
    return encode_record(row["row_id"], row["encrypted_fields"], row["C"],
                         row["A"], row["D"], row["K"], row["side_link"])


def beacon_message(round_number, open_time, value):
    return BEACON_DOMAIN + struct.pack(">IQ", round_number, open_time) + value


def beacon_open_time(round_number):
    return round_number + 100


def parameters_valid(row_count, total, sample_size=SAMPLE_SIZE):
    return (row_count > 0
            and row_count * (1 << 64) < Q
            and (row_count + 1) * (1 << 32) + OMEGA < Q
            and (1 << 64) + (1 << 33) + OMEGA < Q
            and 0 < total < row_count * (1 << 32)
            and total < (1 << 256)
            and 1 <= sample_size <= total)


class LedgerService:
    """Writer/prover role. The verifier never receives this object's keys."""

    def __init__(self, path, master_key, firm_key):
        self.path = str(path)
        self.data_key = derive_key(master_key, b"data")
        self.nonce_key = derive_key(master_key, b"nonce")
        self.blind_key = derive_key(master_key, b"amount-blind")
        self.class_key = derive_key(master_key, b"class-blind")
        self.firm_key = firm_key
        self.db = sqlite3.connect(self.path, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS periods(
          period_key TEXT PRIMARY KEY,
          entity INTEGER NOT NULL,
          period_code INTEGER NOT NULL,
          state TEXT NOT NULL CHECK(state IN ('OPEN','CLOSED')),
          total TEXT, sum_rho TEXT, sum_eta_minus_rho TEXT,
          sum_zeta_plus_rho TEXT,
          anchor BLOB, amount_proof BLOB, side_proof BLOB);
        CREATE TABLE IF NOT EXISTS rows(
          period_key TEXT NOT NULL REFERENCES periods(period_key),
          seq INTEGER NOT NULL,
          row_id BLOB NOT NULL UNIQUE,
          encrypted_fields BLOB NOT NULL,
          C BLOB NOT NULL, A BLOB NOT NULL, D BLOB NOT NULL, K BLOB NOT NULL,
          side_link BLOB NOT NULL,
          PRIMARY KEY(period_key,seq));
        CREATE TABLE IF NOT EXISTS receipts(
          period_key TEXT PRIMARY KEY REFERENCES periods(period_key),
          receipt BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS beacons(
          round INTEGER PRIMARY KEY, open_time INTEGER NOT NULL,
          value BLOB NOT NULL, signature BLOB NOT NULL);
        """)
        self.sample_bp = BP(32, 2 * SAMPLE_SIZE)

    def close(self):
        self.db.close()

    def start_period(self, period_key=PERIOD_KEY):
        self.db.execute(
            "INSERT INTO periods(period_key,entity,period_code,state) "
            "VALUES(?,?,?,'OPEN')", (period_key, ENTITY, PERIOD_CODE))

    def ingest(self, seq, amount, account, period_key=PERIOD_KEY):
        if not -(1 << 32) < amount < (1 << 32):
            raise ValueError("amount outside declared side range")
        if not 0 <= account < 1 << 32:
            raise ValueError("account outside fixed-width encoding")
        row_id = encode_row_id(ENTITY, PERIOD_CODE, PARTITION, LEDGER,
                               10000 + seq, 1)
        rho = scalar(self.blind_key, b"amount/", row_id)
        sigma = scalar(self.class_key, b"account/", row_id)
        eta = scalar(self.blind_key, b"debit/", row_id)
        zeta = scalar(self.blind_key, b"credit/", row_id)
        debit, credit = max(amount, 0), max(-amount, 0)
        C = self.sample_bp.commit(amount + OMEGA, rho)
        A = self.sample_bp.commit(account, sigma)
        D = self.sample_bp.commit(debit, eta)
        K = self.sample_bp.commit(credit, zeta)
        link = (eta - zeta - rho) % Q
        if (D + mul(K, Q - 1) + mul(C, Q - 1)
                + mul(self.sample_bp.G, OMEGA) != mul(self.sample_bp.H, link)):
            raise AssertionError("side-link construction failed")
        nonce = hmac.new(self.nonce_key, b"row/" + row_id, H).digest()[:12]
        payload = struct.pack(">qI", amount, account)
        encrypted = nonce + AESGCM(self.data_key).encrypt(
            nonce, payload, row_id)
        if len(encrypted) != 40:
            raise AssertionError("encrypted-field width changed")
        state = self.db.execute(
            "SELECT state FROM periods WHERE period_key=?",
            (period_key,)).fetchone()
        if not state or state[0] != "OPEN":
            raise ValueError("period is not open")
        self.db.execute(
            "INSERT INTO rows VALUES(?,?,?,?,?,?,?,?,?)",
            (period_key, seq, row_id, encrypted, enc(C), enc(A), enc(D),
             enc(K), link.to_bytes(32, "big")))

    def rows(self, period_key=PERIOD_KEY):
        return [dict(row) for row in self.db.execute(
            "SELECT * FROM rows WHERE period_key=? ORDER BY seq",
            (period_key,))]

    def opening(self, row):
        nonce, ciphertext = row["encrypted_fields"][:12], row["encrypted_fields"][12:]
        amount, account = struct.unpack(">qI", AESGCM(self.data_key).decrypt(
            nonce, ciphertext, row["row_id"]))
        row_id = row["row_id"]
        return dict(
            amount=amount, account=account,
            rho=scalar(self.blind_key, b"amount/", row_id),
            sigma=scalar(self.class_key, b"account/", row_id),
            eta=scalar(self.blind_key, b"debit/", row_id),
            zeta=scalar(self.blind_key, b"credit/", row_id))

    def close_period(self, fail_before_commit=False, period_key=PERIOD_KEY):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            state = self.db.execute(
                "SELECT state FROM periods WHERE period_key=?",
                (period_key,)).fetchone()
            if not state or state[0] != "OPEN":
                raise ValueError("period already closed")
            rows = self.rows(period_key)
            openings = [self.opening(row) for row in rows]
            amounts = [item["amount"] for item in openings]
            if not rows or sum(amounts) != 0:
                raise ValueError("period does not balance")
            n = len(rows)
            if n & (n - 1):
                raise ValueError("prototype uses a power-of-two row count")
            debit = [max(value, 0) for value in amounts]
            credit = [max(-value, 0) for value in amounts]
            rho = [item["rho"] for item in openings]
            eta = [item["eta"] for item in openings]
            zeta = [item["zeta"] for item in openings]
            total = sum(debit)
            if not parameters_valid(n, total):
                raise ValueError("period parameters violate no-wrap bounds")
            amount_bp, side_bp = BP(64, n), BP(32, 2 * n)
            amount_proof = proof_bytes(amount_bp.prove(
                [value + OMEGA for value in amounts], rho))
            side_proof = proof_bytes(side_bp.prove(debit + credit, eta + zeta))
            records = [row_record(row) for row in rows]
            if any(len(record) != P256_RECORD_BYTES for record in records):
                raise AssertionError("canonical record width changed")
            env = encode_anchor_envelope(
                ENTITY, PERIOD_CODE, n, merkle_root(records),
                chain_head(records), PREVIOUS_ANCHOR_HASH, CLOSE_TIME, ROUND)
            anchor = make_anchor(env, self.firm_key)
            if fail_before_commit:
                raise RuntimeError("injected crash before COMMIT")
            self.db.execute(
                "UPDATE periods SET state='CLOSED',total=?,sum_rho=?,"
                "sum_eta_minus_rho=?,sum_zeta_plus_rho=?,anchor=?,"
                "amount_proof=?,side_proof=? "
                "WHERE period_key=?",
                (str(total), str(sum(rho) % Q),
                 str((sum(eta) - sum(rho)) % Q),
                 str((sum(zeta) + sum(rho)) % Q), anchor,
                 amount_proof, side_proof,
                 period_key))
            self.db.execute("COMMIT")
            return anchor
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def prepare_sample(self, side="debit", period_key=PERIOD_KEY):
        if side not in ("debit", "credit"):
            raise ValueError("sample side must be debit or credit")
        period = self.db.execute(
            "SELECT * FROM periods WHERE period_key=?", (period_key,)).fetchone()
        beacon = self.db.execute(
            "SELECT * FROM beacons WHERE round=?", (ROUND,)).fetchone()
        rows = self.rows(period_key)
        openings = [self.opening(row) for row in rows]
        values_by_row = [max(item["amount"], 0) if side == "debit"
                         else max(-item["amount"], 0) for item in openings]
        side_blinds = [item["eta"] if side == "debit" else item["zeta"]
                       for item in openings]
        cumulative, running = [], 0
        for value in values_by_row:
            running += value
            cumulative.append(running)
        units = sample_distinct(period["anchor"] + beacon["value"] + b"mus",
                                running, SAMPLE_SIZE)
        indices, values, blinds, disclosed, paths = [], [], [], [], []
        records = [row_record(row) for row in rows]
        for unit in units:
            index = next(i for i, total in enumerate(cumulative) if unit < total)
            previous = cumulative[index - 1] if index else 0
            indices.append(index)
            values.extend([unit - previous, cumulative[index] - unit - 1])
            before_blind = sum(side_blinds[:index]) % Q
            through_blind = sum(side_blinds[:index + 1]) % Q
            blinds.extend([(-before_blind) % Q, through_blind])
            disclosed.append(openings[index])
            paths.append(merkle_path(records, index))
        commitments = [self.sample_bp.commit(value, blind)
                       for value, blind in zip(values, blinds)]
        return dict(side=side, indices=indices, openings=disclosed, paths=paths,
                    interval_commitments=[enc(point) for point in commitments],
                    interval_proof=proof_bytes(self.sample_bp.prove(values, blinds)))

    def prepare_account_balance(self, account, period_key=PERIOD_KEY):
        rows = self.rows(period_key)
        openings = [self.opening(row) for row in rows]
        indices = [i for i, item in enumerate(openings)
                   if item["account"] == account]
        return dict(
            account=account, indices=indices,
            balance=sum(openings[i]["amount"] for i in indices),
            amount_blind=sum(openings[i]["rho"] for i in indices) % Q,
            account_blinds=[openings[i]["sigma"] for i in indices])


def open_verifier_db(path):
    db = sqlite3.connect(str(path), isolation_level=None)
    db.row_factory = sqlite3.Row
    return db


def countersign(db, period_key, anchor, receipt_time, custodian_key):
    receipt = make_receipt(anchor, receipt_time, custodian_key)
    db.execute("INSERT INTO receipts VALUES(?,?)", (period_key, receipt))
    return receipt


def publish_beacon(db, round_number, open_time, value, beacon_key):
    message = beacon_message(round_number, open_time, value)
    signature = beacon_key.sign(message)
    db.execute("INSERT INTO beacons VALUES(?,?,?,?)",
               (round_number, open_time, value, signature))
    return signature


def public_rows(db, period_key=PERIOD_KEY):
    return [dict(row) for row in db.execute(
        "SELECT * FROM rows WHERE period_key=? ORDER BY seq", (period_key,))]


def verify_period(db, firm_public, custodian_public,
                  expected_previous_anchor_hash=PREVIOUS_ANCHOR_HASH,
                  period_key=PERIOD_KEY):
    period = db.execute(
        "SELECT * FROM periods WHERE period_key=?", (period_key,)).fetchone()
    receipt_row = db.execute(
        "SELECT receipt FROM receipts WHERE period_key=?", (period_key,)).fetchone()
    if not period or period["state"] != "CLOSED" or not receipt_row:
        return False
    try:
        header = verify_anchor(period["anchor"], firm_public)
        receipt_time = verify_receipt(
            receipt_row["receipt"], period["anchor"], custodian_public)
        rows = public_rows(db, period_key)
        records = [row_record(row) for row in rows]
        if (header["entity"] != period["entity"]
                or header["period"] != period["period_code"]
                or header["row_count"] != len(rows)
                or header["previous_anchor_hash"]
                != expected_previous_anchor_hash
                or not header["close_time"] < receipt_time
                < beacon_open_time(header["beacon_round"])
                or not parameters_valid(len(rows), int(period["total"]))
                or any(len(record) != P256_RECORD_BYTES for record in records)
                or len({row["row_id"] for row in rows}) != len(rows)
                or merkle_root(records) != header["root"]
                or chain_head(records) != header["chain_head"]):
            return False
        for row in rows:
            entity, period_code, partition, ledger, _, _ = decode_row_id(
                row["row_id"])
            if (len(row["row_id"]) != ROW_ID_BYTES or entity != ENTITY
                    or period_code != PERIOD_CODE or partition != PARTITION
                    or ledger != LEDGER):
                return False

        C, D, K = ([dec(row[name]) for row in rows]
                   for name in ("C", "D", "K"))
        links = [int.from_bytes(row["side_link"], "big") for row in rows]
        bp = BP(32, 2 * SAMPLE_SIZE)
        if not all(
                D[i] + mul(K[i], Q - 1) + mul(C[i], Q - 1)
                + mul(bp.G, OMEGA) == mul(bp.H, links[i])
                for i in range(len(rows))):
            return False
        n = len(rows)
        amount_bp, side_bp = BP(64, n), BP(32, 2 * n)
        rounds = (64 * n).bit_length() - 1
        amount_proof = parse_proof(period["amount_proof"], rounds)
        side_proof = parse_proof(period["side_proof"], rounds)
        if (amount_bp.verify(C, amount_proof)[0] is not True
                or side_bp.verify(D + K, side_proof)[0] is not True):
            return False
        total = int(period["total"])
        if (add_points(C, bp.G)
                != mul(bp.G, n * OMEGA) + mul(bp.H, int(period["sum_rho"]))):
            return False
        sum_c = add_points(C, bp.G)
        sum_d = add_points(D, bp.G)
        sum_k = add_points(K, bp.G)
        if (sum_d + mul(sum_c, Q - 1) + mul(bp.G, n * OMEGA - total)
                != mul(bp.H, int(period["sum_eta_minus_rho"]))):
            return False
        return (sum_k + sum_c + mul(bp.G, -(n * OMEGA + total))
                == mul(bp.H, int(period["sum_zeta_plus_rho"])))
    except Exception:
        return False


def verify_account_after_period(db, presentation, period_key=PERIOD_KEY):
    """Verify an account presentation after verify_period has accepted."""
    try:
        rows = public_rows(db, period_key)
        indices = presentation["indices"]
        blinds = presentation["account_blinds"]
        if (not indices or len(indices) != len(set(indices))
                or len(indices) != len(blinds)
                or any(index < 0 or index >= len(rows) for index in indices)):
            return False
        bp = BP(32, 2 * SAMPLE_SIZE)
        C = [dec(row["C"]) for row in rows]
        A = [dec(row["A"]) for row in rows]
        for index, blind in zip(indices, blinds):
            if (A[index] != mul(bp.G, presentation["account"])
                    + mul(bp.H, blind)):
                return False
        balance = presentation["balance"]
        if not (-len(indices) * OMEGA <= balance
                < len(indices) * ((1 << 64) - OMEGA)):
            return False
        return (add_points([C[index] for index in indices], bp.G)
                == mul(bp.G, balance + len(indices) * OMEGA)
                + mul(bp.H, presentation["amount_blind"]))
    except Exception:
        return False


def verify_account(db, presentation, firm_public, custodian_public,
                   expected_previous_anchor_hash=PREVIOUS_ANCHOR_HASH,
                   period_key=PERIOD_KEY):
    """Complete account verifier, including its period precondition."""
    return (verify_period(db, firm_public, custodian_public,
                          expected_previous_anchor_hash, period_key)
            and verify_account_after_period(db, presentation, period_key))


def verify_sample_after_period(db, presentation, firm_public, beacon_public,
                               audit_time=AUDIT_TIME,
                               period_key=PERIOD_KEY):
    """Verify a sampling presentation after verify_period has accepted."""
    try:
        period = db.execute(
            "SELECT * FROM periods WHERE period_key=?", (period_key,)).fetchone()
        header = verify_anchor(period["anchor"], firm_public)
        beacon = db.execute(
            "SELECT * FROM beacons WHERE round=?",
            (header["beacon_round"],)).fetchone()
        if not beacon or beacon["open_time"] != beacon_open_time(beacon["round"]):
            return False
        beacon_public.verify(
            beacon["signature"],
            beacon_message(beacon["round"], beacon["open_time"],
                           beacon["value"]))
        if audit_time < beacon["open_time"]:
            return False
        side = presentation["side"]
        if side not in ("debit", "credit"):
            return False
        total = int(period["total"])
        units = sample_distinct(period["anchor"] + beacon["value"] + b"mus",
                                total, SAMPLE_SIZE)
        rows = public_rows(db, period_key)
        records = [row_record(row) for row in rows]
        commitments = [dec(value) for value in presentation["interval_commitments"]]
        bp = BP(32, 2 * SAMPLE_SIZE)
        proof = parse_proof(presentation["interval_proof"],
                            (64 * SAMPLE_SIZE).bit_length() - 1)
        if (len(presentation["indices"]) != SAMPLE_SIZE
                or len(presentation["openings"]) != SAMPLE_SIZE
                or len(presentation["paths"]) != SAMPLE_SIZE
                or bp.verify(commitments, proof)[0] is not True):
            return False
        side_column = "D" if side == "debit" else "K"
        D = [dec(row[side_column]) for row in rows]
        prefix, point = [], mul(bp.G, 0)
        for commitment in D:
            point = point + commitment
            prefix.append(point)
        for pos, (unit, index, opening, path) in enumerate(zip(
                units, presentation["indices"], presentation["openings"],
                presentation["paths"])):
            if index < 0 or index >= len(rows):
                return False
            previous_point = prefix[index - 1] if index else mul(bp.G, 0)
            if (mul(bp.G, unit) + mul(previous_point, Q - 1)
                    != commitments[2 * pos]
                    or prefix[index] + mul(bp.G, -(unit + 1))
                    != commitments[2 * pos + 1]
                    or not verify_path(records[index], path, header["root"])):
                return False
            row = rows[index]
            amount, account = opening["amount"], opening["account"]
            if (bp.commit(amount + OMEGA, opening["rho"]) != dec(row["C"])
                    or bp.commit(account, opening["sigma"]) != dec(row["A"])
                    or bp.commit(max(amount, 0), opening["eta"]) != dec(row["D"])
                    or bp.commit(max(-amount, 0), opening["zeta"]) != dec(row["K"])):
                return False
        return True
    except Exception:
        return False


def verify_sample(db, presentation, firm_public, custodian_public,
                  beacon_public, audit_time=AUDIT_TIME,
                  expected_previous_anchor_hash=PREVIOUS_ANCHOR_HASH,
                  period_key=PERIOD_KEY):
    """Complete sample verifier, including its period precondition."""
    return (verify_period(db, firm_public, custodian_public,
                          expected_previous_anchor_hash, period_key)
            and verify_sample_after_period(db, presentation, firm_public,
                                           beacon_public, audit_time,
                                           period_key))


def main():
    firm = Ed25519PrivateKey.from_private_bytes(H(b"paperb/durable/firm").digest())
    custodian = Ed25519PrivateKey.from_private_bytes(
        H(b"paperb/durable/custodian").digest())
    beacon_key = Ed25519PrivateKey.from_private_bytes(
        H(b"paperb/durable/beacon").digest())
    master_key = H(b"paperb/durable/master").digest()
    amounts = [3, -2, 5, -6, 4, -4, 7, -7,
               2, -3, 6, -5, 1, -2, 8, -7]

    with tempfile.TemporaryDirectory(prefix="paperb-durable-") as directory:
        path = Path(directory) / "ledger.db"
        service = LedgerService(path, master_key, firm)
        service.start_period()
        for i, amount in enumerate(amounts):
            service.ingest(i, amount, 1000 + i % 4)
        duplicate_rejected = False
        try:
            service.ingest(99, 0, 9999)
            service.ingest(99, 0, 9999)
        except sqlite3.IntegrityError:
            duplicate_rejected = True
        service.db.execute("DELETE FROM rows WHERE seq=99")

        crash_rolled_back = False
        try:
            service.close_period(fail_before_commit=True)
        except RuntimeError:
            state = service.db.execute(
                "SELECT state,anchor FROM periods").fetchone()
            crash_rolled_back = (state["state"] == "OPEN"
                                 and state["anchor"] is None)

        anchor = service.close_period()
        countersign(service.db, PERIOD_KEY, anchor, RECEIPT_TIME, custodian)
        second_close_rejected = False
        try:
            service.close_period()
        except ValueError:
            second_close_rejected = True
        conflicting_receipt_rejected = False
        try:
            countersign(service.db, PERIOD_KEY, anchor, RECEIPT_TIME + 1,
                        custodian)
        except sqlite3.IntegrityError:
            conflicting_receipt_rejected = True
        publish_beacon(service.db, ROUND, BEACON_OPEN_TIME,
                       H(b"paperb/durable/beacon/value").digest(), beacon_key)
        sample = service.prepare_sample()
        credit_sample = service.prepare_sample(side="credit")
        account = service.prepare_account_balance(1000)
        service.close()

        # The verifier opens SQLite directly: it has no master/data/blinding key.
        verifier_db = open_verifier_db(path)
        restart_verifies = verify_period(
            verifier_db, firm.public_key(), custodian.public_key())
        sample_verifies = (restart_verifies and verify_sample_after_period(
            verifier_db, sample, firm.public_key(), beacon_key.public_key()))
        credit_sample_verifies = (restart_verifies
            and verify_sample_after_period(
                verifier_db, credit_sample, firm.public_key(),
                beacon_key.public_key()))
        account_verifies = (restart_verifies and verify_account_after_period(
            verifier_db, account))

        foreign = dict(account)
        foreign["indices"] = list(account["indices"])
        foreign["indices"][0] = 1
        foreign_rejected = not verify_account_after_period(verifier_db, foreign)
        pre_open_rejected = not verify_sample_after_period(
            verifier_db, sample, firm.public_key(), beacon_key.public_key(),
            audit_time=BEACON_OPEN_TIME - 1)

        original = verifier_db.execute(
            "SELECT encrypted_fields FROM rows WHERE period_key=? AND seq=0",
            (PERIOD_KEY,)).fetchone()[0]
        tampered = bytes([original[0] ^ 1]) + original[1:]
        verifier_db.execute(
            "UPDATE rows SET encrypted_fields=? WHERE period_key=? AND seq=0",
            (tampered, PERIOD_KEY))
        tamper_rejected = not verify_period(
            verifier_db, firm.public_key(), custodian.public_key())
        verifier_db.execute(
            "UPDATE rows SET encrypted_fields=? WHERE period_key=? AND seq=0",
            (original, PERIOD_KEY))

        signature = verifier_db.execute(
            "SELECT signature FROM beacons WHERE round=?", (ROUND,)).fetchone()[0]
        bad_signature = bytes([signature[0] ^ 1]) + signature[1:]
        verifier_db.execute("UPDATE beacons SET signature=? WHERE round=?",
                            (bad_signature, ROUND))
        forged_beacon_rejected = not verify_sample_after_period(
            verifier_db, sample, firm.public_key(), beacon_key.public_key())
        verifier_db.execute("UPDATE beacons SET signature=? WHERE round=?",
                            (signature, ROUND))

        previous_hash_rejected = not verify_period(
            verifier_db, firm.public_key(), custodian.public_key(),
            expected_previous_anchor_hash=b"\x00" * 32)
        records_have_fixed_width = all(
            len(row_record(row)) == P256_RECORD_BYTES
            for row in public_rows(verifier_db))
        verifier_db.close()

    checks = {
        "duplicate canonical RowID rejected": duplicate_rejected,
        "crash before COMMIT rolls close back": crash_rolled_back,
        "second period close rejected": second_close_rejected,
        "conflicting custodian receipt rejected": conflicting_receipt_rejected,
        "keyless restart verifies anchor and full proofs": restart_verifies,
        "authenticated-beacon sample verifies": sample_verifies,
        "authenticated credit-side sample verifies": credit_sample_verifies,
        "account-balance presentation verifies": account_verifies,
        "foreign account row rejected": foreign_rejected,
        "audit before beacon opening rejected": pre_open_rejected,
        "post-close row tamper rejected": tamper_rejected,
        "forged beacon rejected": forged_beacon_rejected,
        "wrong previous-anchor state rejected": previous_hash_rejected,
        "canonical records are exactly 228 bytes": records_have_fixed_width,
    }
    print("=" * 78)
    print("PAPER B DURABLE, ROLE-SEPARATED PROTOTYPE")
    print("=" * 78)
    print("  SQLite journal_mode=WAL, synchronous=FULL")
    for label, result in checks.items():
        print("  {:55} : {}".format(label, result))
    print("  all checks pass                                      : {}".format(
        all(checks.values())))
    print("  scope: durable research prototype; authenticated beacon stand-in")
    print("  scope: no production ERP, live beacon network, or independent audit")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
