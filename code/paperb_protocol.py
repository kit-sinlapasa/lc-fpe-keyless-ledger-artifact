"""Canonical byte-level protocol encoding shared by Paper B fixtures.

The format is deliberately fixed-width.  A verifier can therefore hash and
sign the exact same bytes without relying on a language-specific serializer.
"""
import struct


ROW_ID_STRUCT = struct.Struct(">IIHHQI")
ANCHOR_PREFIX = b"LEDGER-ANCHOR/1"
ANCHOR_FIELDS = struct.Struct(">III")
ANCHOR_TAIL = struct.Struct(">QI")

ROW_ID_BYTES = ROW_ID_STRUCT.size
ENCRYPTED_FIELDS_BYTES = 40  # 12-byte nonce + 12-byte payload + 16-byte GCM tag
P256_POINT_BYTES = 33
SCALAR_BYTES = 32
ANCHOR_ENV_BYTES = (len(ANCHOR_PREFIX) + ANCHOR_FIELDS.size + 3 * 32
                    + ANCHOR_TAIL.size)
ANCHOR_BYTES = ANCHOR_ENV_BYTES + 64
RECEIPT_BYTES = ANCHOR_BYTES + 8 + 64
P256_RECORD_BYTES = (ROW_ID_BYTES + ENCRYPTED_FIELDS_BYTES
                     + 4 * P256_POINT_BYTES + SCALAR_BYTES)


def _uint(name, value, bits):
    value = int(value)
    if not 0 <= value < 1 << bits:
        raise ValueError("{} does not fit {} bits".format(name, bits))
    return value


def encode_row_id(entity, period, partition, ledger, document, line):
    """Encode (entity, period, partition, ledger, document, line)."""
    return ROW_ID_STRUCT.pack(
        _uint("entity", entity, 32),
        _uint("period", period, 32),
        _uint("partition", partition, 16),
        _uint("ledger", ledger, 16),
        _uint("document", document, 64),
        _uint("line", line, 32),
    )


def decode_row_id(value):
    if len(value) != ROW_ID_BYTES:
        raise ValueError("invalid RowID length")
    return ROW_ID_STRUCT.unpack(value)


def encode_record(row_id, encrypted_fields, C, A, D, K, side_link):
    fields = (C, A, D, K)
    if len(row_id) != ROW_ID_BYTES:
        raise ValueError("invalid RowID length")
    if len(encrypted_fields) != ENCRYPTED_FIELDS_BYTES:
        raise ValueError("invalid encrypted-field length")
    if any(len(point) != P256_POINT_BYTES for point in fields):
        raise ValueError("invalid P-256 point length")
    if len(side_link) != SCALAR_BYTES:
        raise ValueError("invalid side-link scalar length")
    record = row_id + encrypted_fields + b"".join(fields) + side_link
    if len(record) != P256_RECORD_BYTES:
        raise AssertionError("canonical record length changed")
    return record


def encode_anchor_envelope(entity, period, row_count, root, chain_head,
                           previous_anchor_hash, close_time, beacon_round):
    hashes = (root, chain_head, previous_anchor_hash)
    if any(len(value) != 32 for value in hashes):
        raise ValueError("anchor hashes must be 32 bytes")
    env = (ANCHOR_PREFIX
           + ANCHOR_FIELDS.pack(_uint("entity", entity, 32),
                                _uint("period", period, 32),
                                _uint("row_count", row_count, 32))
           + root + chain_head + previous_anchor_hash
           + ANCHOR_TAIL.pack(_uint("close_time", close_time, 64),
                              _uint("beacon_round", beacon_round, 32)))
    if len(env) != ANCHOR_ENV_BYTES:
        raise AssertionError("canonical anchor length changed")
    return env


def decode_anchor_envelope(env):
    if len(env) != ANCHOR_ENV_BYTES or not env.startswith(ANCHOR_PREFIX):
        raise ValueError("invalid anchor envelope")
    pos = len(ANCHOR_PREFIX)
    entity, period, row_count = ANCHOR_FIELDS.unpack_from(env, pos)
    pos += ANCHOR_FIELDS.size
    root, chain_head = env[pos:pos + 32], env[pos + 32:pos + 64]
    previous_anchor_hash = env[pos + 64:pos + 96]
    pos += 96
    close_time, beacon_round = ANCHOR_TAIL.unpack_from(env, pos)
    return dict(entity=entity, period=period, row_count=row_count, root=root,
                chain_head=chain_head,
                previous_anchor_hash=previous_anchor_hash,
                close_time=close_time, beacon_round=beacon_round)


def make_anchor(env, private_key):
    if len(env) != ANCHOR_ENV_BYTES:
        raise ValueError("invalid anchor envelope length")
    anchor = env + private_key.sign(env)
    if len(anchor) != ANCHOR_BYTES:
        raise AssertionError("canonical anchor length changed")
    return anchor


def verify_anchor(anchor, public_key):
    if len(anchor) != ANCHOR_BYTES:
        raise ValueError("invalid anchor length")
    env, signature = anchor[:-64], anchor[-64:]
    public_key.verify(signature, env)
    return decode_anchor_envelope(env)


def make_receipt(anchor, receipt_time, private_key):
    if len(anchor) != ANCHOR_BYTES:
        raise ValueError("invalid anchor length")
    message = anchor + struct.pack(">Q", _uint("receipt_time", receipt_time, 64))
    receipt = message + private_key.sign(message)
    if len(receipt) != RECEIPT_BYTES:
        raise AssertionError("canonical receipt length changed")
    return receipt


def verify_receipt(receipt, anchor, public_key):
    if len(receipt) != RECEIPT_BYTES or receipt[:ANCHOR_BYTES] != anchor:
        raise ValueError("invalid custodian receipt")
    message, signature = receipt[:-64], receipt[-64:]
    public_key.verify(signature, message)
    return struct.unpack(">Q", message[-8:])[0]
