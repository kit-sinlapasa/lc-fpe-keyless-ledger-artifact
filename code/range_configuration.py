#!/usr/bin/env python3
"""Emit the deployed mixed-width range-proof configuration for Paper B."""
import math
import os

ROWS = 20_000
AMOUNT_N = 64
AMOUNT_VALUES = ROWS
AMOUNT_PAD_M = 1 << math.ceil(math.log2(AMOUNT_VALUES))
SIDE_N = 32
SIDE_VALUES = 2 * ROWS
SIDE_PAD_M = 1 << math.ceil(math.log2(SIDE_VALUES))


def p256_proof_size(bit_values):
    k = math.ceil(math.log2(bit_values))
    return 33 * (4 + 2 * k) + 160


def main():
    amount_bits = AMOUNT_N * AMOUNT_PAD_M
    side_bits = SIDE_N * SIDE_PAD_M
    assert amount_bits == side_bits
    each_size = p256_proof_size(amount_bits)
    lines = [
        "rows = %d" % ROWS,
        "amount_width_bits = %d" % AMOUNT_N,
        "amount_real_values = %d" % AMOUNT_VALUES,
        "amount_padded_values = %d" % AMOUNT_PAD_M,
        "amount_padded_bit_values = %d" % amount_bits,
        "side_width_bits = %d" % SIDE_N,
        "side_real_values = %d" % SIDE_VALUES,
        "side_padded_values = %d" % SIDE_PAD_M,
        "side_padded_bit_values = %d" % side_bits,
        "full_real_values = %d" % (AMOUNT_VALUES + SIDE_VALUES),
        "full_padded_bit_values = %d" % (amount_bits + side_bits),
        "proof_size_each_B = %d" % each_size,
        "proof_size_total_B = %d" % (2 * each_size),
        "proof_size_B_per_row = %.4f" % (2 * each_size / ROWS),
    ]
    here = os.path.dirname(os.path.abspath(__file__))
    artifact = (os.path.dirname(here) if os.path.basename(here) == "code"
                else os.path.join(here, "..", "artifact"))
    out = os.path.join(artifact, "results", "range_configuration.txt")
    with open(out, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
