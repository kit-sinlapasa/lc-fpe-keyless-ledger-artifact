#!/usr/bin/env python3
"""
Generate the rows and the derived figures of Paper B's tab:bp.

Two result files feed it. bulletproofs.txt is the original sweep at n=64;
bp_full_partition.txt is a cost-scaling ladder at n=32, and
bp_width_control.txt is the control showing the two widths cost the same at
equal n*m. The deployed verifier is mixed-width: C_i uses n_C=64 while D_i
and K_i use n_S=32, so it needs two aggregated proofs.

Why this script exists rather than a careful transcription: every earlier
defect in this table was a transcribed number that the code no longer
produced. The rows below are written by the same run that measured them, and
so are the sentences' derived figures -- the gap factor and the projected
hours -- which check_claims.py cannot vouch for because they appear in no
result file until this script puts them in one.

It refuses to read a ladder that has not finished. `done` is printed by
bp_full_partition.py only after its loop exits, so a partial file is a
partial table and this script would rather produce nothing.
"""
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = (os.path.dirname(HERE) if os.path.basename(HERE) == "code"
            else os.path.join(HERE, "..", "artifact"))
RES = os.path.join(ARTIFACT, "results")
SWEEP = os.path.join(RES, "bulletproofs.txt")
LADDER = os.path.join(RES, "bp_full_partition.txt")
OUT_ROWS = os.path.join(RES, "tab_bp_rows.tex")
OUT_FACTS = os.path.join(RES, "tab_bp_facts.txt")

ROWS = 20_000
AMOUNT_N = 64
AMOUNT_PAD_M = 32_768         # next power of two above 20,000 C_i values
SIDE_N = 32
SIDE_PAD_M = 65_536           # next power of two above 40,000 D_i,K_i values
LADDER_N = 32
NUM = r"[\d,]+(?:\.\d+)?"

# The sweep starts at n*m = 64, but the table does not have room for every
# measured point once the ladder is added, and the small end is the least
# informative part of it: the size formula is already confirmed by the rows
# above it and a sub-second prove time settles nothing about a partition. So
# the table keeps the largest MAX_ROWS points and drops the rest, which means
# it stays the same height however far the ladder climbs. Everything measured
# stays in results/ and in the artefact; only the table is thinned, and the
# caption says from where.
MAX_ROWS = 7


def f(x):
    return float(x.replace(",", ""))


def read_sweep(path, n_fixed):
    """rows of the n=64 sweep: m N prove verify size formula perrow VERIFIED"""
    out = []
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\s*(%s)\s+(%s)\s+(%s)\s+(%s)\s+(%s)\s+(%s)\s+(%s)\s+(\w+)"
                     % ((NUM,) * 7), line)
        if m and m.group(8).startswith("VERIFIED"):
            out.append(dict(n=n_fixed, m=int(f(m.group(1))), N=int(f(m.group(2))),
                            prove=f(m.group(3)), verify=f(m.group(4)),
                            size=int(f(m.group(5)))))
    return out


def read_ladder(path):
    """rows of the n=32 cost ladder: m N setup prove verify size sound peak"""
    txt = open(path, encoding="utf-8").read()
    if not re.search(r"^\s*done\s*$", txt, re.M):
        sys.exit("bp_full_partition.txt has no 'done' line: the ladder is still "
                 "running or died. Refusing to build a table from a partial run.")
    out = []
    for line in txt.split("\n"):
        m = re.match(r"\s*(%s)\s+(%s)\s+(%s)\s+(%s)\s+(%s)\s+(%s)\s+(\w+)\s+(%s)M"
                     % ((NUM,) * 6 + (NUM,)), line)
        if m and m.group(7) == "yes":
            out.append(dict(n=LADDER_N, m=int(f(m.group(1))), N=int(f(m.group(2))),
                            prove=f(m.group(4)), verify=f(m.group(5)),
                            size=int(f(m.group(6)))))
    return out


def tex_int(v):
    return "{:,}".format(v).replace(",", "{,}")


def main():
    all_rows = read_sweep(SWEEP, 64) + read_ladder(LADDER)
    if not all_rows:
        sys.exit("no verified rows parsed; check the result-file format")
    all_rows.sort(key=lambda r: (r["N"], r["n"]))
    rows = all_rows[-MAX_ROWS:]
    dropped = len(all_rows) - len(rows)
    min_shown = rows[0]["N"]

    # Units live in the column heads, not in every cell: with fourteen rows the
    # repeated \,s and \,B cost width the table does not have.
    lines = []
    for r in rows:
        lines.append("{:>2} & {:>7} & {:>10} & {:>9.2f} & {:>9.2f} & "
                     "{}\\\\".format(
                         r["n"], tex_int(r["m"]), tex_int(r["N"]),
                         r["prove"], r["verify"], tex_int(r["size"])))

    # Facts are written as atomic key = value pairs because the prose quotes
    # them too, and update_bp_prose.py has to read them without re-deriving
    # anything. A derived number that only ever exists inside a sentence is a
    # number check_claims.py cannot vouch for.
    top = rows[-1]
    batch_N = AMOUNT_N * AMOUNT_PAD_M
    assert batch_N == SIDE_N * SIDE_PAD_M
    full_N = 2 * batch_N
    reached_batch_cost = top["N"] >= batch_N
    facts = ["top_n = %d" % top["n"],
             "top_m = %d" % top["m"],
             "top_nm = %d" % top["N"],
             "top_prove_s = %.2f" % top["prove"],
             "top_verify_s = %.2f" % top["verify"],
             "top_size_B = %d" % top["size"],
             "doublings = %d" % round(math.log2(top["N"] / all_rows[0]["N"])),
             "batch_padded_nm = %d" % batch_N,
             "full_padded_nm = %d" % full_N,
             "real_values = %d" % (3 * ROWS),
             "batch_cost_measured = %s" % ("yes" if reached_batch_cost else "no"),
             # A synthetic cost rung is not the full verifier over real C,D,K.
             "full_measured = no"]

    if top["N"] < full_N:
        batch_gap = batch_N / top["N"]
        full_gap = full_N / top["N"]
        k = math.ceil(math.log2(batch_N))
        size = 33 * (4 + 2 * k) + 160
        total_size = 2 * size
        facts += ["batch_gap = %.0f" % batch_gap,
                  "full_work_gap = %.0f" % full_gap,
                  "extrap_prove_h = %.1f" % (top["prove"] * full_gap / 3600),
                  "extrap_verify_h = %.1f" % (top["verify"] * full_gap / 3600),
                  "extrap_k = %d" % k,
                  "batch_size_B = %d" % size,
                  "extrap_size_B = %d" % total_size,
                  "extrap_B_per_amount = %.3g" % (total_size / (3 * ROWS)),
                  "extrap_B_per_row = %.3g" % (total_size / ROWS)]
        lines.append("\\midrule")
        lines.append("{:>2} & {:>7} & {:>10} & \\emph{{extrap.}} & \\emph{{extrap.}} & "
                     "{} ($C$)\\\\".format(
                         AMOUNT_N, tex_int(AMOUNT_PAD_M), tex_int(batch_N),
                         tex_int(size)))
        lines.append("{:>2} & {:>7} & {:>10} & \\emph{{extrap.}} & \\emph{{extrap.}} & "
                     "{} ($D,K$)\\\\".format(
                         SIDE_N, tex_int(SIDE_PAD_M), tex_int(batch_N),
                         tex_int(size)))
        lines.append("\\multicolumn{2}{r}{total} & %s & \\emph{extrap.} & "
                     "\\emph{extrap.} & \\textbf{%s}\\\\" %
                     (tex_int(full_N), tex_int(total_size)))

    facts += ["rows_shown = %d" % len(rows),
              "rows_measured = %d" % len(all_rows),
              "rows_dropped = %d" % dropped,
              "min_nm_shown = %d" % min_shown]

    # per-doubling ratios over every measured point, thinned or not, at both
    # widths: consecutive n*m that differ by exactly a factor of two
    ratios = []
    for i in range(len(all_rows) - 1):
        if all_rows[i + 1]["N"] == 2 * all_rows[i]["N"]:
            ratios.append(all_rows[i + 1]["prove"] / all_rows[i]["prove"])
    if ratios:
        facts += ["ratio_min = %.2f" % min(ratios),
                  "ratio_max = %.2f" % max(ratios),
                  "ratio_mean = %.2f" % (sum(ratios) / len(ratios)),
                  "ratio_n = %d" % len(ratios)]

    open(OUT_ROWS, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    open(OUT_FACTS, "w", encoding="utf-8").write("\n".join(facts) + "\n")
    print("wrote %s (%d rows)" % (OUT_ROWS, len(lines)))
    print("wrote %s" % OUT_FACTS)
    print()
    print("\n".join(facts))


if __name__ == "__main__":
    main()
