#!/usr/bin/env python3
"""
Claim checker: every number printed in a paper must exist in a result file.

WHY THIS EXISTS
Across five review rounds the single most common defect in these manuscripts
was not a wrong argument but a stale number: the paper quoting a figure that
the code no longer produces. Every instance was invisible to `pdflatex`, to
the lint pass and to a careful re-reading, because none of those knows what
the scripts computed. Concretely, the following were all found by a reader
rather than by us:

  * chart-size floor fractions (12.5 / 28.2 / 45.4 %) left over from the
    largest-remainder allocator after the exact one replaced it;
  * `tab:sampling` prover times (66.64 / 35.84 s) from before the credit-side
    commitment was added;
  * a 140-byte anchor in two places after the envelope became 199 bytes;
  * "836 rows/s" and "3.9 s" from a superseded benchmark run;
  * a cross-check quoting 0.0194 against 0.0192, neither of which any script
    had produced for months.

This script closes that class. It extracts every number the manuscripts
assert and every number the canonical result files contain, and reports the
manuscript numbers that appear in no result file.

WHAT IT DOES NOT CATCH, stated so nobody trusts a clean run too far:

  * a real number in the wrong cell. It has no idea which table a number
    belongs to, only whether some script produced it somewhere.
  * a paper and a result file that are stale *together*. The check is
    paper-against-file; if the file was not regenerated after the code
    changed, both agree and both are wrong. That is what the rerun
    discipline and a two-run reproducibility check are for, not this.
  * a rounding coincidence. The matcher accepts a paper number that rounds
    to a file number, so 74 matches 74.3 -- and would also match an
    unrelated 73.8 elsewhere in the corpus.
  * a number that is a *conclusion drawn from* the measurements rather than
    one of them. Writing "prover time multiplies by 2.00 per doubling at
    every step" passed this check while being false -- the measured ratios
    run 1.94 to 2.11 -- because 2.00 was never claimed to come from a file.
    Summary statistics the paper computes in prose (ratios, factors, means,
    "roughly N times") have to be recomputed by hand from the table they
    summarise; the checker only sees the table.
  * a benchmark parameter borrowed from a different configuration. The
    extrapolation row of tab:bp was computed at n=64, the sweep's width,
    while the deployment fixes n=32, so it described a workload twice the
    real one. Every number in the row was arithmetically correct and the
    check passed. The rule this needs is not in the checker: a benchmark
    parameter the prose states must be greppable in the script that produced
    the table.

Derived numbers in general pass this check by rounding coincidence -- 1.92,
117, 0.03 and 0.08 all match something in the corpus without being sourced by
anything. Whitelisting them at least records the arithmetic; the generic
decimals are deliberately left out, because the whitelist is global and
suppressing "0.03" everywhere costs more than it buys.

It catches the number that is not real anywhere, which is what kept
happening.

USAGE
    python check_claims.py            # report unsourced numbers
    python check_claims.py --strict   # exit 1 if any are found (for a gate)

Numbers that are legitimately not measurements -- field widths, group
orders, standard identifiers, budgets we chose -- live in WHITELIST below
with the reason they are there. Adding to it is a deliberate act: if a
number is whitelisted it is no longer checked, so the reason has to say why
no script could produce it.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PAPERS = [os.path.join(HERE, "..", "paper", "paperA.tex"),
          os.path.join(HERE, "..", "paper", "paperB.tex")]

# --------------------------------------------------------------- whitelist
# value -> why it cannot come from a result file. Matched on the normalised
# string, so "10" covers 10, 10.0 and 10%.
WHITELIST = {
    # standard / structural constants
    "100": "FF1 minimum domain in SP 800-38G (2016), and percentages",
    "1000000": "the 10^6 floor of draft Rev.1",
    "10000": "alpha, the four-digit account field",
    "256": "AES-256, SHA-256, bit widths",
    "128": "AES-128 in the NIST KAT list",
    "192": "AES-192 in the NIST KAT list",
    "96": "AEAD nonce width in bits",
    "64": "document-number field width, Feistel half widths, n=64 range proofs",
    "32": "plaintext row bytes, spectral dimensions, n=32 range proofs",
    "33": "compressed P-256 point",
    "24": "line-number field width in bits",
    "16": "AEAD tag bytes, refinement/dimension counts",
    "12": "AEAD nonce bytes",
    "7": "ledger index width in bits",
    "9": "digits in the lifted triple; NIST publishes nine KAT vectors",
    "6": "RowID field count",
    "4": "account-code digits",
    "3": "cost-centre digits",
    "2": "document-type digits",
    "1": "counts and indices",
    "0": "counts and indices",
    "5": "seed counts, line-count mix",
    "8": "pool sizes",
    "40": "the offset Omega = 2^40",
    "48": "an alternative range-proof width",
    "512": "Paillier ciphertext bytes, PRF output bits",
    "2048": "Paillier modulus bits",
    "480": "inclusion-proof bytes, fixed by the tree depth",
    "199": "anchor envelope bytes, fixed by the field widths",
    "271": "anchor with countersignature, fixed by the field widths",
    "176": "anchored record bytes, fixed by the field widths",
    "132": "four commitments per row, fixed by the point width",
    "91": "LC-FPE bytes per row, fixed by the field widths",
    "60": "AES-GCM bytes per row, fixed by the field widths",
    # thresholds and budgets we chose rather than measured
    "0.05": "the ARI budget, a choice stated as such",
    "0.1": "sensitivity threshold quoted against the budget",
    "0.2": "sensitivity threshold quoted against the budget",
    "0.95": "confidence level",
    "95": "confidence level",
    "3.0": "MUS reliability factor at 95% with no expected misstatement",
    # standards, years, identifiers
    "800": "SP 800-38G", "38": "SP 800-38G", "1": "Revision 1",
    "2016": "year", "2019": "year", "2025": "year", "2026": "year",
    "2022": "year", "2012": "year", "2459": "CVE-2012-2459",
    "315": "ISA 315", "530": "ISA 530", "500": "ISA 500",
    "1105": "AS 1105", "2315": "AS 2315", "3161": "RFC 3161",
    "6962": "RFC 6962", "9380": "RFC 9380", "1066": "ePrint 2017/1066",
    "1068": "ePrint 2017/1068",
    # facts taken from cited work, not measured here
    "39": "minimum length of HCTR2-FP, quoted from Denis (ePrint 2025/1888)",
    # quantities DERIVED from result-file numbers rather than printed by a
    # script. Each entry names the arithmetic so a reader can redo it.
    "22": "ratio of Delta 0.2223 (g_drift, delta=2) to 0.0102 = m/2K",
    "21": "26.666 s (LC-FPE) minus 5.688 s (per-field FF1), both in tab_base_rows",
    "111": "886 B / 8 aggregated values, from tab:bp",
    "66": "two P-256 points, 2 x 33 B, the per-doubling growth of the proof",
    "1678": "aggregated proof size at nm = 1.92 M (60,000 values at n=32),"
            " k = ceil(log2(1.92e6)) = 21, from 33(4+2k)+160 = 1,678;"
            " labelled in the table as extrapolated, not measured",
    "1.92": "60,000 values at the deployed n=32 is 1.92 M committed bits;"
            " the sweep in bulletproofs.txt is at n=64 and bp_width_control.txt"
            " shows equal nm costs the same either way",
    "117": "1.92e6 / 16,384, the extrapolation factor beyond the largest"
           " measured proof; the projected 4 h prove / 2.5 h verify are"
           " 126.44 s and 78.33 s times this",
}

# numbers appearing inside these commands are references, not claims
SKIP_CMD = re.compile(r"\\(?:cite|ref|eqref|label|input|include|section|"
                      r"subsection|bibitem|documentclass|usepackage|thanks|"
                      r"setlength|tabcolsep|pgfplotsset|compat|"
                      r"IEEEsetlabelwidth|addlegendentry)"
                      r"\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}?")
# standards, RFCs, CVEs, years and plot geometry are identifiers, not claims
SKIP_REF = re.compile(
    r"RFC~?\s?\d+|ISA~?\s?\d+|AS~?\s?\d+|SP~?\s?800-38G|CVE-\d+-\d+|"
    r"Revision~?\s?\d|Rev\.~?\s?\d|ePrint\s*\d+/\d+|\b(?:19|20)\d{2}\b|"
    r"[xy](?:min|max)=[-\d.]+|domain=[\d.:]+|samples=\d+|"
    r"P-?256|SHA-?\d+|AES-?\d+|Ed\d+|\d+\s*(?:st|nd|rd|th)\b")
MATH_POWER = re.compile(r"\^\{?-?[\d.]+\}?")          # 10^{6}, 2^{-256}
SEP = re.compile(r"(?<=\d)(?:\{,\}|,)(?=\d\d\d)")     # 145{,}035 and 1,150
NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.]*[a-zA-Z])")


def norm(tok):
    """Canonical numeric string."""
    try:
        v = float(tok)
    except ValueError:
        return None
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(round(v, 10)).rstrip("0").rstrip(".")


def numbers_in_results():
    """Every numeric token any canonical result file contains."""
    vals = set()
    if not os.path.isdir(RESULTS):
        return vals
    for fn in sorted(os.listdir(RESULTS)):
        if not fn.endswith((".txt", ".tex")):
            continue
        with open(os.path.join(RESULTS, fn), encoding="utf-8", errors="replace") as fh:
            for tok in NUM.findall(SEP.sub("", fh.read())):
                n = norm(tok)
                if n is not None:
                    vals.add(n)
    return vals


def matches(paper_val, file_vals):
    """A paper number is sourced if some file number equals it, rounds to it,
    or is its percent/fraction counterpart."""
    try:
        p = float(paper_val)
    except ValueError:
        return True
    dec = len(paper_val.split(".")[1]) if "." in paper_val else 0
    for cand in (p, p / 100.0, p * 100.0):
        for f in file_vals:
            try:
                fv = float(f)
            except ValueError:
                continue
            if abs(fv - cand) < 1e-12:
                return True
            # the paper rounds what a script printed: 74 for 74.3, 166 for
            # 166.1, 0.042 for 0.0424. Integers round too, which is why the
            # dec > 0 guard this once carried was wrong.
            if round(fv, dec) == round(cand, dec):
                return True
            # the file may hold the fraction where the paper prints a percent
            if abs(fv * 100.0 - cand) < 10 ** (-dec) / 2 if dec else False:
                return True
            if abs(fv / 100.0 - cand) < 10 ** (-dec) / 2 if dec else False:
                return True
    return False


def scan(path, file_vals):
    out = []
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    in_bib = False
    for i, raw in enumerate(lines, 1):
        if "\\begin{thebibliography}" in raw:
            in_bib = True
        if in_bib:
            continue
        line = raw.split("%")[0] if not raw.lstrip().startswith("%") else ""
        if not line.strip():
            continue
        line = SKIP_CMD.sub(" ", line)
        line = SKIP_REF.sub(" ", line)
        line = MATH_POWER.sub(" ", line)
        line = SEP.sub("", line)
        for tok in NUM.findall(line):
            n = norm(tok)
            if n is None or n in WHITELIST:
                continue
            if not matches(n, file_vals):
                out.append((i, n, raw.strip()[:96]))
    return out


def main():
    strict = "--strict" in sys.argv
    file_vals = numbers_in_results()
    print("claim checker: {:,} distinct numbers across {} result files".format(
        len(file_vals), len([f for f in os.listdir(RESULTS)
                             if f.endswith((".txt", ".tex"))])))
    total = 0
    for p in PAPERS:
        if not os.path.exists(p):
            continue
        bad = scan(p, file_vals)
        total += len(bad)
        print("\n{}: {} unsourced number(s)".format(os.path.basename(p), len(bad)))
        for ln, val, ctx in bad:
            print("  line {:>5}  {:>12}   {}".format(ln, val, ctx))
    print("\n{} unsourced number(s) in total".format(total))
    if strict and total:
        print("FAIL: every number a manuscript asserts must appear in a result file,")
        print("      or be whitelisted in check_claims.py with a reason.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
