#!/usr/bin/env python3
"""
Compare original scripts vs. MedLingo output using BLEU score.

Methodology follows the standard BLEU metric used by Microsoft Translator
(https://github.com/MicrosoftDocs/azure-ai-docs/.../bleu-score.md):
n-gram precision (1- to 4-grams) with a brevity penalty, computed via
sacrebleu (the reference implementation, mteval-compatible).

Usage:
    python3 bleu_compare.py <input file> [--reference COL] [--candidate COL]

Input: .xlsx or .csv with one column containing the original script
(reference) and one containing the MedLingo output (candidate).
If column names are not given, the script tries to auto-detect them,
otherwise it uses the first two columns (first = reference, second = candidate).

Output:
  - Corpus-level BLEU printed to the terminal (the headline number)
  - <input>_bleu_results.xlsx with per-row sentence BLEU scores
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import sacrebleu


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path)
    if path.suffix.lower() in (".csv", ".txt"):
        return pd.read_csv(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    sys.exit(f"Unsupported file type: {path.suffix} (use .xlsx or .csv)")


def pick_columns(df: pd.DataFrame, ref_col: str, cand_col: str):
    cols = list(df.columns)

    def find(keywords):
        for c in cols:
            if any(k in str(c).lower() for k in keywords):
                return c
        return None

    if ref_col is None:
        ref_col = find(["original", "script", "reference", "source", "doctor"])
    if cand_col is None:
        cand_col = find(["medlingo", "output", "candidate", "translation", "patient"])

    if ref_col is None or cand_col is None or ref_col == cand_col:
        ref_col, cand_col = cols[0], cols[1]
        print(f"Note: could not auto-detect columns; using first two: "
              f"reference='{ref_col}', candidate='{cand_col}'")
    return ref_col, cand_col


def interpret(score: float) -> str:
    if score < 10:
        return "Almost useless — output conveys little of the reference"
    if score < 20:
        return "Hard to get the gist"
    if score < 30:
        return "The gist is clear, but significant differences/errors"
    if score < 40:
        return "Understandable to good overlap"
    if score < 50:
        return "High-quality overlap"
    if score < 60:
        return "Very high-quality overlap"
    return "Extremely high overlap (near-identical text)"


def main():
    ap = argparse.ArgumentParser(description="BLEU comparison: original script vs MedLingo output")
    ap.add_argument("input", type=Path, help="xlsx/csv with the two columns")
    ap.add_argument("--reference", help="column with the original script")
    ap.add_argument("--candidate", help="column with the MedLingo output")
    args = ap.parse_args()

    df = load_table(args.input)
    if len(df.columns) < 2:
        sys.exit("Input file needs at least two columns (original script, MedLingo output).")

    ref_col, cand_col = pick_columns(df, args.reference, args.candidate)
    print(f"Reference (original script) column : {ref_col}")
    print(f"Candidate (MedLingo output) column : {cand_col}")

    df = df[[ref_col, cand_col]].dropna()
    refs = df[ref_col].astype(str).str.strip().tolist()
    cands = df[cand_col].astype(str).str.strip().tolist()
    n = len(refs)
    if n == 0:
        sys.exit("No usable rows after removing empty cells.")
    print(f"Rows scored: {n}")

    # Corpus-level BLEU — the number Microsoft's methodology reports.
    corpus = sacrebleu.corpus_bleu(cands, [refs])

    # Per-row sentence BLEU (smoothed, since single sentences often lack 4-gram matches)
    per_row = [
        sacrebleu.sentence_bleu(c, [r], smooth_method="exp").score
        for c, r in zip(cands, refs)
    ]

    out = df.copy()
    out["sentence_BLEU"] = [round(s, 2) for s in per_row]
    out["interpretation"] = [interpret(s) for s in per_row]
    out_path = args.input.with_name(args.input.stem + "_bleu_results.xlsx")
    summary = pd.DataFrame({
        "Metric": ["Corpus BLEU", "Rows scored", "Brevity penalty",
                   "1-gram precision", "2-gram precision", "3-gram precision", "4-gram precision",
                   "Mean sentence BLEU", "Interpretation (corpus)"],
        "Value": [round(corpus.score, 2), n, round(corpus.bp, 3),
                  *[round(p, 1) for p in corpus.precisions],
                  round(sum(per_row) / n, 2), interpret(corpus.score)],
    })
    with pd.ExcelWriter(out_path) as xl:
        out.to_excel(xl, sheet_name="Per-row scores", index=False)
        summary.to_excel(xl, sheet_name="Summary", index=False)

    print()
    print(f"Corpus BLEU: {corpus.score:.2f}   ({interpret(corpus.score)})")
    print(f"  brevity penalty: {corpus.bp:.3f}")
    print(f"  n-gram precisions (1-4): "
          + " / ".join(f"{p:.1f}" for p in corpus.precisions))
    print(f"  mean sentence BLEU: {sum(per_row)/n:.2f}")
    print(f"\nPer-row results written to: {out_path}")


if __name__ == "__main__":
    main()
