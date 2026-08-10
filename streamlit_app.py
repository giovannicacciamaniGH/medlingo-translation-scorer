"""
LLM Translation Scorer — Streamlit app.

Upload a spreadsheet with the original script and the LLM output;
get corpus-level and per-sentence translation scores:
BLEU, chrF, TER (sacrebleu), semantic similarity (sentence embeddings,
per TextSim_MTQE) and COMET (Unbabel wmt22-comet-da).

Run locally:  streamlit run streamlit_app.py
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu
import streamlit as st

st.set_page_config(page_title="LLM Translation Scorer",
                   page_icon="🩺", layout="wide")


# ---------------------------------------------------------------- models

@st.cache_resource(show_spinner="Loading semantic similarity model…")
def embedder():
    from sentence_transformers import SentenceTransformer
    # Same model as https://github.com/fivehills/TextSim_MTQE
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_resource(show_spinner="Loading COMET model (large — first time takes a while)…")
def comet_model():
    from comet import download_model, load_from_checkpoint
    return load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))


# ---------------------------------------------------------------- scoring

def interpret(score):
    if score < 10: return "Almost no overlap"
    if score < 20: return "Low overlap"
    if score < 30: return "Gist preserved, heavily reworded"
    if score < 40: return "Moderate overlap"
    if score < 50: return "High overlap"
    if score < 60: return "Very high overlap"
    return "Near-identical"


def meaning_verdict(sim):
    if sim >= 0.75: return "Meaning preserved"
    if sim >= 0.55: return "Mostly preserved — review"
    return "Possible meaning change"


@st.cache_data(show_spinner=False)
def score_pairs(srcs: tuple, cands: tuple, gts: tuple, use_comet: bool):
    """srcs = original scripts; cands = LLM output; gts = ground-truth
    human translations (optional).

    BLEU and COMET are single scores comparing LLM against the ground
    truth (or the original when no ground truth is given). chrF, TER and
    semantic similarity are computed against the ORIGINAL — for LLM,
    and (when provided) also for the ground truth, so machine and human can
    be compared on the same basis."""
    srcs, cands, gts = list(srcs), list(cands), list(gts)
    has_gt = len(gts) > 0
    bleu_refs = gts if has_gt else srcs

    corpus = sacrebleu.corpus_bleu(cands, [bleu_refs])
    corpus_chrf = sacrebleu.corpus_chrf(cands, [srcs])
    corpus_ter = sacrebleu.corpus_ter(cands, [srcs])
    if has_gt:
        gt_chrf = sacrebleu.corpus_chrf(gts, [srcs])
        gt_ter = sacrebleu.corpus_ter(gts, [srcs])

    model = embedder()

    def encode_norm(texts):
        e = model.encode(texts, batch_size=64, show_progress_bar=False)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    src_emb = encode_norm(srcs)
    cosines = np.sum(src_emb * encode_norm(cands), axis=1).clip(-1, 1)
    if has_gt:
        gt_cosines = np.sum(src_emb * encode_norm(gts), axis=1).clip(-1, 1)

    comet_scores, comet_system = None, None
    if use_comet:
        refs = gts if has_gt else srcs
        data = [{"src": s_, "mt": c, "ref": r}
                for s_, c, r in zip(srcs, cands, refs)]
        out = comet_model().predict(data, batch_size=8, gpus=0,
                                    num_workers=1, progress_bar=False)
        comet_scores = list(out.scores)
        comet_system = float(out.system_score)

    rows = []
    for i, (s_, c) in enumerate(zip(srcs, cands)):
        s = sacrebleu.sentence_bleu(c, [bleu_refs[i]],
                                    smooth_method="exp").score
        sim = float(cosines[i])
        row = {"#": i + 1, "Original script": s_}
        if has_gt:
            row["Ground truth"] = gts[i]
        row.update({"LLM output": c,
                    "BLEU": round(s, 1), "Wording": interpret(s),
                    "Semantic (%)": round(sim * 100),
                    "Meaning": meaning_verdict(sim),
                    "chrF": round(sacrebleu.sentence_chrf(c, [s_]).score, 1),
                    "TER": round(sacrebleu.sentence_ter(c, [s_]).score, 1)})
        if has_gt:
            g = gts[i]
            gt_sent_bleu = sacrebleu.sentence_bleu(
                g, [s_], smooth_method="exp").score
            row["Wording (GT)"] = interpret(gt_sent_bleu)
            row["Semantic GT (%)"] = round(float(gt_cosines[i]) * 100)
            row["Meaning (GT)"] = meaning_verdict(float(gt_cosines[i]))
            row["chrF (GT)"] = round(sacrebleu.sentence_chrf(g, [s_]).score, 1)
            row["TER (GT)"] = round(sacrebleu.sentence_ter(g, [s_]).score, 1)
        if comet_scores is not None:
            row["COMET"] = round(comet_scores[i] * 100)
        rows.append(row)

    summary = {"bleu": corpus.score, "bleu_label": interpret(corpus.score),
               "bp": corpus.bp, "precisions": list(corpus.precisions),
               "chrf": corpus_chrf.score, "ter": corpus_ter.score,
               "sem_mean": float(np.mean(cosines)),
               "sent_bleu_mean": float(np.mean([r["BLEU"] for r in rows])),
               "comet": comet_system,
               "gt_chrf": gt_chrf.score if has_gt else None,
               "gt_ter": gt_ter.score if has_gt else None,
               "gt_sem_mean": float(np.mean(gt_cosines)) if has_gt else None}
    return pd.DataFrame(rows), summary


def autodetect(cols):
    def find(keywords, exclude=()):
        for c in cols:
            if c in exclude:
                continue
            if any(k in str(c).lower() for k in keywords):
                return c
        return None
    src = find(["original", "script", "source", "doctor", "english",
                "dialogue", "dialog"])
    cand = find(["llm", "medlingo", "output", "candidate", "generated",
                 "gpt", "ai"], exclude=(src,))
    gt = find(["ground", "truth", "gold", "human", "reference"],
              exclude=(src, cand))
    # Fill any undetected role with the first unclaimed column — never
    # overwrite a role that was already detected.
    if src is None:
        src = next((c for c in cols if c not in (cand, gt)), cols[0])
    if cand is None:
        cand = next((c for c in cols if c not in (src, gt)), cols[1])
    if src == cand:
        src, cand, gt = cols[0], cols[1], None
    return src, cand, gt


# ---------------------------------------------------------------- UI

st.title("🩺 LLM Translation Scorer")
st.caption("Upload a spreadsheet with the original script, the LLM output, "
           "and — optionally — a ground-truth human reference. You get overall "
           "translation scores (BLEU, chrF, TER, semantic similarity, COMET) and "
           "a score for every sentence. With a ground truth, LLM is scored "
           "against the human reference; without one, it is scored against the "
           "original script.")

uploaded = st.file_uploader(
    "Excel or CSV with two columns (original script, LLM output) "
    "or three (+ ground truth)",
    type=["xlsx", "xlsm", "xls", "csv", "tsv"])

use_comet = st.toggle("Include COMET (slower; needs the 2 GB model)", value=True)

if uploaded:
    suffix = Path(uploaded.name).suffix.lower()
    try:
        if suffix in (".xlsx", ".xlsm", ".xls"):
            df = pd.read_excel(uploaded)
        elif suffix == ".tsv":
            df = pd.read_csv(uploaded, sep="\t")
        else:
            df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()

    if len(df.columns) < 2:
        st.error("The file needs at least two columns "
                 "(original script and LLM output).")
        st.stop()

    cols = list(df.columns)
    src_default, cand_default, gt_default = autodetect(cols)
    c1, c2, c3 = st.columns(3)
    src_col = c1.selectbox("Original script (source) column", cols,
                           index=cols.index(src_default))
    cand_col = c2.selectbox("LLM output column", cols,
                            index=cols.index(cand_default))
    NONE = "— none (score against the original) —"
    gt_opts = [NONE] + cols
    gt_col = c3.selectbox("Ground truth (human reference) column — optional",
                          gt_opts,
                          index=gt_opts.index(gt_default) if gt_default else 0)
    gt_col = None if gt_col == NONE else gt_col
    if len({src_col, cand_col, gt_col} - {None}) < (3 if gt_col else 2):
        st.error("The selected columns must all be different.")
        st.stop()

    use_cols = [src_col, cand_col] + ([gt_col] if gt_col else [])
    sub = df[use_cols].dropna()
    series = [sub[c].astype(str).str.strip() for c in use_cols]
    mask = np.logical_and.reduce([sr != "" for sr in series])
    srcs = series[0][mask].tolist()
    cands = series[1][mask].tolist()
    gts = series[2][mask].tolist() if gt_col else []
    if not srcs:
        st.error("No usable rows (empty cells were removed).")
        st.stop()

    if gt_col:
        st.caption("**3-column mode:** BLEU compares LLM against the "
                   "**ground truth**, and COMET uses its full triplet (source = "
                   "original, translation = LLM, reference = ground truth) "
                   "— one score each. chrF, TER and semantic similarity are "
                   "computed **against the original** for both translations — "
                   "LLM vs original and ground truth vs original — so "
                   "machine and human can be compared on the same basis.")
    else:
        st.caption("**2-column mode:** no ground truth selected — all scores "
                   "compare LLM against the original script (for COMET, the "
                   "original serves as both source and reference).")

    with st.spinner(f"Scoring {len(srcs)} sentences…"):
        table, s = score_pairs(tuple(srcs), tuple(cands), tuple(gts), use_comet)

    # ---- headline scores
    ref_name = f"“{gt_col}” (ground truth)" if gt_col else f"“{src_col}” (original)"
    vs_orig = f"Compares “{cand_col}” vs “{src_col}” (original)."

    if gt_col:
        # Group 1 — single scores (computed once, against the ground truth)
        st.markdown(f"#### 1️⃣ Single scores — “{cand_col}” vs “{gt_col}”")
        c1, c2, _ = st.columns(3)
        c1.metric("Overall BLEU (corpus)", f"{s['bleu']:.1f}",
                  help=f"{s['bleu_label']}. Computed once: “{cand_col}” vs "
                       f"“{gt_col}” (ground truth). Word-sequence overlap.")
        if s["comet"] is not None:
            c2.metric("COMET", f"{s['comet'] * 100:.0f}",
                      help=f"Computed once, full triplet: source = “{src_col}”, "
                           f"translation = “{cand_col}”, reference = “{gt_col}”. "
                           "0–100, higher = better quality.")

        # Groups 2 & 3 — same metrics, two comparisons, aligned columns
        st.markdown(f"#### 2️⃣ LLM vs original — “{cand_col}” vs “{src_col}”")
        m1, m2, m3 = st.columns(3)
        m1.metric("Mean semantic similarity", f"{s['sem_mean'] * 100:.0f}%",
                  help=f"{vs_orig} Meaning similarity from sentence embeddings.")
        m2.metric("chrF (corpus)", f"{s['chrf']:.1f}",
                  help=f"{vs_orig} Character n-gram overlap.")
        m3.metric("TER (corpus, lower = closer)", f"{s['ter']:.1f}",
                  help=f"{vs_orig} Edits needed to match the original.")

        gvs = f"Compares “{gt_col}” vs “{src_col}” (original)."
        st.markdown(f"#### 3️⃣ Ground truth vs original — “{gt_col}” vs "
                    f"“{src_col}” (human benchmark)")
        g1, g2, g3 = st.columns(3)
        g1.metric("Mean semantic similarity", f"{s['gt_sem_mean'] * 100:.0f}%",
                  help=f"{gvs} Meaning similarity from sentence embeddings.")
        g2.metric("chrF (corpus)", f"{s['gt_chrf']:.1f}",
                  help=f"{gvs} Character n-gram overlap.")
        g3.metric("TER (corpus, lower = closer)", f"{s['gt_ter']:.1f}",
                  help=f"{gvs} Edits needed to match the original.")
    else:
        labels = ["Overall BLEU (corpus)", "Mean semantic similarity",
                  "chrF (corpus)", "TER (corpus, lower = closer)"]
        values = [f"{s['bleu']:.1f}", f"{s['sem_mean'] * 100:.0f}%",
                  f"{s['chrf']:.1f}", f"{s['ter']:.1f}"]
        helps = [f"{s['bleu_label']}. {vs_orig} Word-sequence overlap.",
                 f"{vs_orig} Meaning similarity from sentence embeddings.",
                 f"{vs_orig} Character n-gram overlap.",
                 f"{vs_orig} Edits needed to match the original."]
        if s["comet"] is not None:
            labels.insert(2, "COMET")
            values.insert(2, f"{s['comet'] * 100:.0f}")
            helps.insert(2, f"Full triplet with the original as both source "
                            f"and reference (no ground truth selected). "
                            "0–100, higher = better quality.")
        for col, lab, val, hlp in zip(st.columns(len(labels)), labels, values,
                                      helps):
            col.metric(lab, val, help=hlp)

    with st.expander("More statistics"):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sentences scored", len(srcs))
        m2.metric("Mean sentence BLEU", f"{s['sent_bleu_mean']:.1f}")
        m3.metric("Brevity penalty", f"{s['bp']:.2f}")
        m4.metric("1–4-gram precision",
                  " / ".join(f"{p:.0f}" for p in s["precisions"]))

    # ---- filters (chips)
    st.subheader("Per-sentence scores")
    f1, f2 = st.columns(2)
    verdict_opts = ["All"] + [v for v in ["Possible meaning change",
                                          "Mostly preserved — review",
                                          "Meaning preserved"]
                              if v in set(table["Meaning"])]
    label_opts = ["All"] + [l for l in ["Almost no overlap", "Low overlap",
                                        "Gist preserved, heavily reworded",
                                        "Moderate overlap", "High overlap",
                                        "Very high overlap", "Near-identical"]
                            if l in set(table["Wording"])]
    pick_verdict = f1.radio("Filter by meaning", verdict_opts, horizontal=True)
    pick_label = f2.radio("Filter by wording overlap (BLEU)", label_opts,
                          horizontal=True)

    view = table
    if pick_verdict != "All":
        view = view[view["Meaning"] == pick_verdict]
    if pick_label != "All":
        view = view[view["Wording"] == pick_label]
    st.caption(f"{len(view)} of {len(table)} sentences shown")

    WORDING_COLORS = {
        "Almost no overlap":               "background-color:#ffebe9; color:#cf222e",
        "Low overlap":                     "background-color:#ffebe9; color:#cf222e",
        "Gist preserved, heavily reworded":"background-color:#fff1e5; color:#bc4c00",
        "Moderate overlap":                "background-color:#fff8c5; color:#7d4e00",
        "High overlap":                    "background-color:#ddf4ff; color:#0969da",
        "Very high overlap":               "background-color:#dafbe1; color:#1a7f37",
        "Near-identical":                  "background-color:#dafbe1; color:#1a7f37",
    }
    MEANING_COLORS = {
        "Possible meaning change":   "background-color:#ffebe9; color:#cf222e",
        "Mostly preserved — review": "background-color:#fff8c5; color:#7d4e00",
        "Meaning preserved":         "background-color:#dafbe1; color:#1a7f37",
    }
    chip = "; border-radius:999px; text-align:center; font-weight:600"
    wording_cols = [c for c in ("Wording", "Wording (GT)") if c in view.columns]
    meaning_cols = [c for c in ("Meaning", "Meaning (GT)") if c in view.columns]
    styled = view.style.map(
        lambda v: WORDING_COLORS.get(v, "") + chip, subset=wording_cols
    ).map(
        lambda v: MEANING_COLORS.get(v, "") + chip, subset=meaning_cols
    ).format({k: v for k, v in {"BLEU": "{:.1f}", "chrF": "{:.1f}",
                                "TER": "{:.1f}", "Semantic (%)": "{:.0f}",
                                "chrF (GT)": "{:.1f}", "TER (GT)": "{:.1f}",
                                "Semantic GT (%)": "{:.0f}",
                                "COMET": "{:.0f}"}.items()
              if k in view.columns}, na_rep="")
    bleu_vs = f"Compares “{cand_col}” vs {ref_name}."
    col_help = {
        "BLEU": st.column_config.NumberColumn(
            "BLEU", help=f"Single score. {bleu_vs} Word-sequence overlap, "
                         "0–100."),
        "Wording": st.column_config.TextColumn(
            "Wording", help=f"Judgment of the BLEU score. {bleu_vs}"),
        "Semantic (%)": st.column_config.NumberColumn(
            "Semantic (%)", help=f"{vs_orig} Meaning similarity, 0–100%."),
        "Meaning": st.column_config.TextColumn(
            "Meaning", help=f"Verdict from semantic similarity. {vs_orig}"),
        "chrF": st.column_config.NumberColumn(
            "chrF", help=f"{vs_orig} Character n-gram overlap, 0–100."),
        "TER": st.column_config.NumberColumn(
            "TER", help=f"{vs_orig} Edit rate — lower = closer, 0 = identical."),
        "COMET": st.column_config.NumberColumn(
            "COMET", help=f"Neural quality score, single triplet: source = "
                          f"“{src_col}”, translation = “{cand_col}”, "
                          f"reference = {ref_name}."),
        "Wording (GT)": st.column_config.TextColumn(
            "Wording (GT)",
            help=f"Human benchmark: wording-overlap judgment of “{gt_col}” vs "
                 f"“{src_col}” (from its sentence BLEU against the original)."),
        "Semantic GT (%)": st.column_config.NumberColumn(
            "Semantic GT (%)",
            help=f"Human benchmark: “{gt_col}” vs “{src_col}”."),
        "Meaning (GT)": st.column_config.TextColumn(
            "Meaning (GT)",
            help=f"Human benchmark: meaning verdict of “{gt_col}” vs "
                 f"“{src_col}” (from its semantic similarity)."),
        "chrF (GT)": st.column_config.NumberColumn(
            "chrF (GT)", help=f"Human benchmark: “{gt_col}” vs “{src_col}”."),
        "TER (GT)": st.column_config.NumberColumn(
            "TER (GT)", help=f"Human benchmark: “{gt_col}” vs “{src_col}”. "
                             "Lower = closer."),
    }
    st.dataframe(styled, use_container_width=True, hide_index=True, height=520,
                 column_config=col_help)

    # ---- download
    buf = io.BytesIO()
    bleu_target = "ground truth" if gt_col else "original"
    summary_df = pd.DataFrame({
        "Metric": [f"Corpus BLEU (LLM vs {bleu_target})",
                   "Sentences scored", "Brevity penalty",
                   "1-gram precision", "2-gram precision", "3-gram precision",
                   "4-gram precision", "Mean sentence BLEU",
                   "Mean semantic similarity (LLM vs original)",
                   f"COMET system score (src=original, mt=LLM, "
                   f"ref={bleu_target})",
                   "Corpus chrF (LLM vs original)",
                   "Corpus TER (LLM vs original)",
                   "Mean semantic similarity (ground truth vs original)",
                   "Corpus chrF (ground truth vs original)",
                   "Corpus TER (ground truth vs original)"],
        "Value": [round(s["bleu"], 2), len(srcs), round(s["bp"], 3),
                  *[round(p, 1) for p in s["precisions"]],
                  round(s["sent_bleu_mean"], 2), round(s["sem_mean"], 3),
                  round(s["comet"], 3) if s["comet"] is not None else "n/a",
                  round(s["chrf"], 2), round(s["ter"], 2),
                  round(s["gt_sem_mean"], 3) if s["gt_sem_mean"] is not None else "n/a",
                  round(s["gt_chrf"], 2) if s["gt_chrf"] is not None else "n/a",
                  round(s["gt_ter"], 2) if s["gt_ter"] is not None else "n/a"],
    })
    with pd.ExcelWriter(buf) as xl:
        table.to_excel(xl, sheet_name="Per-sentence scores", index=False)
        summary_df.to_excel(xl, sheet_name="Summary", index=False)
    st.download_button("⬇️ Download full results (.xlsx)", buf.getvalue(),
                       file_name="translation_scores.xlsx",
                       mime="application/vnd.openxmlformats-officedocument"
                            ".spreadsheetml.sheet")

    st.info("**Reading the scores:** BLEU, chrF and TER measure *surface* overlap — "
            "they penalize reworded text even when the rewording is a perfect "
            "simplification (TER: lower = closer, 0 = identical). Semantic "
            "similarity and COMET look past wording toward *meaning*: "
            "“myocardial infarction” → “heart attack” scores low on BLEU but high "
            "on both. The sweet spot for LLM: **high semantic/COMET + "
            "low/moderate BLEU** = meaning preserved, wording simplified. Rows "
            "flagged *Possible meaning change* deserve a manual read.")

# ---- legend & references (always visible)
st.divider()
st.subheader("Score legend & references")
st.markdown("""
| Score | Range | Columns compared | What it measures | Code | Publication |
|---|---|---|---|---|---|
| **BLEU** | 0–100, higher = more similar wording | Single score: LLM output **vs** ground truth (or the original script if no ground truth is selected) | Overlap of word sequences (1–4-gram precision) with the reference, plus a brevity penalty. Standard MT metric, per the [Microsoft Translator methodology](https://learn.microsoft.com/azure/ai-services/translator/custom-translator/concepts/bleu-score). | [mjpost/sacrebleu](https://github.com/mjpost/sacrebleu); methodology: [MicrosoftDocs/azure-ai-docs](https://github.com/MicrosoftDocs/azure-ai-docs/blob/main/articles/ai-services/translator/custom-translator/concepts/bleu-score.md) | [Papineni et al. (2002)](https://aclanthology.org/P02-1040/), ACL; implementation: [Post (2018)](https://aclanthology.org/W18-6319/), WMT |
| **chrF** | 0–100, higher = more similar wording | LLM output **vs** original; also ground truth **vs** original (benchmark) | Character n-gram F-score — like BLEU but at character level; more forgiving of small word changes and morphology. | [m-popovic/chrF](https://github.com/m-popovic/chrF) (computed via sacrebleu) | [Popović (2015)](https://aclanthology.org/W15-3049/), WMT |
| **TER** | 0–100+, **lower** = closer (0 = identical) | LLM output **vs** original; also ground truth **vs** original (benchmark) | Translation Edit Rate: edits (insert/delete/substitute/shift) needed to turn the translation into the original. | [mjpost/sacrebleu](https://github.com/mjpost/sacrebleu) | [Snover et al. (2006)](https://aclanthology.org/2006.amta-papers.25/), AMTA |
| **Semantic similarity** | 0–100%, higher = same meaning | LLM output **vs** original; also ground truth **vs** original (benchmark) | Cosine similarity of sentence embeddings (paraphrase-multilingual-MiniLM-L12-v2); measures whether *meaning* is preserved regardless of wording. Drives the meaning verdicts (≥75% preserved, 55–75% review, <55% possible change). | [fivehills/TextSim_MTQE](https://github.com/fivehills/TextSim_MTQE) / [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) | [Reimers & Gurevych (2019)](https://aclanthology.org/D19-1410/), EMNLP |
| **COMET** | 0–100, higher = better quality | Single score, full triplet: source = original script, translation = LLM output, reference = ground truth (or original if none) | Neural metric (wmt22-comet-da) trained on human quality judgments of translations; sensitive to meaning errors rather than wording changes. | [Unbabel/COMET](https://github.com/Unbabel/COMET) | [Rei et al. (2020)](https://aclanthology.org/2020.emnlp-main.213/), EMNLP; model: [Rei et al. (2022)](https://aclanthology.org/2022.wmt-1.52/), WMT |
""")
st.caption(
    "**Implementation signatures** (for exact reproducibility): "
    "BLEU `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp` · "
    "chrF `nrefs:1|case:mixed|eff:yes|nc:6|nw:0` (chrF2, β=2, character-only, "
    "verified against Popović's reference script `chrF++.py -nw 0 -b 2`) · "
    "TER `nrefs:1|case:lc|tok:tercom|norm:no|punct:yes` · "
    "semantic similarity: `sentence-transformers` "
    "paraphrase-multilingual-MiniLM-L12-v2, cosine similarity · "
    "COMET: `Unbabel/wmt22-comet-da` via `comet.load_from_checkpoint(...)"
    ".predict(batch_size=8, gpus=0)`. Corpus scores are computed with "
    "sacrebleu's corpus methods (not averaged sentence scores); per-sentence "
    "BLEU uses sacrebleu's default exponential smoothing.")
