"""
MedLingo Translation Scorer — Streamlit app.

Upload a spreadsheet with the original script and the MedLingo output;
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

st.set_page_config(page_title="MedLingo Translation Scorer",
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
    """srcs = original scripts; cands = MedLingo output; gts = ground-truth
    human translations (optional).

    BLEU/chrF/TER/semantic are computed against the ORIGINAL — for MedLingo,
    and (when provided) also for the ground truth, so machine and human can
    be compared on the same basis. COMET is a single score using its proper
    triplet: src = original, mt = MedLingo, ref = ground truth (or original
    when no ground truth is given)."""
    srcs, cands, gts = list(srcs), list(cands), list(gts)
    has_gt = len(gts) > 0

    corpus = sacrebleu.corpus_bleu(cands, [srcs])
    corpus_chrf = sacrebleu.corpus_chrf(cands, [srcs])
    corpus_ter = sacrebleu.corpus_ter(cands, [srcs])
    if has_gt:
        gt_bleu = sacrebleu.corpus_bleu(gts, [srcs])
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
        s = sacrebleu.sentence_bleu(c, [s_], smooth_method="exp").score
        sim = float(cosines[i])
        row = {"#": i + 1, "Original script": s_}
        if has_gt:
            row["Ground truth"] = gts[i]
        row.update({"MedLingo output": c,
                    "BLEU": round(s, 1), "Wording": interpret(s),
                    "Semantic (%)": round(sim * 100),
                    "Meaning": meaning_verdict(sim),
                    "chrF": round(sacrebleu.sentence_chrf(c, [s_]).score, 1),
                    "TER": round(sacrebleu.sentence_ter(c, [s_]).score, 1)})
        if has_gt:
            g = gts[i]
            row["BLEU (GT)"] = round(
                sacrebleu.sentence_bleu(g, [s_], smooth_method="exp").score, 1)
            row["Semantic GT (%)"] = round(float(gt_cosines[i]) * 100)
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
               "gt_bleu": gt_bleu.score if has_gt else None,
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
    src = find(["original", "script", "source", "doctor", "english"])
    cand = find(["medlingo", "output", "candidate"], exclude=(src,))
    gt = find(["ground", "truth", "gold", "human", "reference"],
              exclude=(src, cand))
    if src is None or cand is None or src == cand:
        src, cand = cols[0], cols[1]
    return src, cand, gt


# ---------------------------------------------------------------- UI

st.title("🩺 MedLingo Translation Scorer")
st.caption("Upload a spreadsheet with the original script, the MedLingo output, "
           "and — optionally — a ground-truth human reference. You get overall "
           "translation scores (BLEU, chrF, TER, semantic similarity, COMET) and "
           "a score for every sentence. With a ground truth, MedLingo is scored "
           "against the human reference; without one, it is scored against the "
           "original script.")

uploaded = st.file_uploader(
    "Excel or CSV with two columns (original script, MedLingo output) "
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
                 "(original script and MedLingo output).")
        st.stop()

    cols = list(df.columns)
    src_default, cand_default, gt_default = autodetect(cols)
    c1, c2, c3 = st.columns(3)
    src_col = c1.selectbox("Original script (source) column", cols,
                           index=cols.index(src_default))
    cand_col = c2.selectbox("MedLingo output column", cols,
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
        st.caption("**3-column mode:** BLEU, chrF, TER and semantic similarity "
                   "are computed **against the original** for both translations "
                   "— MedLingo vs original and ground truth vs original — so "
                   "machine and human can be compared on the same basis. COMET "
                   "is a single score using its full triplet (source = original, "
                   "translation = MedLingo, reference = ground truth).")
    else:
        st.caption("**2-column mode:** no ground truth selected — all scores "
                   "compare MedLingo against the original script (for COMET, the "
                   "original serves as both source and reference).")

    with st.spinner(f"Scoring {len(srcs)} sentences…"):
        table, s = score_pairs(tuple(srcs), tuple(cands), tuple(gts), use_comet)

    # ---- headline scores
    comet_ref = f"“{gt_col}” (ground truth)" if gt_col else f"“{src_col}” (original)"
    vs = f"Compares “{cand_col}” vs “{src_col}” (original)."
    labels = ["Overall BLEU (corpus)", "Mean semantic similarity",
              "chrF (corpus)", "TER (corpus, lower = closer)"]
    values = [f"{s['bleu']:.1f}", f"{s['sem_mean'] * 100:.0f}%",
              f"{s['chrf']:.1f}", f"{s['ter']:.1f}"]
    helps = [f"{s['bleu_label']}. {vs} Word-sequence overlap.",
             f"{vs} Meaning similarity from sentence embeddings.",
             f"{vs} Character n-gram overlap.",
             f"{vs} Edits needed to match the original — lower is closer."]
    if s["comet"] is not None:
        labels.insert(2, "COMET")
        values.insert(2, f"{s['comet'] * 100:.0f}")
        helps.insert(2, f"Single score, full triplet: source = “{src_col}”, "
                        f"translation = “{cand_col}”, reference = {comet_ref}. "
                        "0–100, higher = better quality.")
    if gt_col:
        st.markdown(f"**MedLingo vs original** — “{cand_col}” scored against "
                    f"“{src_col}”")
    for col, lab, val, hlp in zip(st.columns(len(labels)), labels, values, helps):
        col.metric(lab, val, help=hlp)

    if gt_col:
        st.markdown(f"**Human benchmark: ground truth vs original** — "
                    f"“{gt_col}” scored against “{src_col}” with the same "
                    "metrics, for a machine-vs-human comparison")
        gvs = f"Compares “{gt_col}” vs “{src_col}” (original)."
        glabels = ["BLEU (corpus)", "Mean semantic similarity",
                   "chrF (corpus)", "TER (corpus, lower = closer)"]
        gvalues = [f"{s['gt_bleu']:.1f}", f"{s['gt_sem_mean'] * 100:.0f}%",
                   f"{s['gt_chrf']:.1f}", f"{s['gt_ter']:.1f}"]
        ghelps = [f"{gvs} Word-sequence overlap.",
                  f"{gvs} Meaning similarity from sentence embeddings.",
                  f"{gvs} Character n-gram overlap.",
                  f"{gvs} Edits needed to match the original — lower is closer."]
        gcols = st.columns(len(labels))
        if s["comet"] is not None:
            gcols = [c for i, c in enumerate(gcols) if i != 2]  # skip COMET slot
        for col, lab, val, hlp in zip(gcols, glabels, gvalues, ghelps):
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
    styled = view.style.map(
        lambda v: WORDING_COLORS.get(v, "") + chip, subset=["Wording"]
    ).map(
        lambda v: MEANING_COLORS.get(v, "") + chip, subset=["Meaning"]
    ).format({k: v for k, v in {"BLEU": "{:.1f}", "chrF": "{:.1f}",
                                "TER": "{:.1f}", "Semantic (%)": "{:.0f}",
                                "BLEU (GT)": "{:.1f}", "chrF (GT)": "{:.1f}",
                                "TER (GT)": "{:.1f}",
                                "Semantic GT (%)": "{:.0f}",
                                "COMET": "{:.0f}"}.items()
              if k in view.columns}, na_rep="")
    col_help = {
        "BLEU": st.column_config.NumberColumn(
            "BLEU", help=f"{vs} Word-sequence overlap, 0–100."),
        "Wording": st.column_config.TextColumn(
            "Wording", help=f"Judgment of the BLEU score. {vs}"),
        "Semantic (%)": st.column_config.NumberColumn(
            "Semantic (%)", help=f"{vs} Meaning similarity, 0–100%."),
        "Meaning": st.column_config.TextColumn(
            "Meaning", help=f"Verdict from semantic similarity. {vs}"),
        "chrF": st.column_config.NumberColumn(
            "chrF", help=f"{vs} Character n-gram overlap, 0–100."),
        "TER": st.column_config.NumberColumn(
            "TER", help=f"{vs} Edit rate — lower = closer, 0 = identical."),
        "COMET": st.column_config.NumberColumn(
            "COMET", help=f"Neural quality score, single triplet: source = "
                          f"“{src_col}”, translation = “{cand_col}”, "
                          f"reference = {comet_ref}."),
        "BLEU (GT)": st.column_config.NumberColumn(
            "BLEU (GT)", help=f"Human benchmark: “{gt_col}” vs “{src_col}”."),
        "Semantic GT (%)": st.column_config.NumberColumn(
            "Semantic GT (%)",
            help=f"Human benchmark: “{gt_col}” vs “{src_col}”."),
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
    summary_df = pd.DataFrame({
        "Metric": ["Corpus BLEU (MedLingo vs original)",
                   "Sentences scored", "Brevity penalty",
                   "1-gram precision", "2-gram precision", "3-gram precision",
                   "4-gram precision", "Mean sentence BLEU",
                   "Mean semantic similarity (MedLingo vs original)",
                   "COMET system score",
                   "Corpus chrF (MedLingo vs original)",
                   "Corpus TER (MedLingo vs original)",
                   "Corpus BLEU (ground truth vs original)",
                   "Mean semantic similarity (ground truth vs original)",
                   "Corpus chrF (ground truth vs original)",
                   "Corpus TER (ground truth vs original)"],
        "Value": [round(s["bleu"], 2), len(srcs), round(s["bp"], 3),
                  *[round(p, 1) for p in s["precisions"]],
                  round(s["sent_bleu_mean"], 2), round(s["sem_mean"], 3),
                  round(s["comet"], 3) if s["comet"] is not None else "n/a",
                  round(s["chrf"], 2), round(s["ter"], 2),
                  round(s["gt_bleu"], 2) if s["gt_bleu"] is not None else "n/a",
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
            "on both. The sweet spot for MedLingo: **high semantic/COMET + "
            "low/moderate BLEU** = meaning preserved, wording simplified. Rows "
            "flagged *Possible meaning change* deserve a manual read.")

# ---- legend & references (always visible)
st.divider()
st.subheader("Score legend & references")
st.markdown("""
| Score | Range | Columns compared | What it measures | Code | Publication |
|---|---|---|---|---|---|
| **BLEU** | 0–100, higher = more similar wording | MedLingo output **vs** original script; in 3-column mode also ground truth **vs** original (human benchmark, "GT" columns) | Overlap of word sequences (1–4-gram precision) with the original, plus a brevity penalty. Standard MT metric, per the [Microsoft Translator methodology](https://learn.microsoft.com/azure/ai-services/translator/custom-translator/concepts/bleu-score). | [mjpost/sacrebleu](https://github.com/mjpost/sacrebleu); methodology: [MicrosoftDocs/azure-ai-docs](https://github.com/MicrosoftDocs/azure-ai-docs/blob/main/articles/ai-services/translator/custom-translator/concepts/bleu-score.md) | [Papineni et al. (2002)](https://aclanthology.org/P02-1040/), ACL; implementation: [Post (2018)](https://aclanthology.org/W18-6319/), WMT |
| **chrF** | 0–100, higher = more similar wording | MedLingo output **vs** original; also ground truth **vs** original (benchmark) | Character n-gram F-score — like BLEU but at character level; more forgiving of small word changes and morphology. | [m-popovic/chrF](https://github.com/m-popovic/chrF) (computed via sacrebleu) | [Popović (2015)](https://aclanthology.org/W15-3049/), WMT |
| **TER** | 0–100+, **lower** = closer (0 = identical) | MedLingo output **vs** original; also ground truth **vs** original (benchmark) | Translation Edit Rate: edits (insert/delete/substitute/shift) needed to turn the translation into the original. | [mjpost/sacrebleu](https://github.com/mjpost/sacrebleu) | [Snover et al. (2006)](https://aclanthology.org/2006.amta-papers.25/), AMTA |
| **Semantic similarity** | 0–100%, higher = same meaning | MedLingo output **vs** original; also ground truth **vs** original (benchmark) | Cosine similarity of sentence embeddings (paraphrase-multilingual-MiniLM-L12-v2); measures whether *meaning* is preserved regardless of wording. Drives the meaning verdicts (≥75% preserved, 55–75% review, <55% possible change). | [fivehills/TextSim_MTQE](https://github.com/fivehills/TextSim_MTQE) / [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) | [Reimers & Gurevych (2019)](https://aclanthology.org/D19-1410/), EMNLP |
| **COMET** | 0–100, higher = better quality | Single score, full triplet: source = original script, translation = MedLingo output, reference = ground truth (or original if none) | Neural metric (wmt22-comet-da) trained on human quality judgments of translations; sensitive to meaning errors rather than wording changes. | [Unbabel/COMET](https://github.com/Unbabel/COMET) | [Rei et al. (2020)](https://aclanthology.org/2020.emnlp-main.213/), EMNLP; model: [Rei et al. (2022)](https://aclanthology.org/2022.wmt-1.52/), WMT |
""")
