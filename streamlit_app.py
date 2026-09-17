"""
Interpreter Translation Scorer — Streamlit app.

Upload a spreadsheet with the original script and the Interpreter output;
get corpus-level and per-sentence translation scores:
BLEU, chrF++, TER (sacrebleu), semantic similarity (sentence embeddings,
per TextSim_MTQE) and COMET (Unbabel wmt22-comet-da).

Run locally:  streamlit run streamlit_app.py
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu
import streamlit as st

st.set_page_config(page_title="Interpreter Translation Scorer",
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


@st.cache_resource(show_spinner="Loading BERTScore model…")
def bert_scorer():
    from bert_score import BERTScorer
    # multilingual model so English and Spanish rows both score correctly
    return BERTScorer(model_type="bert-base-multilingual-cased")


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
    """srcs = original scripts; cands = Interpreter output; gts = ground-truth
    human translations (optional).

    BLEU, chrF++ and TER are single reference-based scores comparing the
    Interpreter output against the ground truth (or the original when no ground
    truth is given); COMET is a single score using its triplet (source =
    original, translation = Interpreter, reference = ground truth). Semantic
    similarity uses multilingual embeddings and is computed against the
    original for both Interpreter and ground truth (meaning-preservation check,
    valid cross-lingually)."""
    srcs, cands, gts = list(srcs), list(cands), list(gts)
    has_gt = len(gts) > 0
    bleu_refs = gts if has_gt else srcs

    corpus = sacrebleu.corpus_bleu(cands, [bleu_refs])
    # chrF++ (Popović 2017): character 6-grams + word 1/2-grams, beta=2
    corpus_chrf = sacrebleu.corpus_chrf(cands, [bleu_refs], word_order=2)
    corpus_ter = sacrebleu.corpus_ter(cands, [bleu_refs])

    # BERTScore (Zhang et al. 2020): token-level F1 with contextual
    # embeddings, Interpreter output vs the same reference as BLEU/TER
    _, _, bs_f1 = bert_scorer().score(cands, bleu_refs)
    bert_f1 = [float(f) * 100 for f in bs_f1]

    model = embedder()

    def encode_norm(texts):
        e = model.encode(texts, batch_size=64, show_progress_bar=False)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    src_emb = encode_norm(srcs)
    cand_emb = encode_norm(cands)
    cosines = np.sum(src_emb * cand_emb, axis=1).clip(-1, 1)
    if has_gt:
        gt_emb = encode_norm(gts)
        gt_cosines = np.sum(src_emb * gt_emb, axis=1).clip(-1, 1)
        # direct comparison: interpreter output vs ground truth (same language)
        ig_cosines = np.sum(cand_emb * gt_emb, axis=1).clip(-1, 1)

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
        row.update({"Interpreter output": c,
                    "BLEU": round(s, 1), "Wording": interpret(s),
                    "Semantic Int↔Orig (%)": round(sim * 100),
                    "Meaning Int↔Orig": meaning_verdict(sim),
                    "chrF++": round(sacrebleu.sentence_chrf(
                        c, [bleu_refs[i]], word_order=2).score, 1),
                    "TER": round(
                        sacrebleu.sentence_ter(c, [bleu_refs[i]]).score, 1),
                    "BERTScore": round(bert_f1[i], 1)})
        if has_gt:
            row["Semantic Int↔GT (%)"] = round(float(ig_cosines[i]) * 100)
            row["Meaning Int↔GT"] = meaning_verdict(float(ig_cosines[i]))
            row["Semantic GT↔Orig (%)"] = round(float(gt_cosines[i]) * 100)
            row["Meaning GT↔Orig"] = meaning_verdict(float(gt_cosines[i]))
        if comet_scores is not None:
            row["COMET"] = round(comet_scores[i] * 100)
        rows.append(row)

    summary = {"bleu": corpus.score, "bleu_label": interpret(corpus.score),
               "bp": corpus.bp, "precisions": list(corpus.precisions),
               "chrf": corpus_chrf.score, "ter": corpus_ter.score,
               "sem_mean": float(np.mean(cosines)),
               "sent_bleu_mean": float(np.mean([r["BLEU"] for r in rows])),
               "comet": comet_system,
               "bertscore": float(np.mean(bert_f1)),
               "gt_sem_mean": float(np.mean(gt_cosines)) if has_gt else None,
               "ig_sem_mean": float(np.mean(ig_cosines)) if has_gt else None}
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
    cand = find(["interpret", "llm", "medlingo", "output", "candidate",
                 "generated", "gpt", "ai"], exclude=(src,))
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


def results_xlsx(table, summary_df) -> bytes:
    """Results workbook with the same color coding as the on-screen table."""
    from openpyxl.styles import PatternFill, Font
    from openpyxl.utils import get_column_letter
    WORD_X = {
        "Almost no overlap": ("FFEBE9", "CF222E"),
        "Low overlap": ("FFEBE9", "CF222E"),
        "Gist preserved, heavily reworded": ("FFF1E5", "BC4C00"),
        "Moderate overlap": ("FFF8C5", "7D4E00"),
        "High overlap": ("DDF4FF", "0969DA"),
        "Very high overlap": ("DAFBE1", "1A7F37"),
        "Near-identical": ("DAFBE1", "1A7F37"),
    }
    MEAN_X = {
        "Possible meaning change": ("FFEBE9", "CF222E"),
        "Mostly preserved — review": ("FFF8C5", "7D4E00"),
        "Meaning preserved": ("DAFBE1", "1A7F37"),
    }
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        table.to_excel(xl, sheet_name="Per-sentence scores", index=False)
        summary_df.to_excel(xl, sheet_name="Summary", index=False)
        ws = xl.book["Per-sentence scores"]
        cols = list(table.columns)
        chip_maps = {}
        for name, mapping in [("Wording", WORD_X),
                              ("Meaning Int↔Orig", MEAN_X),
                              ("Meaning Int↔GT", MEAN_X),
                              ("Meaning GT↔Orig", MEAN_X)]:
            if name in cols:
                chip_maps[cols.index(name) + 1] = mapping
        rev_idx = (cols.index("Needs review") + 1
                   if "Needs review" in cols else None)
        for r in range(2, len(table) + 2):
            for cidx, mapping in chip_maps.items():
                v = ws.cell(row=r, column=cidx).value
                if v in mapping:
                    bg, fg = mapping[v]
                    cell = ws.cell(row=r, column=cidx)
                    cell.fill = PatternFill("solid", fgColor=bg)
                    cell.font = Font(color=fg, bold=True)
            if rev_idx:
                v = str(ws.cell(row=r, column=rev_idx).value or "")
                pair = (("F3E8FF", "6639BA") if v.startswith("Rule")
                        else ("EAEEF2", "57606A") if v == "Control" else None)
                if pair:
                    cell = ws.cell(row=r, column=rev_idx)
                    cell.fill = PatternFill("solid", fgColor=pair[0])
                    cell.font = Font(color=pair[1], bold=True)
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for name in ("Original script", "Ground truth", "Interpreter output"):
            if name in cols:
                ws.column_dimensions[
                    get_column_letter(cols.index(name) + 1)].width = 45
        xl.book["Summary"].column_dimensions["A"].width = 55
    return buf.getvalue()


FLORES_TYPES = ["Omission", "Addition", "Substitution",
                "Editorialization", "False fluency"]


def reviewer_xlsx(table) -> bytes:
    """Reviewer worksheet: the ENTIRE conversation in its natural order
    (for clinical context), no scores and no reference translation.
    Utterances selected by the review rules are highlighted in yellow —
    the reviewer codes only those rows (one Yes/No column per Flores
    error type, plus clinical significance)."""
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.utils import get_column_letter
    from openpyxl.styles import Font, PatternFill, Alignment

    cols = {"#": table["#"].values}
    if "Direction" in table.columns:
        cols["Direction"] = table["Direction"].values
    cols["Original"] = table["Original script"].values
    cols["Interpretation"] = table["Interpreter output"].values
    cols["REVIEW?"] = ["YES" if f else "" for f in table["Needs review"]]
    for t in FLORES_TYPES:
        cols[t] = ""
    cols["Clinical significance"] = ""
    cols["Notes"] = ""
    review = pd.DataFrame(cols)
    keydf = pd.DataFrame({
        "Row #": table["#"].values,
        "Selected by": table["Needs review"].values,
    })
    keydf = keydf[keydf["Selected by"] != ""]
    instructions = pd.DataFrame({"Instructions for the reviewer": [
        "The sheet contains the ENTIRE conversation, in order, so every "
        "utterance can be judged in its clinical context.",
        "Code ONLY the YELLOW-highlighted rows (REVIEW? = YES): for each, "
        "set Yes/No in the five error-type columns and rate the clinical "
        "significance.",
        "Judge the Interpretation against the Original utterance (what "
        "was actually said), using the surrounding conversation for "
        "context.",
        "Error types (Flores et al., 2003): Omission, Addition, "
        "Substitution, Editorialization, False fluency. An utterance can "
        "have several.",
        "Use Notes for anything relevant, including errors you notice in "
        "NON-highlighted rows.",
    ]})

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        review.to_excel(xl, sheet_name="Review", index=False)
        instructions.to_excel(xl, sheet_name="READ ME", index=False)
        keydf.to_excel(xl, sheet_name="KEY - REMOVE BEFORE SENDING",
                       index=False)
        ws = xl.book["Review"]
        ncols = len(review.columns)
        has_dir = "Direction" in review.columns
        widths = ([6] + ([16] if has_dir else []) + [55, 55, 9]
                  + [14] * len(FLORES_TYPES) + [26, 30])
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w
        header_fill = PatternFill("solid", fgColor="0969DA")
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = header_fill
            cell.alignment = Alignment(vertical="center", wrap_text=True)
        # highlight the full row of every selected utterance in yellow
        yellow = PatternFill("solid", fgColor="FFF200")
        flagged = [i for i, f in enumerate(table["Needs review"]) if f]
        for i in flagged:
            for c in range(1, ncols + 1):
                ws.cell(row=i + 2, column=c).fill = yellow
        for row in ws.iter_rows(min_row=2):
            for cell in row[1 + int(has_dir):3 + int(has_dir)]:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        n = len(review) + 1
        first = 5 + int(has_dir)  # first Flores column
        dv_yn = DataValidation(type="list", allow_blank=True,
                               formula1='"Yes,No"')
        ws.add_data_validation(dv_yn)
        for j in range(len(FLORES_TYPES)):
            col = get_column_letter(first + j)
            dv_yn.add(f"{col}2:{col}{n}")
        dv_sig = DataValidation(
            type="list", allow_blank=True,
            formula1='"No potential consequence,Potential consequence"')
        ws.add_data_validation(dv_sig)
        sig_col = get_column_letter(first + len(FLORES_TYPES))
        dv_sig.add(f"{sig_col}2:{sig_col}{n}")
        ws.freeze_panes = "A2"
        xl.book["READ ME"].column_dimensions["A"].width = 110
    return buf.getvalue()


def render_results(srcs, cands, gts, dirs, key,
                   src_col, cand_col, gt_col, use_comet):
    """Score one subset of rows and render the full results block."""
    with st.spinner(f"Scoring {len(srcs)} sentences…"):
        table, s = score_pairs(tuple(srcs), tuple(cands), tuple(gts), use_comet)
    table = table.copy()
    if dirs:
        table.insert(1, "Direction", list(dirs))

    # ---- review-selection rules (pre-specified, reproducible)
    # A sentence is selected if ANY rule fires; plus a seeded 10% random
    # control sample of unflagged rows to estimate the false-negative rate.
    if gts:
        import random
        flags = []
        for i in range(len(table)):
            fired = []
            meaning = table["Meaning Int↔Orig"].iloc[i]
            if meaning == "Possible meaning change":
                fired.append("1")
            elif meaning == "Mostly preserved — review":
                fired.append("2")
            if table["Semantic Int↔GT (%)"].iloc[i] < 75:
                fired.append("3")
            if ("COMET" in table.columns
                    and table["COMET"].iloc[i] < 70
                    and meaning == "Meaning preserved"):
                fired.append("4")
            n_gt = max(len(str(gts[i]).split()), 1)
            if len(str(cands[i]).split()) < 0.6 * n_gt:
                fired.append("5")
            flags.append("Rule " + "+".join(fired) if fired else "")
        rng = random.Random(42)  # fixed seed -> reproducible control sample
        unflagged = [i for i, f in enumerate(flags) if not f]
        n_ctrl = round(0.10 * len(unflagged))
        for i in (rng.sample(unflagged, n_ctrl) if n_ctrl else []):
            flags[i] = "Control"
        table["Needs review"] = flags

    # ---- headline scores
    ref_name = f"“{gt_col}” (ground truth)" if gt_col else f"“{src_col}” (original)"
    vs_orig = f"Compares “{cand_col}” vs “{src_col}” (original)."

    if gt_col:
        # Group 1 — single reference-based scores (Interpreter vs ground truth)
        st.markdown(f"#### 1️⃣ Single scores — “{cand_col}” vs “{gt_col}”")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Overall BLEU (corpus)", f"{s['bleu']:.1f}",
                  help=f"{s['bleu_label']}. COMPARES: “{cand_col}” vs "
                       f"“{gt_col}” (ground truth). WHY: measures how close "
                       "the interpreter's wording is to the certified "
                       "reference (word-sequence overlap, 0–100). EXAMPLE: "
                       "interpreter “Tome dos tabletas de metformina…” vs "
                       "ground truth “Tome dos pastillas de metformina…” — "
                       "almost every word sequence matches except "
                       "tabletas/pastillas, so BLEU is high but not 100.")
        c2.metric("chrF++ (corpus)", f"{s['chrf']:.1f}",
                  help=f"COMPARES: “{cand_col}” vs “{gt_col}” (ground truth). "
                       "WHY: like BLEU but at character level (+ word "
                       "1–2-grams), so near-misses get partial credit "
                       "(chrF++, Popović 2017). EXAMPLE: “intervención "
                       "inmediata” vs “tratamiento inmediato” — BLEU sees no "
                       "matching words, but chrF++ credits the shared "
                       "characters of inmediata/inmediato.")
        c3.metric("TER (corpus, lower = closer)", f"{s['ter']:.1f}",
                  help=f"COMPARES: “{cand_col}” vs “{gt_col}” (ground truth). "
                       "WHY: counts the edits (insert/delete/substitute/"
                       "shift) needed to turn the interpreter's sentence "
                       "into the reference — LOWER is closer, 0 = identical. "
                       "EXAMPLE: “I feel dizzy when I stand up fast” → "
                       "“I get dizzy when I stand up quickly” takes 2 "
                       "substitutions out of 9 words ≈ TER 22.")
        c4.metric("BERTScore F1", f"{s['bertscore']:.1f}",
                  help=f"COMPARES: “{cand_col}” vs “{gt_col}” (ground truth). "
                       "WHY: matches words by MEANING, not spelling, using "
                       "contextual embeddings (Zhang et al. 2020) — so a "
                       "correct synonym isn't punished. EXAMPLE: interpreter "
                       "“heart attack” vs ground truth “myocardial "
                       "infarction” — zero word overlap (low BLEU), but "
                       "BERTScore stays high because the tokens mean the "
                       "same thing.")
        if s["comet"] is not None:
            c5.metric("COMET", f"{s['comet'] * 100:.0f}",
                      help=f"COMPARES all three columns at once: source = "
                           f"“{src_col}”, translation = “{cand_col}”, "
                           f"reference = “{gt_col}”. WHY: a neural model "
                           "trained on human quality ratings of translations "
                           "— it sees the original too, so it judges overall "
                           "translation quality, and collapses on meaning "
                           "errors even when the sentence is fluent. "
                           "EXAMPLE: rendering “Soy alérgico a la "
                           "penicilina” as the fluent-but-wrong “I have had "
                           "high blood pressure” drops COMET from ~90 to "
                           "~60.")

        # Groups 2 & 3 — semantic meaning-preservation vs the original
        # (multilingual embeddings, valid across languages)
        st.markdown(f"#### 2️⃣ Semantic similarity (meaning)")
        m0, m1, m2, _ = st.columns(4)
        m0.metric("Interpreter ↔ ground truth", f"{s['ig_sem_mean'] * 100:.0f}%",
                  help=f"COMPARES: “{cand_col}” vs “{gt_col}” directly (both in "
                       "the same language). WHY: the semantic counterpart of "
                       "the group-1 scores — how close in MEANING the "
                       "interpreter's rendition is to the certified reference, "
                       "regardless of wording. EXAMPLE: “heart attack” vs "
                       "“myocardial infarction” → high; unrelated content → "
                       "low. Cosine of multilingual sentence embeddings, "
                       "averaged over all sentences.")
        m1.metric("Interpreter vs original", f"{s['sem_mean'] * 100:.0f}%",
                  help=f"COMPARES: “{cand_col}” vs “{src_col}” (the original "
                       "utterance). WHY: checks how much of the original's "
                       "MEANING survived into the interpretation, regardless "
                       "of wording or language (multilingual embeddings). "
                       "EXAMPLE: original “Soy alérgico a la penicilina” vs "
                       "interpreter “I am allergic to penicillin” ≈ 95% "
                       "(meaning kept across languages); vs “I have high "
                       "blood pressure” ≈ 30% (meaning lost). Shown value = "
                       "average over all sentences.")
        m2.metric("Ground truth vs original (human ceiling)",
                  f"{s['gt_sem_mean'] * 100:.0f}%",
                  help=f"COMPARES: “{gt_col}” (certified human translation) "
                       f"vs “{src_col}” (original) — the SAME measurement as "
                       "the card on the left, applied to the human "
                       "reference. WHY: calibration. Cross-language "
                       "similarity never reaches 100% even for a perfect "
                       "translation, so this shows what a professional "
                       "scores on the same sentences — the human ceiling. "
                       "EXAMPLE: interpreter 88% vs human 87% → the "
                       "interpreter preserves meaning at human level; "
                       "interpreter 74% vs human 87% → meaning is being "
                       "lost — check the red rows in the table.")
    else:
        labels = ["Overall BLEU (corpus)", "Mean semantic similarity",
                  "chrF++ (corpus)", "TER (corpus, lower = closer)",
                  "BERTScore F1"]
        values = [f"{s['bleu']:.1f}", f"{s['sem_mean'] * 100:.0f}%",
                  f"{s['chrf']:.1f}", f"{s['ter']:.1f}",
                  f"{s['bertscore']:.1f}"]
        helps = [f"{s['bleu_label']}. {vs_orig} Word-sequence overlap.",
                 f"{vs_orig} Meaning similarity from sentence embeddings.",
                 f"{vs_orig} Character + word n-gram overlap (chrF++).",
                 f"{vs_orig} Edits needed to match the original.",
                 f"{vs_orig} Token-level semantic F1 (Zhang et al. 2020)."]
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
                              if v in set(table["Meaning Int↔Orig"])]
    label_opts = ["All"] + [l for l in ["Almost no overlap", "Low overlap",
                                        "Gist preserved, heavily reworded",
                                        "Moderate overlap", "High overlap",
                                        "Very high overlap", "Near-identical"]
                            if l in set(table["Wording"])]
    pick_verdict = f1.radio("Filter by meaning", verdict_opts, horizontal=True, key=f"vm_{key}")
    pick_label = f2.radio("Filter by wording overlap (BLEU)", label_opts,
                          horizontal=True, key=f"wb_{key}")

    view = table
    if pick_verdict != "All":
        view = view[view["Meaning Int↔Orig"] == pick_verdict]
    if pick_label != "All":
        view = view[view["Wording"] == pick_label]
    if "Needs review" in table.columns:
        n_sel = int((table["Needs review"] != "").sum())
        if st.checkbox(f"Show only sentences selected for clinical review "
                       f"({n_sel} of {len(table)})", key=f"nr_{key}",
                       help="Selection is automatic and reproducible — a "
                            "sentence is selected if ANY rule fires. The "
                            "rules test only the interpreter's own output: "
                            "Rule 1 = Meaning Int↔Orig red (Semantic "
                            "Int↔Orig <55%). Rule 2 = Meaning Int↔Orig "
                            "yellow (55–75%). Rule 3 = Semantic Int↔GT "
                            "<75% (rendition far in meaning from the "
                            "certified reference). Rule 4 = COMET <70 "
                            "despite green Meaning Int↔Orig "
                            "(fluent-but-wrong screen). Rule 5 = "
                            "interpreter output <60% of the ground truth's "
                            "word count (omission screen). 'Control' = "
                            "random 10% of unflagged rows (fixed seed 42) "
                            "to estimate the screen's false-negative rate. "
                            "The 'Needs review' column shows which rule "
                            "fired for each row."):
            view = view[view["Needs review"] != ""]
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
    wording_cols = [c for c in ("Wording",) if c in view.columns]
    meaning_cols = [c for c in ("Meaning Int↔Orig", "Meaning Int↔GT",
                                "Meaning GT↔Orig") if c in view.columns]
    review_cols = [c for c in ("Needs review",) if c in view.columns]

    def review_style(v):
        if str(v).startswith("Rule"):
            return "background-color:#f3e8ff; color:#6639ba" + chip
        if v == "Control":
            return "background-color:#eaeef2; color:#57606a" + chip
        return ""

    styled = view.style.map(
        lambda v: WORDING_COLORS.get(v, "") + chip, subset=wording_cols
    ).map(
        lambda v: MEANING_COLORS.get(v, "") + chip, subset=meaning_cols
    ).map(review_style, subset=review_cols
    ).format({k: v for k, v in {"BLEU": "{:.1f}", "chrF++": "{:.1f}",
                                "TER": "{:.1f}", "BERTScore": "{:.1f}",
                                "Semantic Int↔Orig (%)": "{:.0f}",
                                "Semantic Int↔GT (%)": "{:.0f}",
                                "Semantic GT↔Orig (%)": "{:.0f}",
                                "COMET": "{:.0f}"}.items()
              if k in view.columns}, na_rep="")
    bleu_vs = f"Compares “{cand_col}” vs {ref_name}."
    col_help = {
        "#": st.column_config.NumberColumn(
            "#", help="Row number (order of the sentences in your file, after "
                      "removing rows with empty cells)."),
        "Original script": st.column_config.TextColumn(
            "Original script",
            help=f"Your column “{src_col}”: the original text, used as the "
                 "source for COMET and as the reference for the semantic "
                 "meaning-preservation check."),
        "Ground truth": st.column_config.TextColumn(
            "Ground truth",
            help=f"Your column “{gt_col}”: the human reference translation. "
                 "Serves as the reference for BLEU, chrF++, TER and COMET."),
        "Interpreter output": st.column_config.TextColumn(
            "Interpreter output",
            help=f"Your column “{cand_col}”: the Interpreter-generated translation "
                 "being evaluated. All non-GT scores in this table rate this "
                 "text."),
        "BLEU": st.column_config.NumberColumn(
            "BLEU",
            help=f"WHAT: word-sequence overlap, 0–100 (higher = more similar "
                 f"wording). {bleu_vs} HOW: sacrebleu sentence BLEU — "
                 "geometric mean of 1–4-gram precisions × brevity penalty, "
                 "13a tokenizer, exponential smoothing (Papineni 2002)."),
        "Wording": st.column_config.TextColumn(
            "Wording",
            help=f"WHAT: plain-English judgment of the BLEU score. {bleu_vs} "
                 "HOW: banded from BLEU — <10 almost no overlap, 10–20 low, "
                 "20–30 gist/reworded, 30–40 moderate, 40–50 high, 50–60 very "
                 "high, ≥60 near-identical."),
        "Semantic Int↔Orig (%)": st.column_config.NumberColumn(
            "Semantic Int↔Orig (%)",
            help=f"WHAT: meaning similarity, 0–100% (higher = same meaning, "
                 f"regardless of wording). {vs_orig} HOW: cosine similarity "
                 "between sentence embeddings from "
                 "paraphrase-multilingual-MiniLM-L12-v2 (Sentence-BERT, "
                 "Reimers & Gurevych 2019; method of TextSim_MTQE)."),
        "Meaning Int↔Orig": st.column_config.TextColumn(
            "Meaning Int↔Orig",
            help=f"WHAT: verdict on whether the meaning was preserved. "
                 f"{vs_orig} HOW: banded from semantic similarity — ≥75% "
                 "meaning preserved, 55–75% mostly preserved (review), "
                 "<55% possible meaning change."),
        "chrF++": st.column_config.NumberColumn(
            "chrF++",
            help=f"WHAT: character + word n-gram overlap, 0–100 (higher = "
                 f"more similar; forgiving of small word changes). {bleu_vs} "
                 "HOW: sacrebleu chrF++ — F-score over character 1–6-grams "
                 "and word 1–2-grams with β=2 (recall weighted double; "
                 "Popović 2017)."),
        "TER": st.column_config.NumberColumn(
            "TER",
            help=f"WHAT: Translation Edit Rate, 0–100+ (LOWER = closer, 0 = "
                 f"identical). {bleu_vs} HOW: sacrebleu TER — minimum "
                 "word-level edits (insert/delete/substitute/shift) to turn "
                 "the Interpreter output into the reference, divided by reference "
                 "length; tercom tokenization, case-insensitive (Snover 2006)."),
        "BERTScore": st.column_config.NumberColumn(
            "BERTScore",
            help=f"WHAT: token-level semantic F1, 0–100 (higher = closer in "
                 f"meaning; robust to rewording). {bleu_vs} HOW: official "
                 "bert-score package — greedy cosine matching of contextual "
                 "token embeddings (bert-base-multilingual-cased, layer 9), "
                 "F1 of precision/recall over tokens (Zhang et al., ICLR "
                 "2020)."),
        "COMET": st.column_config.NumberColumn(
            "COMET",
            help=f"WHAT: neural translation-quality estimate, 0–100 (higher "
                 f"= better), trained on human judgments. HOW: "
                 "Unbabel/wmt22-comet-da model scoring the triplet source = "
                 f"“{src_col}”, translation = “{cand_col}”, reference = "
                 f"{ref_name} (Rei et al. 2020/2022)."),
        "Semantic Int↔GT (%)": st.column_config.NumberColumn(
            "Semantic Int↔GT (%)",
            help=f"WHAT: meaning similarity of the interpreter's rendition "
                 f"to the certified reference, 0–100%. COMPARES: “{cand_col}” "
                 f"vs “{gt_col}” directly (same language). HOW: cosine of "
                 "multilingual sentence embeddings — the semantic "
                 "counterpart of BLEU/chrF++/TER."),
        "Meaning Int↔GT": st.column_config.TextColumn(
            "Meaning Int↔GT",
            help=f"WHAT: verdict on the Semantic Int↔GT (%) score — is the "
                 f"interpreter's rendition close in meaning to “{gt_col}”? "
                 "HOW: same ≥75% / 55–75% / <55% bands as the other Meaning "
                 "columns. Note: both texts are in the same language here, "
                 "so similarities run higher than the cross-lingual "
                 "comparisons — read borderline verdicts accordingly."),
        "Semantic GT↔Orig (%)": st.column_config.NumberColumn(
            "Semantic GT↔Orig (%)",
            help=f"WHAT: human benchmark — meaning similarity of “{gt_col}” "
                 f"vs “{src_col}”, 0–100%. HOW: same embedding cosine as the "
                 "other semantic columns. Compare with Semantic Int↔Orig (%) "
                 "to see whether the Interpreter preserves meaning as well "
                 "as the human."),
        "Meaning GT↔Orig": st.column_config.TextColumn(
            "Meaning GT↔Orig",
            help=f"WHAT: human benchmark — meaning verdict for “{gt_col}” vs "
                 f"“{src_col}”. HOW: same ≥75% / 55–75% / <55% bands as "
                 "Meaning Int↔Orig, applied to Semantic GT↔Orig (%)."),
    }
    if dirs:
        col_help["Direction"] = st.column_config.TextColumn(
            "Direction", help="Your direction/speaker label for this row "
                              "(e.g. Doctor En→Es / Patient Es→En).")
    if "Needs review" in view.columns:
        col_help["Needs review"] = st.column_config.TextColumn(
            "Needs review",
            help="Pre-specified selection for clinical review; a sentence "
                 "is selected if ANY rule fires. The rules test only the "
                 "interpreter's own output. Rule 1: Meaning Int↔Orig red "
                 "(Semantic Int↔Orig <55%). Rule 2: Meaning Int↔Orig "
                 "yellow (55–75%). Rule 3: Semantic Int↔GT <75% (rendition "
                 "far in meaning from the certified reference). Rule 4: "
                 "COMET <70 despite green Meaning Int↔Orig "
                 "(fluent-but-wrong screen). Rule 5: interpreter output "
                 "<60% of the ground truth's word count (omission screen). "
                 "'Control' = random 10% of unflagged rows (seed 42) to "
                 "estimate the screen's false-negative rate.")
    st.dataframe(styled, use_container_width=True, hide_index=True, height=520,
                 column_config=col_help)

    # ---- download
    bleu_target = "ground truth" if gt_col else "original"
    metric_labels = [f"Corpus BLEU (Interpreter vs {bleu_target})",
                     "Sentences scored", "Brevity penalty",
                     "1-gram precision", "2-gram precision",
                     "3-gram precision", "4-gram precision",
                     "Mean sentence BLEU",
                     "Mean semantic similarity (Interpreter vs original)",
                     f"COMET system score (src=original, mt=Interpreter, "
                     f"ref={bleu_target})",
                     f"Corpus TER (Interpreter vs {bleu_target})",
                     f"Corpus chrF++ (Interpreter vs {bleu_target})",
                     f"BERTScore F1 (Interpreter vs {bleu_target})",
                     "Mean semantic similarity (Interpreter vs ground truth)",
                     "Mean semantic similarity (ground truth vs original)"]

    def _summary_values(s_, n_):
        return [round(s_["bleu"], 2), n_, round(s_["bp"], 3),
                *[round(p, 1) for p in s_["precisions"]],
                round(s_["sent_bleu_mean"], 2), round(s_["sem_mean"], 3),
                round(s_["comet"], 3) if s_["comet"] is not None else "n/a",
                round(s_["ter"], 2), round(s_["chrf"], 2),
                round(s_["bertscore"], 2),
                round(s_["ig_sem_mean"], 3)
                if s_["ig_sem_mean"] is not None else "n/a",
                round(s_["gt_sem_mean"], 3)
                if s_["gt_sem_mean"] is not None else "n/a"]

    sdata = {"Metric": metric_labels}
    if dirs and len(set(dirs)) > 1:
        # one column per direction next to the overall values
        sdata[f"All ({len(srcs)})"] = _summary_values(s, len(srcs))
        for g in dict.fromkeys(dirs):
            idx = [i for i, d in enumerate(dirs) if d == g]
            _, sg = score_pairs(tuple(srcs[i] for i in idx),
                                tuple(cands[i] for i in idx),
                                tuple(gts[i] for i in idx) if gts else (),
                                use_comet)
            sdata[f"{g} ({len(idx)})"] = _summary_values(sg, len(idx))
    else:
        sdata["Value"] = _summary_values(s, len(srcs))
    summary_df = pd.DataFrame(sdata)
    st.download_button("⬇️ Download full results (.xlsx)",
                       results_xlsx(table, summary_df),
                       file_name="translation_scores.xlsx",
                       mime="application/vnd.openxmlformats-officedocument"
                            ".spreadsheetml.sheet", key=f"dl_{key}")

    # ---- blinded reviewer worksheet
    if "Needs review" in table.columns:
        if (table["Needs review"] != "").any():
            st.download_button(
                "🧑‍⚕️ Download reviewer worksheet (.xlsx)",
                reviewer_xlsx(table),
                file_name="reviewer_worksheet.xlsx",
                mime="application/vnd.openxmlformats-officedocument"
                     ".spreadsheetml.sheet", key=f"rev_{key}",
                help="The entire conversation in its natural order (for "
                     "clinical context), scores and reference translation "
                     "hidden. Utterances selected by the review rules are "
                     "highlighted in yellow — the reviewer codes only "
                     "those: one Yes/No column per Flores error type plus "
                     "clinical significance. Delete the KEY sheet before "
                     "sending; keep your copy for un-blinding.")

    st.info("**Reading the scores:** BLEU, chrF++ and TER measure *surface* "
            "overlap with the reference — they reward wording close to the "
            "ground truth (TER: lower = closer, 0 = identical). Semantic "
            "similarity and COMET look past wording toward *meaning*. A "
            "translation can phrase things differently from the ground truth "
            "(lower BLEU/chrF++) and still be excellent — check COMET and the "
            "semantic scores in that case. Rows flagged *Possible meaning "
            "change* deserve a manual read.")


@st.cache_data(show_spinner=False)
def template_xlsx() -> bytes:
    """The input template offered for download on the start screen."""
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    data = pd.DataFrame({
        "Encounter ID": ["EXAMPLE-01", "EXAMPLE-01", ""],
        "Turn": [1, 2, ""],
        "Direction": ["Doctor (En→Es)", "Patient (Es→En)", ""],
        "Original script": [
            "The patient presents with acute myocardial infarction and "
            "requires immediate intervention.",
            "Me duele el pecho desde esta mañana y me falta el aire.", ""],
        "Interpreter output": [
            "El paciente tiene un ataque cardiaco grave y necesita "
            "tratamiento de inmediato.",
            "My chest has been hurting since this morning and I am short "
            "of breath.", ""],
        "Ground truth": [
            "El paciente presenta un infarto agudo de miocardio y requiere "
            "intervención inmediata.",
            "My chest has been hurting since this morning and I feel short "
            "of breath.", ""],
    })
    instructions = pd.DataFrame({"Instructions": [
        "ONE file for the whole study - every turn, one row per utterance.",
        "",
        "Direction          = who is speaking; use two consistent labels, "
        "e.g. 'Doctor (En→Es)' and 'Patient (Es→En)'. Scores are reported "
        "per direction plus overall.",
        "Original script    = what was actually said, in the speaker's "
        "language (doctor rows: English line; patient rows: Spanish line).",
        "Interpreter output = the interpreter's rendition (verified "
        "transcript).",
        "Ground truth       = the certified reference translation of that "
        "turn.",
        "Encounter ID and Turn are optional helpers - ignored by the app.",
        "",
        "Rules:",
        "- All three text cells in a row MUST be the same utterance "
        "(alignment is critical).",
        "- Delete the two gray example rows before scoring.",
        "- Leave unusable cells truly empty - never write N/A or '-'.",
        "- Do not pre-clean casing/punctuation; keep disfluency handling "
        "consistent across columns.",
        "- Keep Direction labels spelled identically on every row.",
    ]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        data.to_excel(xl, sheet_name="Data", index=False)
        instructions.to_excel(xl, sheet_name="READ ME", index=False)
        ws = xl.book["Data"]
        for i, w in enumerate([13, 7, 17, 52, 52, 52], 1):
            ws.column_dimensions[get_column_letter(i)].width = w
        fill = PatternFill("solid", fgColor="0969DA")
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = fill
            cell.alignment = Alignment(vertical="center")
        for row in ws.iter_rows(min_row=2, max_row=3):
            for cell in row:
                cell.font = Font(italic=True, color="888888")
        xl.book["READ ME"].column_dimensions["A"].width = 115
        xl.book["READ ME"]["A1"].font = Font(bold=True)
    return buf.getvalue()


# ---------------------------------------------------------------- UI

st.title("🩺 Interpreter Translation Scorer")
st.caption("Upload a spreadsheet with the original script, the Interpreter output, "
           "and — optionally — a ground-truth human reference. You get overall "
           "translation scores (BLEU, chrF++, TER, semantic similarity, COMET) and "
           "a score for every sentence. With a ground truth, Interpreter is scored "
           "against the human reference; without one, it is scored against the "
           "original script.")

t1, t2 = st.columns([1, 2])
t1.download_button("📥 Download the input template (.xlsx)", template_xlsx(),
                   file_name="template_encounter.xlsx",
                   mime="application/vnd.openxmlformats-officedocument"
                        ".spreadsheetml.sheet",
                   key="tpl_dl")
t2.markdown(":red[**Your file must follow this template.**] One utterance "
            "per row; columns: Direction, Original script, Interpreter "
            "output, Ground truth. The template's READ ME sheet has the "
            "full rules — delete its gray example rows before uploading.")

uploaded = st.file_uploader(
    "Excel or CSV: original script, Interpreter output, ground truth "
    "(optional), and a direction/speaker column (optional — scores each "
    "direction separately)",
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
                 "(original script and Interpreter output).")
        st.stop()

    cols = list(df.columns)
    src_default, cand_default, gt_default = autodetect(cols)
    c1, c2, c3, c4 = st.columns(4)
    src_col = c1.selectbox("Original script (source) column", cols,
                           index=cols.index(src_default))
    cand_col = c2.selectbox("Interpreter output column", cols,
                            index=cols.index(cand_default))
    NONE = "— none (score against the original) —"
    gt_opts = [NONE] + cols
    gt_col = c3.selectbox("Ground truth (human reference) column — optional",
                          gt_opts,
                          index=gt_opts.index(gt_default) if gt_default else 0)
    gt_col = None if gt_col == NONE else gt_col
    DNONE = "— none (score everything together) —"
    dir_default = next(
        (c for c in cols
         if any(k in str(c).lower() for k in ("direction", "speaker", "role",
                                              "who", "turn type"))
         and c not in (src_col, cand_col, gt_col)), None)
    d_opts = [DNONE] + cols
    dir_col = c4.selectbox(
        "Direction/speaker column — optional",
        d_opts, index=d_opts.index(dir_default) if dir_default else 0,
        help="e.g. 'Doctor (En→Es)' / 'Patient (Es→En)'. When set, scores are "
             "reported separately per direction (in tabs) plus an overall tab.")
    dir_col = None if dir_col == DNONE else dir_col
    picked = [c for c in (src_col, cand_col, gt_col, dir_col) if c is not None]
    if len(set(picked)) < len(picked):
        st.error("The selected columns must all be different.")
        st.stop()

    use_cols = [src_col, cand_col] + ([gt_col] if gt_col else [])
    sub = df[use_cols].dropna()
    series = [sub[c].astype(str).str.strip() for c in use_cols]
    mask = np.logical_and.reduce([sr != "" for sr in series])
    srcs = series[0][mask].tolist()
    cands = series[1][mask].tolist()
    gts = series[2][mask].tolist() if gt_col else []
    dirs = []
    if dir_col:
        dseries = (df.loc[sub.index, dir_col].fillna("Unlabeled")
                   .astype(str).str.strip().replace("", "Unlabeled"))
        dirs = dseries[mask].tolist()
    if not srcs:
        st.error("No usable rows (empty cells were removed).")
        st.stop()

    if gt_col:
        st.caption("**3-column mode:** BLEU, chrF++ and TER compare the Interpreter "
                   "output against the **ground truth** (reference-based, as "
                   "these metrics were designed), and COMET uses its full "
                   "triplet (source = original, translation = Interpreter, reference = "
                   "ground truth) — one score each. Semantic similarity uses "
                   "**multilingual** embeddings and is computed against the "
                   "original for both Interpreter and ground truth, as a "
                   "meaning-preservation check that works across languages.")
    else:
        st.caption("**2-column mode:** no ground truth selected — all scores "
                   "compare Interpreter against the original script (for COMET, the "
                   "original serves as both source and reference).")

    if dirs and len(set(dirs)) > 1:
        groups = list(dict.fromkeys(dirs))  # preserve file order
        tabs = st.tabs([f"All directions ({len(srcs)})"] +
                       [f"{g} ({dirs.count(g)})" for g in groups])
        with tabs[0]:
            render_results(srcs, cands, gts, dirs, "all",
                           src_col, cand_col, gt_col, use_comet)
        for n, (tab, g) in enumerate(zip(tabs[1:], groups)):
            idx = [i for i, d in enumerate(dirs) if d == g]
            with tab:
                render_results([srcs[i] for i in idx],
                               [cands[i] for i in idx],
                               [gts[i] for i in idx] if gts else [],
                               [dirs[i] for i in idx], f"g{n}",
                               src_col, cand_col, gt_col, use_comet)
    else:
        render_results(srcs, cands, gts, [], "all",
                       src_col, cand_col, gt_col, use_comet)

# ---- legend & references (always visible)
st.divider()
st.subheader("Score legend & references")
st.markdown("""
| Score | Range | Columns compared | What it measures | Code | Publication |
|---|---|---|---|---|---|
| **BLEU** | 0–100, higher = more similar wording | Single score: Interpreter output **vs** ground truth (or the original script if no ground truth is selected) | Overlap of word sequences (1–4-gram precision) with the reference, plus a brevity penalty. Standard MT metric, per the [Microsoft Translator methodology](https://learn.microsoft.com/azure/ai-services/translator/custom-translator/concepts/bleu-score). | [mjpost/sacrebleu](https://github.com/mjpost/sacrebleu); methodology: [MicrosoftDocs/azure-ai-docs](https://github.com/MicrosoftDocs/azure-ai-docs/blob/main/articles/ai-services/translator/custom-translator/concepts/bleu-score.md) | [Papineni et al. (2002)](https://aclanthology.org/P02-1040/), ACL; implementation: [Post (2018)](https://aclanthology.org/W18-6319/), WMT |
| **chrF++** | 0–100, higher = more similar wording | Single score: Interpreter output **vs** ground truth (or the original if no ground truth is selected) | F-score over character 1–6-grams **and** word 1–2-grams (β=2); more forgiving of small word changes and morphology than BLEU. | [m-popovic/chrF](https://github.com/m-popovic/chrF) (computed via sacrebleu, `word_order=2`) | [Popović (2015)](https://aclanthology.org/W15-3049/), WMT; chrF++: [Popović (2017)](https://aclanthology.org/W17-4770/), WMT |
| **TER** | 0–100+, **lower** = closer (0 = identical) | Single score: Interpreter output **vs** ground truth (or the original if no ground truth is selected) | Translation Edit Rate: edits (insert/delete/substitute/shift) needed to turn the Interpreter output into the reference. | [mjpost/sacrebleu](https://github.com/mjpost/sacrebleu) | [Snover et al. (2006)](https://aclanthology.org/2006.amta-papers.25/), AMTA |
| **BERTScore** | 0–100, higher = closer in meaning | Single score: Interpreter output **vs** ground truth (or the original if no ground truth is selected) | Token-level F1 from greedy cosine matching of contextual token embeddings (bert-base-multilingual-cased); rewards semantic matches even when the wording differs. | [Tiiiger/bert_score](https://github.com/Tiiiger/bert_score) | [Zhang et al. (2020)](https://openreview.net/forum?id=SkeHuCVFDr), ICLR |
| **Semantic similarity** | 0–100%, higher = same meaning | Three comparisons: Interpreter output **vs** ground truth (direct); Interpreter output **vs** original; ground truth **vs** original (human ceiling). Multilingual embeddings, so cross-language comparisons are valid | Cosine similarity of sentence embeddings (paraphrase-multilingual-MiniLM-L12-v2); measures whether *meaning* is preserved regardless of wording or language. Drives the meaning verdicts (≥75% preserved, 55–75% review, <55% possible change). | [fivehills/TextSim_MTQE](https://github.com/fivehills/TextSim_MTQE) / [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) | Method: [Reimers & Gurevych (2019)](https://aclanthology.org/D19-1410/), EMNLP; multilingual model: [Reimers & Gurevych (2020)](https://aclanthology.org/2020.emnlp-main.365/), EMNLP; validation framework (human-judgment correlation, incl. cross-lingual En–Es): [Cer et al. (2017)](https://aclanthology.org/S17-2001/), SemEval |
| **COMET** | 0–100, higher = better quality | Single score, full triplet: source = original script, translation = Interpreter output, reference = ground truth (or original if none) | Neural metric (wmt22-comet-da) trained on human quality judgments of translations; sensitive to meaning errors rather than wording changes. | [Unbabel/COMET](https://github.com/Unbabel/COMET) | [Rei et al. (2020)](https://aclanthology.org/2020.emnlp-main.213/), EMNLP; model: [Rei et al. (2022)](https://aclanthology.org/2022.wmt-1.52/), WMT |
""")
st.caption(
    "**Implementation signatures** (for exact reproducibility): "
    "BLEU `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp` · "
    "chrF++ `nrefs:1|case:mixed|eff:yes|nc:6|nw:2` (β=2, character 6-grams + "
    "word 2-grams, matching the defaults of Popović's reference script "
    "`chrF++.py`) · "
    "TER `nrefs:1|case:lc|tok:tercom|norm:no|punct:yes` · "
    "semantic similarity: `sentence-transformers` "
    "paraphrase-multilingual-MiniLM-L12-v2, cosine similarity · "
    "BERTScore: `bert-score` package, "
    "`bert-base-multilingual-cased_L9_no-idf` (F1, no baseline rescaling) · "
    "COMET: `Unbabel/wmt22-comet-da` via `comet.load_from_checkpoint(...)"
    ".predict(batch_size=8, gpus=0)`. Corpus scores are computed with "
    "sacrebleu's corpus methods (not averaged sentence scores); per-sentence "
    "BLEU uses sacrebleu's default exponential smoothing; the corpus "
    "BERTScore is the mean per-sentence F1, as the bert-score tool reports.")
