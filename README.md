# MedLingo Translation Scorer

A web app to evaluate MedLingo's patient-friendly rewrites of medical scripts
against the original text, using standard machine-translation metrics:

| Score | Measures | Implementation |
|---|---|---|
| **BLEU** | word-sequence overlap (1–4-grams + brevity penalty) | [sacrebleu](https://github.com/mjpost/sacrebleu), per the [Microsoft Translator methodology](https://github.com/MicrosoftDocs/azure-ai-docs/blob/main/articles/ai-services/translator/custom-translator/concepts/bleu-score.md) |
| **chrF** | character n-gram F-score | [m-popovic/chrF](https://github.com/m-popovic/chrF) via sacrebleu |
| **TER** | edit rate back to the original (lower = closer) | sacrebleu |
| **Semantic similarity** | meaning preservation via sentence embeddings | [TextSim_MTQE](https://github.com/fivehills/TextSim_MTQE) / [sentence-transformers](https://github.com/UKPLab/sentence-transformers) |
| **COMET** | neural quality estimate trained on human judgments | [Unbabel/COMET](https://github.com/Unbabel/COMET) (wmt22-comet-da) |

Upload an Excel/CSV with one column of original scripts and one of MedLingo
output; get corpus-level scores, per-sentence scores with meaning verdicts
("Meaning preserved" / "Mostly preserved — review" / "Possible meaning change"),
filters, and a downloadable results spreadsheet. Full references for each
metric are shown in the app's legend.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The first run downloads the scoring models (the COMET model is ~2.3 GB).
COMET can be toggled off in the app for faster scoring.

## Deploy

Works on [Streamlit Community Cloud](https://share.streamlit.io) (toggle COMET
off — the model exceeds the free tier's memory) or Hugging Face Spaces
(free CPU tier fits full COMET).

## Interpreting the scores

BLEU, chrF and TER measure *surface* wording overlap, so a good
simplification ("myocardial infarction" → "heart attack") scores low on them
by design. Semantic similarity and COMET measure *meaning*. The desired
pattern for patient-friendly rewriting is **high semantic/COMET with
low-to-moderate BLEU**: meaning preserved, wording simplified. Sentences
flagged "Possible meaning change" warrant manual review.

## Note on data

Files uploaded to a publicly hosted instance are processed on the hosting
provider's servers. Do not upload patient-identifiable data to public
deployments; run locally instead.
