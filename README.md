# Interpreter Translation Scorer

A web app to evaluate medical interpreters' translations against a certified
human ground truth, using standard machine-translation metrics. Built for
bidirectional encounters (e.g. doctor English → Spanish, patient Spanish →
English) scored from a single spreadsheet.

| Score | Compares | Implementation |
|---|---|---|
| **BLEU** | Interpreter output vs ground truth (single score) | [sacrebleu](https://github.com/mjpost/sacrebleu) (Papineni 2002; Post 2018), per the [Microsoft Translator methodology](https://github.com/MicrosoftDocs/azure-ai-docs/blob/main/articles/ai-services/translator/custom-translator/concepts/bleu-score.md) |
| **chrF++** | Interpreter output vs ground truth (single score) | [m-popovic/chrF](https://github.com/m-popovic/chrF) via sacrebleu, `word_order=2` (Popović 2015, 2017) |
| **TER** | Interpreter output vs ground truth (single score, lower = closer) | sacrebleu (Snover 2006) |
| **BERTScore** | Interpreter output vs ground truth (single score) | [bert_score](https://github.com/Tiiiger/bert_score), bert-base-multilingual-cased F1 (Zhang et al., ICLR 2020) |
| **COMET** | Full triplet: source = original, translation = interpreter, reference = ground truth | [Unbabel/COMET](https://github.com/Unbabel/COMET), wmt22-comet-da (Rei et al. 2020, 2022) |
| **Semantic similarity** | Interpreter vs original AND ground truth vs original (cross-lingual meaning-preservation check) | [sentence-transformers](https://github.com/UKPLab/sentence-transformers) multilingual embeddings, cosine (Reimers & Gurevych 2019, 2020; validated per Cer et al. 2017); method per [TextSim_MTQE](https://github.com/fivehills/TextSim_MTQE) |

## Input format

One Excel/CSV, one utterance per row:

- **Original script** — what was actually said, in the speaker's language
- **Interpreter output** — the interpreter's rendition (verified transcript)
- **Ground truth** — the certified reference translation of that turn
- **Direction** (optional) — e.g. `Doctor (En→Es)` / `Patient (Es→En)`;
  when present, results are shown per direction (tabs) plus overall
- Extra columns (encounter ID, turn number, …) are ignored

Columns are auto-detected and can be overridden in the UI. Rows with empty
cells are dropped. Every score carries an info tooltip stating what it
measures and how it is computed, and the in-app legend lists all references
and exact implementation signatures. All computations are verified
numerically against the reference implementations above.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

First run downloads the models (COMET ~2.3 GB, BERTScore ~700 MB,
embeddings ~500 MB). COMET can be toggled off for faster scoring.

## Deploy

- **Hugging Face Spaces** (recommended): free CPU tier fits the full COMET
  model. Create a Space with the Streamlit SDK and push this repo.
- **Streamlit Community Cloud**: works with the COMET toggle **off**
  (the model exceeds the free tier's memory).

## Note on data

Files uploaded to a publicly hosted instance are processed on the hosting
provider's servers. Do not upload patient-identifiable data to public
deployments; run locally instead.
