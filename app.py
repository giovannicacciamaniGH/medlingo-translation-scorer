#!/usr/bin/env python3
"""
MedLingo Translation Scorer — local web interface.

Upload an .xlsx or .csv with a column of original scripts and a column of
MedLingo output; see the overall (corpus) BLEU score and per-sentence scores.
BLEU per the standard Microsoft Translator methodology (sacrebleu: 1-4-gram
precision + brevity penalty).

Run:  python3 app.py   →  http://localhost:5089
"""

import io
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu
from flask import Flask, request, render_template_string, send_file, abort

# Same model as https://github.com/fivehills/TextSim_MTQE — loaded once, lazily.
_EMBEDDER = None

def embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _EMBEDDER


# COMET (https://github.com/Unbabel/COMET) — loaded once, lazily.
_COMET = None

def comet_model():
    global _COMET
    if _COMET is None:
        from comet import download_model, load_from_checkpoint
        _COMET = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    return _COMET


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

UPLOADS = {}   # token -> DataFrame
RESULTS = {}   # token -> results xlsx bytes


def load_table(filename: str, data: bytes) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    buf = io.BytesIO(data)
    if suffix in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(buf)
    if suffix in (".csv", ".txt"):
        return pd.read_csv(buf)
    if suffix == ".tsv":
        return pd.read_csv(buf, sep="\t")
    raise ValueError(f"Unsupported file type: {suffix} — use .xlsx or .csv")


def autodetect(cols):
    def find(keywords, exclude=None):
        for c in cols:
            name = str(c).lower()
            if exclude and c == exclude:
                continue
            if any(k in name for k in keywords):
                return c
        return None

    ref = find(["original", "script", "reference", "source", "doctor"])
    cand = find(["medlingo", "output", "candidate", "translation", "patient"], exclude=ref)
    if ref is None or cand is None or ref == cand:
        ref, cand = cols[0], cols[1]
    return ref, cand


def interpret(score):
    if score < 10: return "Almost no overlap"
    if score < 20: return "Low overlap"
    if score < 30: return "Gist preserved, heavily reworded"
    if score < 40: return "Moderate overlap"
    if score < 50: return "High overlap"
    if score < 60: return "Very high overlap"
    return "Near-identical"


def meaning_verdict(sim):
    """Verdict from semantic cosine similarity (0-1)."""
    if sim >= 0.75: return "Meaning preserved"
    if sim >= 0.55: return "Mostly preserved — review"
    return "Possible meaning change"


def score_df(df, ref_col, cand_col):
    sub = df[[ref_col, cand_col]].dropna()
    refs = sub[ref_col].astype(str).str.strip()
    cands = sub[cand_col].astype(str).str.strip()
    mask = (refs != "") & (cands != "")
    refs, cands = refs[mask].tolist(), cands[mask].tolist()
    if not refs:
        raise ValueError("No usable rows (empty cells were removed).")

    corpus = sacrebleu.corpus_bleu(cands, [refs])
    corpus_chrf = sacrebleu.corpus_chrf(cands, [refs])
    corpus_ter = sacrebleu.corpus_ter(cands, [refs])

    # Semantic cosine similarity, per TextSim_MTQE
    model = embedder()
    ref_emb = model.encode(refs, batch_size=64, show_progress_bar=False)
    cand_emb = model.encode(cands, batch_size=64, show_progress_bar=False)
    ref_emb = ref_emb / np.linalg.norm(ref_emb, axis=1, keepdims=True)
    cand_emb = cand_emb / np.linalg.norm(cand_emb, axis=1, keepdims=True)
    cosines = np.sum(ref_emb * cand_emb, axis=1).clip(-1, 1)

    # COMET (Unbabel/wmt22-comet-da): original script serves as src and ref
    comet_data = [{"src": r, "mt": c, "ref": r} for r, c in zip(refs, cands)]
    comet_out = comet_model().predict(comet_data, batch_size=8, gpus=0,
                                      num_workers=1, progress_bar=False)
    comet_scores = list(comet_out.scores)
    comet_system = float(comet_out.system_score)

    rows = []
    for i, (r, c) in enumerate(zip(refs, cands), 1):
        s = sacrebleu.sentence_bleu(c, [r], smooth_method="exp").score
        sim = float(cosines[i - 1])
        rows.append({"n": i, "ref": r, "cand": c,
                     "bleu": s, "label": interpret(s),
                     "sem": sim, "verdict": meaning_verdict(sim),
                     "chrf": sacrebleu.sentence_chrf(c, [r]).score,
                     "ter": sacrebleu.sentence_ter(c, [r]).score,
                     "comet": float(comet_scores[i - 1])})
    extras = {"chrf": corpus_chrf.score, "ter": corpus_ter.score,
              "comet": comet_system}
    return corpus, rows, extras


PAGE = """
<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MedLingo Translation Scorer</title>
<style>
:root { --bg:#f6f8fa; --card:#fff; --ink:#1f2328; --muted:#59636e;
        --accent:#0969da; --border:#d1d9e0; }
* { box-sizing:border-box; }
body { margin:0; font-family:-apple-system,'Segoe UI',Roboto,sans-serif;
       background:var(--bg); color:var(--ink); }
.wrap { max-width:1100px; margin:0 auto; padding:32px 20px 60px; }
h1 { font-size:26px; margin:0 0 4px; }
.sub { color:var(--muted); margin:0 0 28px; }
.card { background:var(--card); border:1px solid var(--border);
        border-radius:12px; padding:24px; margin-bottom:20px; }
.drop { border:2px dashed var(--border); border-radius:12px; padding:44px 20px;
        text-align:center; color:var(--muted); cursor:pointer; transition:.15s; }
.drop.hover { border-color:var(--accent); background:#eef4fc; }
.drop strong { color:var(--accent); }
input[type=file] { display:none; }
.btn { background:var(--accent); color:#fff; border:0; border-radius:8px;
       padding:10px 22px; font-size:15px; cursor:pointer; }
.btn:disabled { opacity:.45; cursor:default; }
select { padding:7px 10px; border:1px solid var(--border); border-radius:8px;
         font-size:14px; background:#fff; }
.controls { display:flex; gap:18px; flex-wrap:wrap; align-items:end; margin-top:18px; }
.controls label { display:block; font-size:12px; color:var(--muted); margin-bottom:4px; }
.scoregrid { display:flex; gap:16px; flex-wrap:wrap; }
.scoregrid.main { display:grid; grid-template-columns:repeat(5, 1fr);
                  margin-bottom:16px; }
@media (max-width: 900px) { .scoregrid.main { grid-template-columns:repeat(2, 1fr); } }
.stat { flex:1 1 140px; background:var(--card); border:1px solid var(--border);
        border-radius:12px; padding:18px 20px; }
.stat .v { font-size:30px; font-weight:700; }
.stat.big { border-color:var(--accent); }
.stat.big .v { font-size:34px; color:var(--accent); }
.stat .k { color:var(--muted); font-size:13px; margin-top:2px; }
table { width:100%; border-collapse:collapse; font-size:14px; }
th, td { text-align:left; padding:10px 12px; border-bottom:1px solid var(--border);
         vertical-align:top; }
th { color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase;
     letter-spacing:.04em; position:sticky; top:0; background:var(--card); }
td.num { font-variant-numeric:tabular-nums; font-weight:700; white-space:nowrap; }
.pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px;
        font-weight:600; white-space:nowrap; }
.p0 { background:#ffebe9; color:#cf222e; } .p1 { background:#fff1e5; color:#bc4c00; }
.p2 { background:#fff8c5; color:#7d4e00; } .p3 { background:#ddf4ff; color:#0969da; }
.p4 { background:#dafbe1; color:#1a7f37; }
.m0 { background:#ffebe9; color:#cf222e; } .m1 { background:#fff8c5; color:#7d4e00; }
.m2 { background:#dafbe1; color:#1a7f37; }
.chiplabel { font-size:12px; color:var(--muted); font-weight:600; margin:14px 0 6px;
             text-transform:uppercase; letter-spacing:.04em; }
.chips { display:flex; gap:8px; flex-wrap:wrap; margin:0 0 14px; }
.chip { border:1.5px solid var(--border); background:var(--card); border-radius:999px;
        padding:6px 14px; font-size:13px; font-weight:600; cursor:pointer;
        color:var(--muted); transition:.12s; }
.chip:hover { border-color:var(--accent); color:var(--accent); }
.chip.active { border-color:var(--accent); background:var(--accent); color:#fff; }
.chip .cnt { font-weight:400; opacity:.75; margin-left:4px; }
.note { font-size:13px; color:var(--muted); line-height:1.5; margin-top:14px; }
a.dl { color:var(--accent); font-weight:600; text-decoration:none; }
.tablewrap { max-height:520px; overflow:auto; border:1px solid var(--border);
             border-radius:10px; }
.err { background:#ffebe9; border:1px solid #cf222e33; color:#cf222e;
       border-radius:10px; padding:12px 16px; margin-bottom:20px; }
table.legend td { font-size:13px; line-height:1.5; min-width:120px; }
table.legend td:nth-child(3) { min-width:240px; }
table.legend a { color:var(--accent); text-decoration:none; }
</style></head><body><div class="wrap">
<h1>MedLingo Translation Scorer</h1>
<p class="sub">Upload a spreadsheet with the original script and the MedLingo output
— get overall translation scores (BLEU, chrF, TER, semantic similarity, COMET)
and a score for every sentence.</p>

{% if error %}<div class="err">{{ error }}</div>{% endif %}

<div class="card">
  <form id="f" method="post" action="/score" enctype="multipart/form-data">
    <div class="drop" id="drop">
      <div style="font-size:34px">&#128196;</div>
      <p><strong>Choose a file</strong> or drag it here</p>
      <p style="font-size:12px">.xlsx or .csv — one column with the original script,
      one with the MedLingo output</p>
      <p id="fname" style="font-weight:600;color:var(--ink)"></p>
      <input type="file" id="file" name="file" accept=".xlsx,.xlsm,.xls,.csv,.tsv,.txt">
    </div>
    <div class="controls">
      <button class="btn" id="go" type="submit" disabled>Compute translation scores</button>
      <span class="note" style="margin-top:0">Columns are auto-detected — you can change them after upload.</span>
    </div>
  </form>
</div>

{% if result %}
<div class="card">
  <form method="post" action="/rescore" class="controls" style="margin-top:0">
    <input type="hidden" name="token" value="{{ token }}">
    <div><label>Original script (reference) column</label>
      <select name="ref">{% for c in columns %}
        <option value="{{ c }}" {% if c == ref_col %}selected{% endif %}>{{ c }}</option>
      {% endfor %}</select></div>
    <div><label>MedLingo output column</label>
      <select name="cand">{% for c in columns %}
        <option value="{{ c }}" {% if c == cand_col %}selected{% endif %}>{{ c }}</option>
      {% endfor %}</select></div>
    <button class="btn" type="submit">Re-score</button>
  </form>
</div>

<div class="scoregrid main">
  <div class="stat big"><div class="v">{{ "%.1f" % corpus.score }}</div>
    <div class="k">Overall BLEU (corpus) — {{ corpus_label }}</div></div>
  <div class="stat big"><div class="v">{{ "%.0f" % (mean_sem * 100) }}%</div>
    <div class="k">Mean semantic similarity (meaning)</div></div>
  <div class="stat big"><div class="v">{{ "%.0f" % (comet_system * 100) }}</div>
    <div class="k">COMET (system, 0&ndash;100)</div></div>
  <div class="stat big"><div class="v">{{ "%.1f" % chrf_corpus }}</div><div class="k">chrF (corpus)</div></div>
  <div class="stat big"><div class="v">{{ "%.1f" % ter_corpus }}</div><div class="k">TER (corpus, lower = closer)</div></div>
</div>
<div class="scoregrid">
  <div class="stat"><div class="v">{{ nrows }}</div><div class="k">Sentences scored</div></div>
  <div class="stat"><div class="v">{{ "%.1f" % mean_sent }}</div><div class="k">Mean sentence BLEU</div></div>
  <div class="stat"><div class="v">{{ "%.2f" % corpus.bp }}</div><div class="k">Brevity penalty</div></div>
  <div class="stat"><div class="v">{{ precisions }}</div><div class="k">1&ndash;4-gram precision</div></div>
</div>

<div class="card" style="margin-top:20px">
  <p style="margin:0 0 14px"><a class="dl" href="/download/{{ token }}">&#11015;&#65039;
    Download full results (.xlsx)</a></p>
  <div class="chiplabel">Filter by meaning (semantic similarity)</div>
  <div class="chips" data-group="verdict">
    <button type="button" class="chip active" data-value="__all__"
      onclick="filterRows(this)">All <span class="cnt">{{ nrows }}</span></button>
    {% for lab, cnt in verdict_counts %}
    <button type="button" class="chip" data-value="{{ lab }}"
      onclick="filterRows(this)">{{ lab }} <span class="cnt">{{ cnt }}</span></button>
    {% endfor %}
  </div>
  <div class="chiplabel">Filter by wording overlap (BLEU)</div>
  <div class="chips" data-group="label">
    <button type="button" class="chip active" data-value="__all__"
      onclick="filterRows(this)">All <span class="cnt">{{ nrows }}</span></button>
    {% for lab, cnt in label_counts %}
    <button type="button" class="chip" data-value="{{ lab }}"
      onclick="filterRows(this)">{{ lab }} <span class="cnt">{{ cnt }}</span></button>
    {% endfor %}
  </div>
  <div class="tablewrap" style="margin-top:14px"><table>
    <tr><th>#</th><th>Original script</th><th>MedLingo output</th>
      <th>BLEU</th><th>Wording</th><th>Semantic</th><th>Meaning</th>
      <th>COMET</th><th>chrF</th><th>TER</th></tr>
    {% for r in rows %}
    <tr data-label="{{ r.label }}" data-verdict="{{ r.verdict }}">
      <td>{{ r.n }}</td><td>{{ r.ref }}</td><td>{{ r.cand }}</td>
      <td class="num">{{ "%.1f" % r.bleu }}</td>
      <td><span class="pill p{{ r.cls }}">{{ r.label }}</span></td>
      <td class="num">{{ "%.0f" % (r.sem * 100) }}%</td>
      <td><span class="pill m{{ r.mcls }}">{{ r.verdict }}</span></td>
      <td class="num">{{ "%.0f" % (r.comet * 100) }}</td>
      <td class="num">{{ "%.1f" % r.chrf }}</td>
      <td class="num">{{ "%.1f" % r.ter }}</td></tr>
    {% endfor %}
  </table></div>
  <p class="note"><b>Reading the scores:</b>
  <b>BLEU</b>, <b>chrF</b> and <b>TER</b> (via <a href="https://github.com/mjpost/sacrebleu">sacrebleu</a>)
  measure surface overlap with the original &mdash; they penalize
  reworded text even when the rewording is a perfect simplification. chrF works on
  character n-grams (more forgiving of small word changes); TER counts the edits needed
  to turn the output back into the original (<i>lower</i> = closer, 0 = identical).
  <b>Semantic similarity</b> (%, sentence embeddings, per
  <a href="https://github.com/fivehills/TextSim_MTQE">TextSim_MTQE</a>) and
  <b>COMET</b> (<a href="https://github.com/Unbabel/COMET">Unbabel wmt22-comet-da</a>,
  a neural metric trained on human quality judgments) look past wording toward
  <i>meaning</i>: &ldquo;myocardial infarction&rdquo; &rarr; &ldquo;heart attack&rdquo;
  scores low on BLEU but high on both.
  The sweet spot for MedLingo: <b>high semantic/COMET + low/moderate BLEU</b> =
  meaning preserved, wording simplified. Rows flagged
  <span class="pill m0">Possible meaning change</span> deserve a manual read.</p>
</div>

<div class="card">
  <h2 style="font-size:18px;margin:0 0 4px">Score legend &amp; references</h2>
  <p class="note" style="margin:0 0 14px">Each metric with its range, direction,
  implementation, and original publication.</p>
  <div class="tablewrap"><table class="legend">
    <tr><th>Score</th><th>Range</th><th>What it measures</th><th>Code</th><th>Publication</th></tr>
    <tr><td><b>BLEU</b></td>
      <td>0&ndash;100, higher = more similar wording</td>
      <td>Overlap of word sequences (1&ndash;4-gram precision) with the original,
        with a brevity penalty. The standard machine-translation metric, per the
        <a href="https://learn.microsoft.com/azure/ai-services/translator/custom-translator/concepts/bleu-score">Microsoft Translator methodology</a>.</td>
      <td><a href="https://github.com/mjpost/sacrebleu">mjpost/sacrebleu</a>;
        methodology: <a href="https://github.com/MicrosoftDocs/azure-ai-docs/blob/main/articles/ai-services/translator/custom-translator/concepts/bleu-score.md">MicrosoftDocs/azure-ai-docs (BLEU score)</a></td>
      <td>Papineni et&nbsp;al. (2002), <a href="https://aclanthology.org/P02-1040/">&ldquo;BLEU:
        a Method for Automatic Evaluation of Machine Translation&rdquo;</a>, ACL.
        Implementation: Post (2018), <a href="https://aclanthology.org/W18-6319/">&ldquo;A Call
        for Clarity in Reporting BLEU Scores&rdquo;</a>, WMT.</td></tr>
    <tr><td><b>chrF</b></td>
      <td>0&ndash;100, higher = more similar wording</td>
      <td>Character n-gram F-score &mdash; like BLEU but on character level, so it is
        more forgiving of small word changes and morphology.</td>
      <td><a href="https://github.com/m-popovic/chrF">m-popovic/chrF</a>
        (computed via sacrebleu)</td>
      <td>Popovi&#263; (2015), <a href="https://aclanthology.org/W15-3049/">&ldquo;chrF:
        character n-gram F-score for automatic MT evaluation&rdquo;</a>, WMT.</td></tr>
    <tr><td><b>TER</b></td>
      <td>0&ndash;100+, <b>lower</b> = closer (0 = identical)</td>
      <td>Translation Edit Rate: how many edits (insert/delete/substitute/shift) are
        needed to turn the MedLingo output back into the original.</td>
      <td><a href="https://github.com/mjpost/sacrebleu">mjpost/sacrebleu</a></td>
      <td>Snover et&nbsp;al. (2006), <a href="https://aclanthology.org/2006.amta-papers.25/">&ldquo;A
        Study of Translation Edit Rate with Targeted Human Annotation&rdquo;</a>, AMTA.</td></tr>
    <tr><td><b>Semantic similarity</b></td>
      <td>0&ndash;100%, higher = same meaning</td>
      <td>Cosine similarity of sentence embeddings
        (paraphrase-multilingual-MiniLM-L12-v2) &mdash; measures whether the
        <i>meaning</i> is preserved regardless of wording. Drives the meaning verdict
        chips (&ge;75% preserved, 55&ndash;75% review, &lt;55% possible change).</td>
      <td><a href="https://github.com/fivehills/TextSim_MTQE">fivehills/TextSim_MTQE</a>
        / <a href="https://github.com/UKPLab/sentence-transformers">UKPLab/sentence-transformers</a></td>
      <td>Reimers &amp; Gurevych (2019), <a href="https://aclanthology.org/D19-1410/">&ldquo;Sentence-BERT:
        Sentence Embeddings using Siamese BERT-Networks&rdquo;</a>, EMNLP.</td></tr>
    <tr><td><b>COMET</b></td>
      <td>0&ndash;100, higher = better quality</td>
      <td>Neural metric (model wmt22-comet-da) trained on human quality judgments of
        translations; sensitive to meaning errors rather than wording changes.</td>
      <td><a href="https://github.com/Unbabel/COMET">Unbabel/COMET</a></td>
      <td>Rei et&nbsp;al. (2020), <a href="https://aclanthology.org/2020.emnlp-main.213/">&ldquo;COMET:
        A Neural Framework for MT Evaluation&rdquo;</a>, EMNLP; model: Rei et&nbsp;al. (2022),
        <a href="https://aclanthology.org/2022.wmt-1.52/">&ldquo;COMET-22: Unbabel-IST 2022
        Submission for the Metrics Shared Task&rdquo;</a>, WMT.</td></tr>
  </table></div>
</div>
{% endif %}
</div>
<script>
function filterRows(chip) {
  const group = chip.closest('.chips');
  group.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
  chip.classList.add('active');
  const want = {};
  document.querySelectorAll('.chips[data-group]').forEach(g => {
    want[g.dataset.group] = g.querySelector('.chip.active').dataset.value;
  });
  document.querySelectorAll('tr[data-label]').forEach(tr => {
    const okLabel = want.label === '__all__' || tr.dataset.label === want.label;
    const okVerdict = want.verdict === '__all__' || tr.dataset.verdict === want.verdict;
    tr.style.display = (okLabel && okVerdict) ? '' : 'none';
  });
}
const drop = document.getElementById('drop'), input = document.getElementById('file'),
      go = document.getElementById('go'), fname = document.getElementById('fname');
drop.addEventListener('click', () => input.click());
input.addEventListener('change', () => { if (input.files.length) {
  fname.textContent = input.files[0].name; go.disabled = false; }});
['dragover','dragenter'].forEach(e => drop.addEventListener(e, ev => {
  ev.preventDefault(); drop.classList.add('hover'); }));
['dragleave','drop'].forEach(e => drop.addEventListener(e, ev => {
  ev.preventDefault(); drop.classList.remove('hover'); }));
drop.addEventListener('drop', ev => { input.files = ev.dataTransfer.files;
  if (input.files.length) { fname.textContent = input.files[0].name; go.disabled = false; }});
</script>
</body></html>
"""


def pill_class(score):
    if score < 20: return 0
    if score < 30: return 1
    if score < 40: return 2
    if score < 50: return 3
    return 4


def render(df=None, token=None, ref_col=None, cand_col=None, error=None):
    ctx = {"error": error, "result": False}
    if df is not None:
        try:
            corpus, rows, extras = score_df(df, ref_col, cand_col)
        except ValueError as e:
            return render_template_string(PAGE, error=str(e), result=False)
        verdict_cls = {"Possible meaning change": 0,
                       "Mostly preserved — review": 1,
                       "Meaning preserved": 2}
        for r in rows:
            r["cls"] = pill_class(r["bleu"])
            r["mcls"] = verdict_cls[r["verdict"]]
        mean_sent = sum(r["bleu"] for r in rows) / len(rows)
        mean_sem = sum(r["sem"] for r in rows) / len(rows)

        # build downloadable results file
        out = pd.DataFrame({
            "Original script": [r["ref"] for r in rows],
            "MedLingo output": [r["cand"] for r in rows],
            "Sentence BLEU": [round(r["bleu"], 2) for r in rows],
            "BLEU interpretation": [r["label"] for r in rows],
            "Semantic similarity": [round(r["sem"], 3) for r in rows],
            "Meaning verdict": [r["verdict"] for r in rows],
            "COMET": [round(r["comet"], 3) for r in rows],
            "chrF": [round(r["chrf"], 2) for r in rows],
            "TER": [round(r["ter"], 2) for r in rows],
        })
        summary = pd.DataFrame({
            "Metric": ["Corpus BLEU", "Sentences scored", "Brevity penalty",
                       "1-gram precision", "2-gram precision",
                       "3-gram precision", "4-gram precision", "Mean sentence BLEU",
                       "Mean semantic similarity", "COMET system score",
                       "Corpus chrF", "Corpus TER"],
            "Value": [round(corpus.score, 2), len(rows), round(corpus.bp, 3),
                      *[round(p, 1) for p in corpus.precisions],
                      round(mean_sent, 2), round(mean_sem, 3),
                      round(extras["comet"], 3), round(extras["chrf"], 2),
                      round(extras["ter"], 2)],
        })
        buf = io.BytesIO()
        with pd.ExcelWriter(buf) as xl:
            out.to_excel(xl, sheet_name="Per-sentence scores", index=False)
            summary.to_excel(xl, sheet_name="Summary", index=False)
        RESULTS[token] = buf.getvalue()

        label_order = ["Almost no overlap", "Low overlap",
                       "Gist preserved, heavily reworded", "Moderate overlap",
                       "High overlap", "Very high overlap", "Near-identical"]
        counts = {}
        for r in rows:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        label_counts = [(lab, counts[lab]) for lab in label_order if lab in counts]

        verdict_order = ["Possible meaning change", "Mostly preserved — review",
                         "Meaning preserved"]
        vcounts = {}
        for r in rows:
            vcounts[r["verdict"]] = vcounts.get(r["verdict"], 0) + 1
        verdict_counts = [(v, vcounts[v]) for v in verdict_order if v in vcounts]

        ctx.update(result=True, token=token, columns=list(df.columns),
                   ref_col=ref_col, cand_col=cand_col, corpus=corpus,
                   corpus_label=interpret(corpus.score), rows=rows,
                   label_counts=label_counts, verdict_counts=verdict_counts,
                   mean_sem=mean_sem,
                   comet_system=extras["comet"], chrf_corpus=extras["chrf"],
                   ter_corpus=extras["ter"],
                   nrows=len(rows), mean_sent=mean_sent,
                   precisions=" / ".join(f"{p:.0f}" for p in corpus.precisions))
    return render_template_string(PAGE, **ctx)


@app.get("/")
def index():
    return render()


@app.post("/score")
def score():
    f = request.files.get("file")
    if not f or not f.filename:
        return render(error="Please choose a file first.")
    try:
        df = load_table(f.filename, f.read())
    except Exception as e:
        return render(error=f"Could not read the file: {e}")
    if len(df.columns) < 2:
        return render(error="The file needs at least two columns "
                            "(original script and MedLingo output).")
    token = uuid.uuid4().hex[:12]
    UPLOADS[token] = df
    ref_col, cand_col = autodetect(list(df.columns))
    return render(df, token, ref_col, cand_col)


@app.post("/rescore")
def rescore():
    token = request.form.get("token")
    df = UPLOADS.get(token)
    if df is None:
        return render(error="Session expired — please upload the file again.")
    ref_col, cand_col = request.form.get("ref"), request.form.get("cand")
    if ref_col == cand_col:
        return render(df, token, *autodetect(list(df.columns)),
                      error="Reference and MedLingo columns must be different.")
    if ref_col not in df.columns or cand_col not in df.columns:
        abort(400)
    return render(df, token, ref_col, cand_col)


@app.get("/download/<token>")
def download(token):
    data = RESULTS.get(token)
    if data is None:
        abort(404)
    return send_file(io.BytesIO(data), as_attachment=True,
                     download_name="bleu_results.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument"
                              ".spreadsheetml.sheet")


if __name__ == "__main__":
    # Load models in the main thread at startup — transformers can fail with
    # meta-tensor errors when first loaded inside a Flask worker thread.
    print("Loading scoring models (first time can take a minute)...")
    embedder()
    comet_model()
    print("Models ready.")
    app.run(host="127.0.0.1", port=5089, debug=False)
