# Excensia Vantage

Frame-level ball possession from automatic football tracking. Entry of team **Excensia** for
*More than a Hack 2027, Phase 1* (Metrica Sports, FC Barcelona, Mobile World Capital Barcelona).

For every frame of a match Vantage says whether the home team has the ball (`H`), the away team has
it (`A`) or the ball is dead (`D`). It reads the provided automatic tracking and Smart Tagging and
nothing else, and everything it has learned comes from **other** matches.

- [docs/TECHNICAL_DOCUMENT.md](docs/TECHNICAL_DOCUMENT.md): the method, the data, the limits.
- `viewer/`: a browser viewer for a result (possession timeline, team shape, the evidence behind
  each call). It ships with a simulated match and contains no match data. To see a real run in
  it, `python -m vantage export --data-dir /path/to/data --out run.json --page match.html` writes
  the viewer with the match built in, as one local file, and `node viewer/scripts/show.mjs
  match.html` opens it as a window of its own; `scripts/export_external.py` does the same for
  one of the open matches. A page made from the competition match holds its tracking and must
  stay on your machine. It shows that match drawn from its tracking and never its video: the
  organisers' rules do not allow output to be checked against the video, the run file of that
  match says so, and the viewer then opens no video beside it. `viewer/scripts/page-check.mjs`
  walks every tab of such a page and reports counts only, with no picture. What may be watched
  with that match's video is the organisers' own public five minutes, since they allow
  inspecting the provided data and the video to understand what counts as possession:
  `scripts/public_truth_run.py` writes those labels as a page, and `node viewer/scripts/show.mjs
  that-page.html --video=/path/to/tactical.mp4` plays the two together. That page also carries
  the camera of the video (the provided homography), so the viewer draws the provided tracking
  on the real picture: the ball close-up is cut from the video, and the teams' outlines and the
  players' runs are drawn on it, in the small evidence pictures and on the large video.
  `viewer/scripts/footage-check.mjs` proves the placing on a film made for the purpose, and
  `viewer/scripts/footage-align.mjs` measures it on a real video in numbers only.

## Reproduce the submitted file

You need the organisers' data bundle (the folder that contains `tactical/`). It is not in this
repository and must not be added to it.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt            # Windows: .venv\Scripts\pip
.venv/bin/pip install --no-deps -e .
.venv/bin/python -m vantage run --data-dir /path/to/data --out Excensia.csv
```

Python 3.12 (`.python-version` has the exact one used). The pinned packages in `requirements.txt`
are the whole runtime: numpy, scipy, lightgbm and lightgbm's one dependency. On Linux, LightGBM
also needs the OpenMP runtime, which a minimal image may lack (`apt-get install libgomp1`); on
macOS, `brew install libomp`.

The run takes about a minute on a laptop, uses no GPU and no network, and is deterministic. The
trained models are fixed files in `models/`, the features use only exact arithmetic (no
transcendental functions, no linear-algebra library), and a prediction is a sum of constants chosen
by threshold tests. The same file, byte for byte, came out on Windows 11 and on Ubuntu 24.04.

The command prints the SHA-256 of the file it wrote and stores it in `Excensia.receipt.json`
together with the hashes of the inputs, the models and the package versions. The submitted file is

```
sha256  c2e341a77b403bb6253bfda58799193aa1de95c34cce739a571ae523561d0c08
```

Check any file before sending it:

```bash
python /path/to/more-than-a-hack-2027/evaluation/validate_submission.py Excensia.csv
python scripts/pipeline_check.py Excensia.csv --data-dir /path/to/data \
    --official /path/to/more-than-a-hack-2027
```

The first is the organisers' own validator. The second repeats it, prints the file's label-free
totals, and reads the public five minutes once as a pipeline check (frames aligned, home and away
the right way round). It chooses nothing.

## What is in here

```
vantage/
  epts.py            reader for the FIFA EPTS tracking format (follows the layout blocks)
  competition.py     the match: published frame ranges, loading, the submission writer
  features.py        per-frame feature table: ball, gaps in the ball, touches, team shape
  model.py           two gradient-boosted tree models, scored in both mirror views
  decode.py          Viterbi decoding into whole possessions, with minimum lengths
  tagging.py         Smart Tagging as bounded votes
  degrade.py         simulator that makes corrected tracking look like automatic tracking
  labels.py          the external possession labels, and the challenge's definition
  dfl.py, metrica_sample.py, external.py     the ten external matches
  skillcorner.py     twenty more, real automatic tracking: a check on the method, not training data
  broadcast.py       the second recording's tracking placed on the tactical frames; the entry
                     does not use it
  evaluate.py, _official_metrics.py          scoring with the organisers' own metric code
  viewer_export.py   writes a run for the viewer: labels, confidence, evidence, tracking
  anymatch.py        the same engine on any other match's tracking (`python -m vantage tracking`);
                     the entry does not use it
  cli.py             `python -m vantage run` and `python -m vantage export`
models/
  vantage_state.txt      frame classifier: home, away or dead ball
  vantage_boundary.txt   boundary model: turnover, stoppage or restart happening here
  vantage.json           the card: features, decoding settings, provenance, file hashes
scripts/             fetch the external data, cross-validate, train, compare domains (below)
tests/               checks on generated data only
viewer/              the browser viewer
```

## Train it again from nothing

Training never touches the competition match.

```bash
pip install -r requirements-dev.txt
python scripts/fetch_external.py                       # open DFL dataset, 2.6 GB, CC BY 4.0
git clone https://github.com/metrica-sports/sample-data data/external/metrica-sample
python scripts/cv.py prepare                           # simulate automatic tracking, build tables
python scripts/cv.py fold --tag cv --test J03WMX metrica1
python scripts/cv.py fold --tag cv --test J03WN1 metrica2
python scripts/cv.py fold --tag cv --test J03WOH metrica3
python scripts/cv.py fold --tag cv --test J03WOY J03WPY
python scripts/cv.py fold --tag cv --test J03WQQ J03WR9
python scripts/cv.py summary --tag cv --sources metrica  # one decoding setting for all folds
python scripts/train_final.py --decode '<json printed by the summary>'
```

The defaults of these scripts are the settings of the shipped models (`models/vantage.json`
records them: the weight on short possessions, the rounds, the seed). A fold takes about twenty
minutes on eight cores. Every random draw is seeded from the match name, so the tables and the
models can be rebuilt. Retraining is not needed to reproduce the submission: the models that
produced it are committed.

The other scripts are the checks behind statements in the technical document:

| Script | What it prints |
|---|---|
| `match_check.py` | The loader against the figures published in DATA.md. Totals only. |
| `domain_report.py`, `gap_report.py`, `player_report.py`, `slow_report.py` | Label-free totals of the provided tracking beside the same totals for simulated tracking. They shaped the simulator. |
| `stoppage_check.py` | Where the ball is untracked through the match and how much of that a file calls dead ball (no labels), and, on the public five minutes, whether dead ball is found with and without a tracked ball. |
| `tag_check.py`, `set_piece_profile.py` | Label-free totals of Smart Tagging: which tags are fixed-length clips, where tags sit relative to the tracked ball, and where in a set-piece clip the restart falls. |
| `behaviour_report.py` | How the method behaves on held-out external matches, in the totals `pipeline_check.py` prints for a submission file. |
| `error_report.py` | Which possessions are missed on held-out external matches, by length and by kind of miss. |
| `label_stats.py` | The external labels: possessions per minute, lengths, and what a minimum length would do to them. |
| `export_external.py` | A run file for the viewer from one of the open matches, so the whole path can be seen on open data. |
| `public_truth_run.py` | The organisers' public five minutes as a run file for the viewer: their labels and the provided tracking, nothing from the model. Totals only are printed. |
| `adversarial.py` | Whether a classifier can tell simulated from provided tracking, and on which features. |
| `fetch_skillcorner.py`, `skillcorner_check.py`, `headline_cv.py` | SkillCorner's open A-League matches (1.9 GB, MIT licence): download with checksums, the shipped models run on them unchanged, and the parts of the score for any saved set of predictions. |
| `skillcorner_look.py`, `skillcorner_labels.py`, `skillcorner_flag.py`, `skillcorner_dead.py` | How that source's files agree with each other, what its labels look like, and when its stoppages begin and end. |
| `broadcast_align.py`, `broadcast_try.py` | The broadcast tracking placed on the tactical frames, and how many labels its ball would change. Totals only; nothing is written into the submission. |
| `average_cv.py`, `stack_cv.py` | Two things tried on the finished method and not adopted: averaging seeds, and a second model on the first one's held-out predictions. |

## Tests

```bash
python -m pytest
```

They run on generated data and need no match data of any kind.

## Data and licences

- **No match data is in this repository**, and none may be added: the competition assets are
  licensed for the competition only. `.gitignore` excludes `data/`, `runs/` and video files.
- Training and validation use ten other matches from two open sources:
  - Bassek, Rein, Weber, Memmert (2025), *An integrated dataset of spatiotemporal and event data
    in elite soccer*, doi:10.6084/m9.figshare.28196177.v1, CC BY 4.0 (seven matches);
  - Metrica Sports sample data, https://github.com/metrica-sports/sample-data (three matches).
- A third open source is used to check the finished method and not to train it: SkillCorner Open
  Data, https://github.com/SkillCorner/opendata, MIT licence (twenty matches; credit: SkillCorner).
- `vantage/_official_metrics.py` is the organisers' metric code, vendored unchanged under its MIT
  licence.
- This repository's own code is released under the MIT licence (see `LICENSE`).
