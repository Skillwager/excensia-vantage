# Excensia Vantage

Frame-level ball possession from automatic football tracking. Entry of team **Excensia** for
*More than a Hack 2027, Phase 1* (Metrica Sports, FC Barcelona, Mobile World Capital Barcelona).

For every frame of a match Vantage says whether the home team has the ball (`H`), the away team has
it (`A`) or the ball is dead (`D`). It reads the provided automatic tracking and Smart Tagging and
nothing else, and everything it has learned comes from **other** matches.

- [docs/TECHNICAL_DOCUMENT.md](docs/TECHNICAL_DOCUMENT.md): the method, the data, the limits.
- `viewer/`: a browser viewer for a result (possession timeline, team shape, the evidence behind
  each call). It ships with a simulated match and contains no match data.

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
sha256  e45e63ad3cb0f8095b48949cfe56c1adcf300aadeca99c56b886ff77d090a5dd
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
  evaluate.py, _official_metrics.py          scoring with the organisers' own metric code
  viewer_export.py   writes a run for the viewer: labels, confidence, evidence, tracking
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
python scripts/cv.py fold --tag v7 --test J03WMX metrica1
python scripts/cv.py fold --tag v7 --test J03WN1 metrica2
python scripts/cv.py fold --tag v7 --test J03WOH metrica3
python scripts/cv.py fold --tag v7 --test J03WOY J03WPY
python scripts/cv.py fold --tag v7 --test J03WQQ J03WR9
python scripts/cv.py summary --tag v7                  # one decoding setting for all folds
python scripts/train_final.py --decode '<json printed by the summary>'
```

Every random draw is seeded from the match name, so the tables and the models can be rebuilt.
Retraining is not needed to reproduce the submission: the models that produced it are committed.

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
| `adversarial.py` | Whether a classifier can tell simulated from provided tracking, and on which features. |

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
- `vantage/_official_metrics.py` is the organisers' metric code, vendored unchanged under its MIT
  licence.
- This repository's own code is released under the MIT licence (see `LICENSE`).
