# Excensia Vantage

Frame-level ball possession from automatic football tracking. Entry of team **Excensia** for
*More than a Hack 2027, Phase 1* (Metrica Sports, FC Barcelona, Mobile World Capital Barcelona).

For every frame of a match Vantage says whether the home team has the ball (`H`), the away team has
it (`A`) or the ball is dead (`D`). It reads the provided automatic tracking and Smart Tagging and
nothing else, and everything it has learned comes from **other** matches.

- [docs/TECHNICAL_DOCUMENT.md](docs/TECHNICAL_DOCUMENT.md): the method, the data, the limits.
- `viewer/`: a browser viewer for the result (possession timeline, team shape, evidence).

## Reproduce the submitted file

You need the organisers' data bundle (the folder that contains `tactical/`). It is not in this
repository and must not be added to it.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt            # Windows: .venv\Scripts\pip
.venv/bin/pip install --no-deps -e .
.venv/bin/python -m vantage run --data-dir /path/to/data --out excensia.csv
```

Python 3.12 (`.python-version` has the exact one used). The three pinned packages in
`requirements.txt` are the whole runtime: numpy, scipy and lightgbm. The run takes about two
minutes on a laptop, uses no GPU and no network, and is deterministic: the trained model is a
fixed file in `models/`, and its prediction is a sum of constants chosen by threshold tests.

The command prints the SHA-256 of the file it wrote and stores it in `excensia.receipt.json`
together with the hashes of the inputs, the model and the package versions. The submitted file is

```
sha256  <filled in at submission>
```

Check any file before sending it, with the organisers' own validator:

```bash
python /path/to/more-than-a-hack-2027/evaluation/validate_submission.py excensia.csv
```

## What is in here

```
vantage/
  epts.py            reader for the FIFA EPTS tracking format (follows the layout blocks)
  competition.py     the match: published frame ranges, loading, the submission writer
  features.py        per-frame feature table: ball, gaps in the ball, who is near it, team shape
  model.py           the frame classifier (gradient-boosted trees), scored in both mirror views
  decode.py          Viterbi decoding into whole possessions, with minimum lengths
  tagging.py         Smart Tagging as bounded votes
  degrade.py         simulator that makes corrected tracking look like automatic tracking
  labels.py          brings other providers' possession flags to the challenge's definition
  dfl.py, metrica_sample.py, external.py     the ten external matches
  evaluate.py, _official_metrics.py          scoring with the organisers' own metric code
models/              the trained model and its card (features, decoding settings, provenance)
scripts/             fetch the external data, cross-validate, train, compare domains
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
python scripts/cv.py fold --tag v3 --test J03WMX metrica1    # ...one command per fold
python scripts/cv_summary.py --tag v3                  # picks the decoding settings
python scripts/train_final.py --decode '<json printed by cv_summary>'
```

Every random draw is seeded from the match name, so the tables and the model can be rebuilt.
Retraining is not needed to reproduce the submission: the model that produced it is committed.

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
