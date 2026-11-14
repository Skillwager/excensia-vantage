"""Training tables drawn with a third of the simulator's rate of live losses. External matches only.

    python scripts/third_loss_tables.py                    # build the tables, under the tag g8
    python scripts/third_loss_tables.py --report g7 g8     # the share of each tag's rows with no ball
    python scripts/cv_variants.py --spec scripts/specs/third_loss.json --data-tag g8 \
        --test J03WMX metrica1                             # ... and the other four folds
    python scripts/cv_variants.py --spec scripts/specs/third_loss_mixed.json --data-tag g8 \
        --conditions mixed --test J03WMX metrica1          # ... and the other four folds
    python scripts/compare_cv.py --base S7all_g7 --other R8_g7                      # each at its own setting
    python scripts/compare_cv.py --base S7all_g7 --other R8_g7 --setting '<json>'   # both at one
    python scripts/condition_compare.py --pairs S7all_g7:S7allM_g7 R8_g7:R8M_g7
    python scripts/condition_compare.py --pairs S7all_g7:S7allM_g7 R8_g7:R8M_g7 --setting '<json>'

    python scripts/cv.py summary --tag R8_g7 --sources metrica     # by condition, and the setting
    python scripts/train_final.py --data-tag g8 --decode '<json printed by the summary>' \
        --out runs/models/third_loss/vantage                       # not into models/
    python scripts/skillcorner_check.py                            # the submitted models: sc_shipped
    python scripts/skillcorner_check.py --model runs/models/third_loss/vantage.json --tag sc_third
    python scripts/skillcorner_compare.py --base sc_shipped --other sc_third \
        --other-model runs/models/third_loss/vantage.json          # each at its own setting
    python scripts/skillcorner_compare.py --base sc_shipped --other sc_third   # both at the submitted one
    python scripts/skillcorner_compare.py --base sc_shipped --other sc_third \
        --base-model runs/models/third_loss/vantage.json \
        --other-model runs/models/third_loss/vantage.json          # both at the new fit's

For every training recording the simulator draws how often the tracker loses the ball in live
play (`live_loss_rate`, drawn in vantage/degrade.py). This builds the tables `cv.py prepare`
builds, from the same seeds and the same draw of every recording's settings, except that this
one drawn number is divided by three. The recordings then have no tracked ball on about half as
many of their live frames, not a third (--report): the ball is also lost in ways that rate does
not govern. The submitted models' recipe fitted on these tables is the last line of "what was
tried and not adopted" in section 6 of the technical document. It asks whether the submitted
models were trained on recordings that lose the ball too often. The commands above make that
line and the figures quoted with it: the tables, the five folds under the six conditions and
under the seventh, the two fits side by side at their own settings and at one setting for both,
and models fitted on all ten matches for SkillCorner's recordings, read the same three ways. The
specs name no tables; it is `--data-tag g8` that puts the fit on these.

The package is not changed. The function that draws a recording's settings is wrapped here, for
this process only, so nothing inside a table says which draw made it: these tables carry the
same generation and the same code stamp as the tables `cv.py prepare` makes. That is why they
are kept under a tag of their own. The tags of the simulator's generations are refused, so is a
folder that already holds tables unless this script began it, and the folder gets a note that
says what is in it.

--report reads tables that are already there: for each simulated recording, the share of its
dead-ball rows and of its live rows that have no tracked ball. No competition data is read.
"""

import argparse
import glob
import os
import time
from dataclasses import replace

import numpy as np

from vantage import D, external
from vantage.external import ALL_MATCHES

PART = 3.0
NOTE = "THIRD_OF_THE_LIVE_LOSSES.txt"


def a_third_of_the_rate(drawn):
    """The simulator's draw of a recording's settings, with the rate of live losses divided."""
    def setting(rng):
        config = drawn(rng)
        return replace(config, live_loss_rate=config.live_loss_rate / PART)
    return setting


def build(args):
    if args.data_tag in external.GENERATIONS:
        raise SystemExit("{} is the tag of a simulator's own tables: give these another one".format(
            args.data_tag))
    folder = os.path.join(external.CACHE, "train_" + args.data_tag)
    note = os.path.join(folder, NOTE)
    if glob.glob(os.path.join(folder, "*.npz")) and not os.path.isfile(note):
        raise SystemExit("{} already holds tables that this script did not begin: give these "
                         "another tag".format(folder))
    os.makedirs(folder, exist_ok=True)
    with open(note, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("The tables in this folder were built by scripts/third_loss_tables.py: the simulator's\n"
                     "drawn rate of live losses divided by {:g}. Nothing inside a table says so.\n".format(PART))
    # prepare_match looks the name up in its module each time it draws
    external.sampled = a_third_of_the_rate(external.sampled)
    for match in (args.matches or ALL_MATCHES):
        started = time.time()
        external.prepare_match(match, args.data_tag, args.variants, args.stride)
        print("{} prepared in {:.0f}s".format(match, time.time() - started), flush=True)


def report(tags):
    for tag in tags:
        dead_share, live_share = [], []
        for path in sorted(glob.glob(os.path.join(external.CACHE, "train_" + tag, "*_own.npz"))):
            with np.load(path) as data:
                names = [str(name) for name in data["names"]]
                seen = data["table"][:, names.index("seen")] > 0.5
                dead = data["target"] == D
            dead_share.append((dead & ~seen).sum() / max(1, dead.sum()))
            live_share.append((~dead & ~seen).sum() / max(1, (~dead).sum()))
        if not dead_share:
            raise SystemExit("no tables under the tag {}".format(tag))
        print("{}: {} simulated recordings".format(tag, len(dead_share)))
        for name, share in (("dead-ball rows without a tracked ball", dead_share),
                            ("live rows without a tracked ball", live_share)):
            low, middle, high = np.percentile(share, [10, 50, 90])
            print("  {:<38} median {:.2f}   a tenth under {:.2f}, a tenth over {:.2f}".format(
                name, middle, low, high))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-tag", default="g8")
    parser.add_argument("--variants", type=int, default=6)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--matches", nargs="*", default=None)
    parser.add_argument("--report", nargs="+", metavar="TAG", default=None,
                        help="print how often the recordings of these tags lose the ball, and build nothing")
    args = parser.parse_args()
    if args.report:
        report(args.report)
    else:
        build(args)


if __name__ == "__main__":
    main()
