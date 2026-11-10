"""Download the external (other-match) data this project trains and validates on.

Nothing here touches the competition match. The only source is the open DFL dataset:

    Bassek, Rein, Weber, Memmert (2025). An integrated dataset of spatiotemporal and event
    data in elite soccer. Scientific Data. doi:10.6084/m9.figshare.28196177.v1  (CC BY 4.0)

Seven matches, each with official 25 Hz position data that carries a per-frame ball status
(alive / dead) and ball possession (home / away) flag. Those two flags are the labels.

    python scripts/fetch_external.py                 # into data/external/dfl
    python scripts/fetch_external.py --out D:/dfl

Every file is checked against the MD5 that figshare publishes, so a truncated download fails
loudly instead of training on half a match.
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request

ARTICLE_API = "https://api.figshare.com/v2/articles/28196177"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(os.path.dirname(HERE), "data", "external", "dfl")
CHUNK = 1 << 20
# Event files are not used: the labels come from the position data itself.
WANTED_PREFIXES = ("DFL_02_01_matchinformation", "DFL_04_03_positions_raw_observed")


def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def listing():
    request = urllib.request.Request(ARTICLE_API, headers={"User-Agent": "excensia-vantage"})
    with urllib.request.urlopen(request, timeout=60) as response:
        article = json.load(response)
    return [entry for entry in article["files"] if entry["name"].startswith(WANTED_PREFIXES)]


def fetch(entry, out_dir):
    target = os.path.join(out_dir, entry["name"])
    expected = entry.get("computed_md5") or entry.get("supplied_md5")
    if os.path.isfile(target) and md5_of(target) == expected:
        return target, "kept"

    partial = target + ".part"
    request = urllib.request.Request(entry["download_url"],
                                     headers={"User-Agent": "excensia-vantage"})
    with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as handle:
        for block in iter(lambda: response.read(CHUNK), b""):
            handle.write(block)

    actual = md5_of(partial)
    if actual != expected:
        os.remove(partial)
        raise IOError("{}: md5 {} != published {}".format(entry["name"], actual, expected))
    os.replace(partial, target)
    return target, "downloaded"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    entries = sorted(listing(), key=lambda entry: entry["size"])
    total = sum(entry["size"] for entry in entries)
    print("{} files, {:.2f} GB -> {}".format(len(entries), total / 1e9, args.out))
    for entry in entries:
        _, how = fetch(entry, args.out)
        print("  {:<10} {:>8.1f} MB  {}".format(how, entry["size"] / 1e6, entry["name"]))
        sys.stdout.flush()
    print("done")


if __name__ == "__main__":
    main()
