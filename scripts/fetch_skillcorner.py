"""Download SkillCorner's open data: A-League matches with tracking from the broadcast picture.

    SkillCorner Open Data, https://github.com/SkillCorner/opendata  (MIT licence)

Twenty matches of the 2024/2025 Australian A-League. For each: the match sheet, tracking at ten
frames a second made automatically from the television picture (players who are off screen are
extrapolated and marked as such), a per-frame flag of who is in possession, the list of on-ball
and off-ball events, and the phases of play, which exist only while the ball is in play.

    python scripts/fetch_skillcorner.py                 # into data/external/skillcorner
    python scripts/fetch_skillcorner.py --matches 3     # the first three only

The tracking files are stored with Git LFS. Each is fetched from GitHub's media address and
checked against the size and SHA-256 that the repository's pointer file gives, so a truncated
download fails loudly. Nothing here touches the competition match.
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request

REPO = "SkillCorner/opendata"
BRANCH = "master"
RAW = "https://raw.githubusercontent.com/{}/{}/".format(REPO, BRANCH)
MEDIA = "https://media.githubusercontent.com/media/{}/{}/".format(REPO, BRANCH)
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(os.path.dirname(HERE), "data", "external", "skillcorner")
CHUNK = 1 << 20
SMALL = ("{id}_match.json", "{id}_dynamic_events.csv", "{id}_phases_of_play.csv")
TRACKING = "{id}_tracking_extrapolated.jsonl"


def get(url, timeout=120):
    request = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    return urllib.request.urlopen(request, timeout=timeout)


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def save(url, target):
    """Download to `target` through a .part file. Returns the number of bytes."""
    os.makedirs(os.path.dirname(target), exist_ok=True)
    partial = target + ".part"
    count = 0
    with get(url) as response, open(partial, "wb") as handle:
        for block in iter(lambda: response.read(CHUNK), b""):
            handle.write(block)
            count += len(block)
    os.replace(partial, target)
    return count


def small(path, out_dir):
    target = os.path.join(out_dir, *path.split("/"))
    if os.path.isfile(target) and os.path.getsize(target) > 0:
        return "kept", os.path.getsize(target)
    return "downloaded", save(RAW + path, target)


def tracking(path, out_dir):
    """A file kept in Git LFS: the pointer says what it must be, the media address serves it."""
    target = os.path.join(out_dir, *path.split("/"))
    with get(RAW + path) as response:
        pointer = response.read(4096).decode("utf-8", "replace")
    fields = dict(line.split(" ", 1) for line in pointer.strip().splitlines() if " " in line)
    if not fields.get("oid", "").startswith("sha256:"):
        raise IOError("{}: not a Git LFS pointer".format(path))
    expected, size = fields["oid"][7:], int(fields["size"])
    if os.path.isfile(target) and os.path.getsize(target) == size and sha256_of(target) == expected:
        return "kept", size
    count = save(MEDIA + path, target)
    actual = sha256_of(target)
    if count != size or actual != expected:
        os.remove(target)
        raise IOError("{}: got {} bytes with sha256 {}, the repository says {} and {}".format(
            path, count, actual, size, expected))
    return "downloaded", size


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--matches", type=int, default=None, help="only the first so many matches")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    total = 0
    for path in ("README.md", "LICENSE", "data/matches.json"):
        how, size = small(path, args.out)
        total += size
        print("  {:<10} {:>8.2f} MB  {}".format(how, size / 1e6, path), flush=True)
    with open(os.path.join(args.out, "data", "matches.json"), encoding="utf-8") as handle:
        matches = json.load(handle)
    ids = [int(match["id"]) for match in matches]
    if args.matches is not None:
        ids = ids[:args.matches]
    print("{} matches -> {}".format(len(ids), args.out), flush=True)
    for match_id in ids:
        folder = "data/matches/{}/".format(match_id)
        for name in SMALL:
            how, size = small(folder + name.format(id=match_id), args.out)
            total += size
        how, size = tracking(folder + TRACKING.format(id=match_id), args.out)
        total += size
        print("  {:<10} {:>8.1f} MB  {}".format(how, size / 1e6, TRACKING.format(id=match_id)), flush=True)
    print("done: {:.2f} GB in all".format(total / 1e9))


if __name__ == "__main__":
    sys.exit(main())
