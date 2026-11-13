"""How the recording's tracker loses and finds the ball, and whether the simulator does the same.

A label-free check. It prints totals only: rates, shares and percentiles over all the tracks and
gaps of a match, and over all the tagged set pieces of one kind, never a frame, a position or a
time. A track is an unbroken stretch of frames with a ball position; a gap is what lies between
two tracks.

For the recording, and for simulated recordings of the open matches under the simulator's
conditions, it prints

  * how much of the time the ball is tracked, how long the tracks are, and how many gaps there
    are of each length (the recording has none between 0.6 and 5 seconds);
  * for the gaps of each length: how far the ball is from where it was lost when it is found
    again, in which part of the pitch it is lost and found, and how fast the players move
    during the gap against the four seconds before it (a stoppage slows them, a ball lost in
    live play does not);
  * how near the nearest player is when a track begins and when it ends, and how fast the ball
    is over a track's first steps;
  * when a track begins relative to a tagged restart, and how the gaps that hold a tagged
    restart differ from the others. For the recording the restarts are the set pieces of Smart
    Tagging (a tag puts a restart 3.5 seconds into its clip, section 2 of the technical
    document); for the open matches they are the imitated tags every simulated recording
    carries (`external.tagged_restarts`), so the two are read the same way;
  * for the recording, by kind of set piece: where the ball is when the tracker picks it up
    after a set piece it did not follow. After a throw-in that is at the touchline with the
    ball moving in: the tracker finds the ball as the throw is taken, and it is the tag that
    comes early.

With --labels it adds the yardsticks the open matches' own labels give: where play stops and
restarts, how the players move through a true stoppage and through live play, and how often the
ball lies still next to a player for more than five seconds.

    python scripts/track_report.py --data-dir /path/to/data       # the recording and the simulator
    python scripts/track_report.py --conditions typical --labels  # the simulator and the yardsticks
"""

import argparse
import os
import sys

import numpy as np

from vantage import D, FPS, competition, epts, external, metrica_sample, tagging

LENGTHS = ((5, 10), (10, 30), (30, 1e9))
ZONES = ("touchline", "goal line", "box", "field")


def q(values, points=(25, 50, 75), form="{:.1f}"):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return "/".join(form.format(v) for v in np.percentile(values, points)) if len(values) else "n/a"


def pc(flags):
    flags = np.asarray(flags, dtype=np.float64)
    return "{:.0%}".format(float(np.mean(flags))) if len(flags) else "n/a"


def zone(point, pitch):
    """touchline or goal line (within 3 m of it, or beyond it), box (a penalty area), field."""
    length, width = pitch
    x, y = abs(point[0]), abs(point[1])
    if width / 2 - y < 3.0:
        return 0
    if length / 2 - x < 3.0:
        return 1
    if length / 2 - x < 16.5 and y < 20.16:
        return 2
    return 3


def zones(codes):
    codes = np.asarray(codes)
    return ", ".join("{} {:.0%}".format(name, np.mean(codes == index)) for index, name in enumerate(ZONES))


def nearest_player(ball, home, away):
    """Distance from the ball to the nearest player on every frame (inf where there is none)."""
    best = np.full(len(ball), np.inf)
    for team in (home, away):
        delta = team.astype(np.float64) - ball[:, None, :]
        distance = np.sqrt(np.sum(delta * delta, axis=-1))
        best = np.minimum(best, np.where(np.isfinite(distance), distance, np.inf).min(axis=1))
    return best


def player_speed(half, window=1):
    """Mean speed of all the players on each frame, m/s, from positions two fifths of a second
    apart. `window` averages the positions first, as a tracker that smooths its paths has."""
    total = np.zeros(half.n_frames)
    count = np.zeros(half.n_frames)
    for xy in (half.home_xy, half.away_xy):
        xy = xy.astype(np.float64)
        if window > 1:
            reach = window // 2
            ok = np.isfinite(xy)
            padded = np.concatenate((np.zeros((reach,) + xy.shape[1:]), np.where(ok, xy, 0.0),
                                     np.zeros((reach,) + xy.shape[1:])))
            weight = np.concatenate((np.zeros((reach,) + xy.shape[1:]), ok.astype(np.float64),
                                     np.zeros((reach,) + xy.shape[1:])))
            sums = np.cumsum(np.concatenate((np.zeros((1,) + xy.shape[1:]), padded)), axis=0)
            counts = np.cumsum(np.concatenate((np.zeros((1,) + xy.shape[1:]), weight)), axis=0)
            averaged = (sums[window:] - sums[:-window]) / np.maximum(counts[window:] - counts[:-window], 1.0)
            xy = np.where(ok, averaged, np.nan)
        step = np.full(xy.shape[:2], np.nan)
        moved = xy[10:] - xy[:-10]
        step[5:-5] = np.sqrt(np.sum(moved * moved, axis=2)) * (FPS / 10.0)
        ok = np.isfinite(step) & (step < 12.0)
        total += np.where(ok, step, 0.0).sum(axis=1)
        count += ok.sum(axis=1)
    out = np.full(half.n_frames, np.nan)
    good = count > 0
    out[good] = total[good] / count[good]
    return out


def slowed(speed, a, b):
    """The players' speed through [a, b) over their speed in the four seconds before it."""
    if a < 6 * FPS or b > len(speed) - 6 * FPS or b - a < 3 * FPS:
        return np.nan
    inside, before = speed[a + FPS:b - FPS], speed[a - 5 * FPS:a - FPS]
    if not np.isfinite(inside).any() or not np.isfinite(before).any():
        return np.nan
    return float(np.nanmean(inside) / np.nanmean(before))


def read(title, pairs):
    """pairs: [(half, [tagged restart frames in the half])]."""
    frames = seen_frames = lonely = 0
    gaps, tracks, start_near, end_near, start_speed = [], [], [], [], []
    tag_after, tag_through, tags = [], 0, 0
    shortest = None
    for half, tag_frames in pairs:
        ball = half.ball_xy.astype(np.float64)
        t = len(ball)
        length, width = half.pitch
        seen = np.isfinite(ball[:, 0])
        frames += t
        seen_frames += int(seen.sum())
        edges = np.diff(np.concatenate(([0], seen.astype(np.int8), [0])))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        near = nearest_player(np.where(seen[:, None], ball, 0.0), half.home_xy, half.away_xy)
        near[~seen] = np.nan
        speed = player_speed(half)
        moved = ball[1:] - ball[:-1]
        step = np.full(t, np.nan)
        step[1:] = np.sqrt(np.sum(moved * moved, axis=1)) * FPS
        for a, b in zip(starts, stops):
            tracks.append((b - a) / FPS)
            if b - a >= 30:
                start_near.append(near[a])
                end_near.append(near[b - 1])
                start_speed.append(np.nanmean(step[a + 1:a + 6]))
        gap_list = [(int(a), int(b)) for a, b in zip(stops[:-1], starts[1:])]
        owner = {}
        for frame in tag_frames:
            if frame < 2 * FPS + 10 or frame >= t - 2 * FPS - 10:
                continue
            tags += 1
            if seen[frame - 2 * FPS:frame + 2 * FPS].all():
                tag_through += 1
                continue
            around = starts[(starts >= frame - 10 * FPS) & (starts <= frame + 10 * FPS)]
            if len(around):
                tag_after.append(int(around[np.argmin(np.abs(around - frame))]) - frame)
            best = None
            for index, (a, b) in enumerate(gap_list):
                if a - 3 * FPS <= frame <= b + 3 * FPS:
                    distance = 0 if a <= frame <= b else min(abs(frame - a), abs(frame - b))
                    if best is None or distance < best[0]:
                        best = (distance, index)
            if best is not None:
                owner.setdefault(best[1], frame)
        for index, (a, b) in enumerate(gap_list):
            if b - a > 15:
                shortest = b - a if shortest is None else min(shortest, b - a)
            dead = float(np.mean(half.labels[a:b] == D)) if half.labels is not None else np.nan
            gaps.append(((b - a) / FPS, float(np.sqrt(np.sum((ball[b] - ball[a - 1]) ** 2))),
                         zone(ball[a - 1], half.pitch), zone(ball[b], half.pitch), slowed(speed, a, b),
                         1.0 if index in owner else 0.0, near[a - 1], near[b], dead))
        # strong changes of the ball's velocity with nobody near
        velocity = np.full(ball.shape, np.nan)
        velocity[2:-2] = (ball[4:] - ball[:-4]) * (FPS / 4.0)
        change = np.zeros(t)
        delta = velocity[24:] - velocity[:-24]
        change[12:-12] = np.nan_to_num(np.sqrt(np.sum(delta * delta, axis=1)))
        peak = change >= 8.0
        for offset in range(1, 13):
            peak[offset:] &= change[offset:] >= change[:-offset]
            peak[:-offset] &= change[:-offset] > change[offset:]
        with np.errstate(invalid="ignore"):
            lonely += int(np.sum(peak & (near > 3.0)))
    gaps = np.array(gaps, dtype=np.float64).reshape(-1, 9)
    minutes = frames / (60.0 * FPS)
    seconds = gaps[:, 0]

    def rate(low, high):
        return np.sum((seconds >= low) & (seconds < high)) / minutes

    print("== " + title)
    print("   ball tracked on {:.0%} of frames; {:.2f} tracks a minute, seconds p25/50/75 {}".format(
        seen_frames / frames, len(tracks) / minutes, q(tracks)))
    print("   gaps a minute by length: under 0.6 s {:.2f}; 0.6 to 5 s {:.2f}; 5 to 7 s {:.2f}; 7 to 10 s {:.2f}; "
          "10 to 15 s {:.2f}; 15 to 30 s {:.2f}; over 30 s {:.2f}. Shortest gap over 0.6 s: {}".format(
              rate(0, 0.6), rate(0.6, 5), rate(5, 7), rate(7, 10), rate(10, 15), rate(15, 30), rate(30, 1e9),
              "{} frames".format(shortest) if shortest is not None else "none"))
    for low, high in LENGTHS:
        pick = gaps[(seconds >= low) & (seconds < high)]
        if not len(pick):
            continue
        line = ("   gaps of {:.0f} to {} s ({}): the ball is found {} m from where it was lost; lost at {}; found at {}; "
                "the players' speed during the gap over the four seconds before {}, under 0.75 at {}").format(
                    low, "{:.0f}".format(high) if high < 1e8 else "...", len(pick), q(pick[:, 1]), zones(pick[:, 2]),
                    zones(pick[:, 3]), q(pick[:, 4], form="{:.2f}"), pc(pick[np.isfinite(pick[:, 4]), 4] < 0.75))
        if np.isfinite(pick[:, 8]).any():
            line += "; a stoppage for most of it (the labels): {}".format(pc(pick[:, 8] >= 0.5))
        print(line)
    print("   gaps under 0.6 s ({}): the ball is found {} m from where it was lost".format(
        int(np.sum(seconds < 0.6)), q(gaps[seconds < 0.6, 1])))
    print("   a track begins {} m from the nearest player (within 1.5 m {}), the ball at {} m/s over its first steps; "
          "it ends {} m from the nearest player (within 1.5 m {})".format(
              q(start_near, form="{:.2f}"), pc(np.array(start_near) < 1.5), q(start_speed, (50,)),
              q(end_near, form="{:.2f}"), pc(np.array(end_near) < 1.5)))
    longer = gaps[seconds >= 5]
    tagged, other = longer[longer[:, 5] > 0], longer[longer[:, 5] == 0]
    tag_after = np.array(tag_after, dtype=np.float64)
    print("   {} tagged restarts: ball tracked right through {:.0%}; otherwise the nearest beginning of a track is, in "
          "frames after the tag's restart, p10/25/50/75/90 {}, within a second at {}".format(
              tags, tag_through / max(tags, 1), q(tag_after, (10, 25, 50, 75, 90), "{:.0f}"),
              pc(np.abs(tag_after) <= FPS)))
    print("   gaps of 5 s or more with a tagged restart ({}) against the others ({}): the track before ends within "
          "1.5 m of a player {} against {}; the next begins within 1.5 m {} against {}; metres lost to found {} "
          "against {}".format(len(tagged), len(other), pc(tagged[:, 6] < 1.5), pc(other[:, 6] < 1.5),
                              pc(tagged[:, 7] < 1.5), pc(other[:, 7] < 1.5), q(tagged[:, 1]), q(other[:, 1])))
    print("   strong changes of the ball's velocity with nobody within 3 m: {:.1f} a tracked minute".format(
        lonely / max(seen_frames / (60.0 * FPS), 1e-9)))
    sys.stdout.flush()


def set_pieces(data_dir, halves):
    """By kind of set piece: where the ball is when the tracker picks it up again."""
    paths = competition.tactical_paths(data_dir)
    pieces = sorted((item["start"] + int(round(tagging.RESTART_LEAD * FPS)), item["tags"].get("Type", "?"))
                    for item in tagging.load(paths["tagging"]) if item["code"] == tagging.SET_PIECE)
    rows = {}
    for half in halves:
        ball = half.ball_xy.astype(np.float64)
        t = half.n_frames
        length, width = half.pitch
        seen = np.isfinite(ball[:, 0])
        starts = np.flatnonzero(seen[1:] & ~seen[:-1]) + 1
        near = nearest_player(np.where(seen[:, None], ball, 0.0), half.home_xy, half.away_xy)
        touch = width / 2 - np.abs(ball[:, 1])
        goal = length / 2 - np.abs(ball[:, 0])
        corner = np.sqrt(goal ** 2 + touch ** 2)
        margin = np.minimum(touch, goal)
        moved = ball[1:] - ball[:-1]
        step = np.full(t, np.nan)
        step[1:] = np.sqrt(np.sum(moved * moved, axis=1)) * FPS
        for frame, kind in pieces:
            frame -= half.first_frame
            if frame < 7 * FPS or frame >= t - 11 * FPS:
                continue
            row = rows.setdefault(kind, {key: [] for key in ("through", "after", "touch", "goal", "corner",
                                                             "speed", "inward", "player")})
            if seen[frame - 2 * FPS:frame + 2 * FPS].all():
                row["through"].append(1)
                continue
            row["through"].append(0)
            around = starts[(starts >= frame - 10 * FPS) & (starts <= frame + 10 * FPS)]
            if not len(around):
                continue
            start = int(around[np.argmin(np.abs(around - frame))])
            row["after"].append(start - frame)
            row["touch"].append(touch[start])
            row["goal"].append(goal[start])
            row["corner"].append(corner[start])
            row["speed"].append(np.nanmean(step[start + 1:start + 6]))
            row["inward"].append((margin[min(t - 1, start + 5)] - margin[start]) * FPS / 5.0)
            row["player"].append(near[start])
    print("== the recording, by kind of set piece: where the tracker picks the ball up after one it did not follow")
    pooled, names = None, []
    for kind in sorted(rows, key=lambda k: -len(rows[k]["through"])):
        row = rows[kind]
        if len(row["after"]) < 8:           # too few to print apart: no single set piece is shown
            pooled = row if pooled is None else {key: pooled[key] + row[key] for key in row}
            names.append(kind)
            continue
        say_kind(kind, row)
    if pooled is not None:
        say_kind("the other kinds together (" + ", ".join(names) + ")", pooled)


def say_kind(kind, row):
    touch, goal, corner = np.array(row["touch"]), np.array(row["goal"]), np.array(row["corner"])
    print("   {}: {} tags, tracked right through {}; at the other {} the track begins {} frames after the tag's "
          "restart (p25/50/75), {} m inside the nearest touchline (less than 2 m inside it, or beyond it: {}), {} m "
          "inside the nearest goal line (less than 2 m inside it, or beyond it: {}), within 3 m of a corner {}; the ball "
          "at {} m/s, moving in from the nearest line at {} m/s; nearest player {} m".format(
              kind, len(row["through"]), pc(row["through"]), len(row["after"]), q(row["after"], form="{:.0f}"),
              q(touch), pc(touch < 2.0), q(goal), pc(goal < 2.0), pc(corner < 3.0),
              q(row["speed"]), q(row["inward"]), q(row["player"])))


def yardsticks():
    """What the open matches' own labels say about stoppages, as published."""
    rng = np.random.default_rng(0)
    stopped, live, holds, minutes = [], [], [], 0.0
    for match in external.ALL_MATCHES:
        for half in external.labelled_halves(match):
            ball = half.ball_xy.astype(np.float64)
            t = len(ball)
            minutes += t / (60.0 * FPS)
            dead = half.labels == D
            speed = player_speed(half, 21)
            edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
            lengths = []
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                if b - a < 5 * FPS or a == 0 or b >= t:
                    continue
                lengths.append(int(b - a))
                if np.isfinite(ball[a - 1, 0]) and np.isfinite(ball[b, 0]):
                    stopped.append(((b - a) / FPS, float(np.sqrt(np.sum((ball[b] - ball[a - 1]) ** 2))),
                                    zone(ball[a - 1], half.pitch), zone(ball[b], half.pitch), slowed(speed, int(a), int(b))))
            for length in lengths:              # live play of the same lengths, placed at random
                for _ in range(20):
                    a = int(rng.integers(6 * FPS, t - length - 6 * FPS))
                    if not dead[a - 5 * FPS:a + length + 5 * FPS].any():
                        live.append((length / FPS, slowed(speed, a, a + length)))
                        break
            # the ball lying still within a player's reach for 5.36 s or more
            source = half
            if external.source_of(match) == "metrica":
                source = metrica_sample.with_dead_ball_track(half, np.random.default_rng(1))
                ball = source.ball_xy.astype(np.float64)
            near = nearest_player(np.where(np.isfinite(ball), ball, 1e6), source.home_xy, source.away_xy) < 1.5
            gone = np.full(t, np.inf)
            moved = ball[5:] - ball[:-5]
            gone[2:-3] = np.nan_to_num(np.sqrt(np.sum(moved * moved, axis=1)) * (FPS / 5.0), nan=np.inf)
            still = near & (gone < 2.5)
            edges = np.diff(np.concatenate(([0], still.astype(np.int8), [0])))
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                if b - a >= 134:
                    holds.append(float(np.mean(dead[a:b])) >= 0.5)
    stopped = np.array(stopped)
    live = np.array(live)
    print("== the open matches as published, their own labels ({:.0f} minutes)".format(minutes))
    for low, high in LENGTHS:
        pick = stopped[(stopped[:, 0] >= low) & (stopped[:, 0] < high)]
        other = live[(live[:, 0] >= low) & (live[:, 0] < high)]
        print("   stoppages of {:.0f} to {} s ({}): play restarts {} m from where it stopped; it stops at {}; restarts at "
              "{}; the players' speed during it over the four seconds before {}, under 0.75 at {} (through live play of "
              "the same lengths: {}, {})".format(
                  low, "{:.0f}".format(high) if high < 1e8 else "...", len(pick), q(pick[:, 1]), zones(pick[:, 2]),
                  zones(pick[:, 3]), q(pick[:, 4], form="{:.2f}"), pc(pick[np.isfinite(pick[:, 4]), 4] < 0.75),
                  q(other[:, 1], form="{:.2f}"), pc(other[np.isfinite(other[:, 1]), 1] < 0.75)))
    holds = np.array(holds)
    print("   the ball lies still (under 2.5 m/s) within 1.5 m of a player for 5.36 s or more: {:.2f} times a minute "
          "inside stoppages, {:.2f} in live play".format(holds.sum() / minutes, (~holds).sum() / minutes))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", help="the competition data; without it only the simulator is read")
    parser.add_argument("--generation", default=None, help="a simulator generation other than the one in use")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="the simulator's conditions to read (default: all six; none: --conditions with nothing after it)")
    parser.add_argument("--labels", action="store_true", help="add the yardsticks from the open matches' labels")
    args = parser.parse_args()
    if args.data_dir:
        halves = competition.load_halves(args.data_dir)
        paths = competition.tactical_paths(args.data_dir)
        home_name, away_name = epts.read_team_names(paths["meta"])
        tagged = tagging.restarts(tagging.load(paths["tagging"]), home_name, away_name)
        read("the recording", [(half, [frame - half.first_frame for frame, _ in tagged
                                        if 0 <= frame - half.first_frame < half.n_frames]) for half in halves])
        set_pieces(args.data_dir, halves)
    if args.generation:
        os.environ["VANTAGE_SIM"] = args.generation
    wanted = [name for name in external.CONDITIONS if name != "clean"] if args.conditions is None else args.conditions
    if wanted:
        clean = {match: external.labelled_halves(match) for match in external.ALL_MATCHES}
    for condition in wanted:
        for source, said in (("dfl", "the seven DFL matches"), ("metrica", "Metrica's three sample games")):
            recorded = [external.condition_half(half, match, condition)
                        for match in external.ALL_MATCHES if external.source_of(match) == source
                        for half in clean[match]]
            read("simulated ({}), {}, {}".format(external.generation(), condition, said),
                 [(half, [frame for frame, _ in half.meta.get("restarts", ())]) for half in recorded])
    if args.labels:
        yardsticks()


if __name__ == "__main__":
    main()
