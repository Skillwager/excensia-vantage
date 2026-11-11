"""A first look at one SkillCorner match: what its files hold and how they fit together.

    python scripts/skillcorner_look.py --match 2017461

Open data, other matches only. Prints counts and a few sample rows.
"""

import argparse
import collections
import csv
import json
import os

from vantage.external import ROOT

FOLDER = os.path.join(ROOT, "data", "external", "skillcorner", "data", "matches")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match", required=True)
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()
    stem = os.path.join(FOLDER, args.match, args.match)

    with open(stem + "_match.json", encoding="utf-8") as handle:
        match = json.load(handle)
    print("match.json keys:", sorted(match.keys()))
    for key in ("home_team", "away_team", "pitch_length", "pitch_width", "date_time", "match_periods",
                "home_team_side", "status", "ball", "home_team_kit", "away_team_kit"):
        if key in match:
            print("  {}: {}".format(key, json.dumps(match[key])[:300]))
    players = match.get("players", [])
    print("  players:", len(players), "first:", json.dumps(players[0])[:600] if players else None)

    for name in ("_phases_of_play.csv", "_dynamic_events.csv"):
        with open(stem + name, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        print("\n{}: {} rows, {} columns".format(name, len(rows), len(rows[0])))
        print("  columns:", list(rows[0].keys()))
        for row in rows[:3]:
            print("  ", {key: value for key, value in row.items() if value not in ("", None)} if name.startswith("_phases")
                  else {key: row[key] for key in list(row)[:40]})
        for column in ("event_type", "event_subtype", "team_in_possession_phase_type", "team_out_of_possession_phase_type",
                       "period", "start_type", "end_type", "game_interruption_before", "game_interruption_after",
                       "team_possession_loss_in_phase", "game_state"):
            if column in rows[0]:
                print("  {}: {}".format(column, dict(collections.Counter(row[column] for row in rows).most_common(14))))

    path = stem + "_tracking_extrapolated.jsonl"
    if os.path.isfile(path):
        groups = collections.Counter()
        ball_seen = total = 0
        detected = players_total = 0
        periods = collections.Counter()
        shown = 0
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                frame = json.loads(line)
                total += 1
                periods[frame.get("period")] += 1
                possession = frame.get("possession") or {}
                groups[possession.get("group")] += 1
                ball = frame.get("ball_data") or {}
                ball_seen += bool(ball.get("is_detected"))
                for player in frame.get("player_data") or []:
                    players_total += 1
                    detected += bool(player.get("is_detected"))
                if shown < args.frames and frame.get("player_data"):
                    shown += 1
                    brief = dict(frame, player_data=frame["player_data"][:2])
                    print("\nframe:", json.dumps(brief)[:900])
        print("\ntracking: {} frames; by period {}".format(total, dict(periods)))
        print("  possession group: {}".format(dict(groups)))
        print("  ball detected on {:.1%} of frames".format(ball_seen / max(1, total)))
        print("  player rows {} ({:.1f} a frame), detected {:.1%}".format(
            players_total, players_total / max(1, total), detected / max(1, players_total)))
    else:
        print("\ntracking file not here yet")


if __name__ == "__main__":
    main()
