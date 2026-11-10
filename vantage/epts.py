"""Read FIFA EPTS tracking: a metadata XML plus a raw text file.

This is the format of the competition's automatic tracking (`*_atd.xml` + `*_atd.txt`) and of
Metrica's public Sample Game 3, so one reader serves both and is tested on the public one.

A raw line is

    <frame>:<slot>;<slot>;...:<ball>

where each slot is comma-separated channel values (x, y and sometimes speed) and a missing
observation is `NaN`. Which slot is which is *not* fixed by position alone: the metadata holds
one or more `DataFormatSpecification` blocks, each valid for a range of frames, listing the
channels in the order they appear. A file with substitutions has several blocks. This reader
follows those blocks instead of assuming a layout.

Coordinates are returned exactly as stored (normalised 0-1); converting to metres is the
caller's job, because the true pitch size is a per-match fact.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np


@dataclass
class Epts:
    frame: np.ndarray        # (T,) frame ids as written in the file
    xy: np.ndarray           # (T, P, 2) normalised positions, NaN where missing
    speed: np.ndarray        # (T, P) as stored, NaN where missing or not provided
    ball: np.ndarray         # (T, 2) normalised, NaN where missing
    team: np.ndarray         # (P,) 0 for the first team in the metadata (home), 1 for the second
    team_names: tuple        # (home, away)
    parameters: dict         # provider parameters, e.g. first_half_start
    speed_unit: str          # as declared for the speed channel, '' when absent


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def read(meta_path, raw_path):
    root = ET.parse(meta_path).getroot()
    for element in root.iter():
        element.tag = _local(element.tag)

    teams = [(team.get("id"), team.findtext("Name") or "") for team in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    home_id = teams[0][0]

    player_ids = [player.get("id") for player in root.iter("Player")]
    index_of = {player_id: index for index, player_id in enumerate(player_ids)}
    team = np.array([0 if player.get("teamId") == home_id else 1
                     for player in root.iter("Player")], dtype=np.int8)
    channels = {channel.get("id"): (channel.get("playerId"), channel.get("channelId"))
                for channel in root.iter("PlayerChannel")}
    parameters = {parameter.findtext("Name"): parameter.findtext("Value")
                  for parameter in root.iter("ProviderParameter")}

    speed_unit = ""
    for channel in root.iter("Channel"):
        name = (channel.findtext("Name") or channel.get("id") or "").strip().lower()
        if name in ("s", "speed"):
            speed_unit = (channel.findtext("Unit") or "").strip().lower()

    # One layout per frame range: for every slot, (player index, [channel names in order]).
    layouts = []
    for spec in root.iter("DataFormatSpecification"):
        slots = []
        for group in spec.iter("SplitRegister"):
            refs = [ref.get("playerChannelId") for ref in group.findall("PlayerChannelRef")]
            if not refs:
                continue
            owners = {channels[ref][0] for ref in refs}
            if len(owners) != 1:
                raise ValueError("a slot mixes players: {}".format(refs))
            slots.append((index_of[owners.pop()], [channels[ref][1] for ref in refs]))
        if not slots:
            raise ValueError("a DataFormatSpecification lists no player channels")
        layouts.append((int(spec.get("startFrame")), int(spec.get("endFrame")), slots))
    if not layouts:
        raise ValueError("{}: no DataFormatSpecification".format(meta_path))
    layouts.sort(key=lambda item: item[0])

    with open(raw_path) as handle:
        lines = handle.read().splitlines()
    t, p = len(lines), len(player_ids)
    frame = np.empty(t, dtype=np.int64)
    xy = np.full((t, p, 2), np.nan, dtype=np.float32)
    speed = np.full((t, p), np.nan, dtype=np.float32)
    ball = np.full((t, 2), np.nan, dtype=np.float32)

    cursor = 0
    for row, line in enumerate(lines):
        head, body, tail = line.split(":")
        frame_id = int(head)
        frame[row] = frame_id
        while cursor + 1 < len(layouts) and frame_id > layouts[cursor][1]:
            cursor += 1
        start, end, slots = layouts[cursor]
        if not start <= frame_id <= end:
            raise ValueError("frame {} is outside every DataFormatSpecification".format(frame_id))

        groups = body.split(";")
        if len(groups) != len(slots):
            raise ValueError("frame {}: {} slots in the line, {} in the metadata".format(
                frame_id, len(groups), len(slots)))
        for group, (player, names) in zip(groups, slots):
            if group[0] == "N":          # NaN: nothing observed for this slot on this frame
                continue
            values = group.split(",")
            for name, value in zip(names, values):
                if name == "x":
                    xy[row, player, 0] = float(value)
                elif name == "y":
                    xy[row, player, 1] = float(value)
                elif name in ("s", "speed"):
                    speed[row, player] = float(value)
        if tail and tail[0] != "N":
            values = tail.split(",")
            ball[row, 0] = float(values[0])
            ball[row, 1] = float(values[1])

    return Epts(frame=frame, xy=xy, speed=speed, ball=ball, team=team,
                team_names=(teams[0][1], teams[1][1]), parameters=parameters,
                speed_unit=speed_unit)


def read_team_names(meta_path):
    """(home, away) team names from the metadata, without reading the tracking."""
    root = ET.parse(meta_path).getroot()
    names = [team.findtext("{*}Name") or team.findtext("Name") or ""
             for team in root.iter() if _local(team.tag) == "Team"]
    if len(names) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(names)))
    return names[0], names[1]
