"""Who has the ball, and when it was passed or shot."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .models import Event, FrameState, Point


@dataclass
class EventConfig:
    bbox_expand: float = 0.25  # the ball must be inside the player's box widened by this fraction
    hand_reach: float = 0.55  # ...and within this fraction of the player's height from the hands (covers dribbling)
    min_hold_frames: int = 3  # shorter possessions are noise (ball passing in front of a player)
    max_gap_frames: int = 8  # ball briefly hidden while the same player still has it
    max_flight_s: float = 2.0  # longest time between release and reaching the rim or a teammate


def ball_holder(state: FrameState, cfg: EventConfig) -> int | None:
    if state.ball is None:
        return None
    bx, by = state.ball
    best, best_d = None, math.inf
    for p in state.players.values():
        x1, y1, x2, y2 = p.bbox
        pad = cfg.bbox_expand * (x2 - x1)
        if not (x1 - pad <= bx <= x2 + pad and y1 - pad <= by <= y2):
            continue
        hands = ((x1 + x2) / 2, y1 + 0.45 * (y2 - y1))
        d = math.dist(hands, (bx, by))
        if d <= cfg.hand_reach * (y2 - y1) and d < best_d:
            best, best_d = p.track_id, d
    return best


def smooth_possession(raw: list[int | None], cfg: EventConfig) -> list[int | None]:
    held = list(raw)
    # Fill short gaps where the same player has the ball on both sides.
    last, last_i = None, -1
    for i, h in enumerate(held):
        if h is None:
            continue
        if h == last and 0 < i - last_i - 1 <= cfg.max_gap_frames:
            held[last_i + 1:i] = [h] * (i - last_i - 1)
        last, last_i = h, i
    # Drop runs that are too short to be a real possession.
    i = 0
    while i < len(held):
        j = i
        while j < len(held) and held[j] == held[i]:
            j += 1
        if held[i] is not None and j - i < cfg.min_hold_frames:
            held[i:j] = [None] * (j - i)
        i = j
    return held


def possession_runs(held: list[int | None]) -> list[tuple[int, int, int]]:
    """(holder, first frame, last frame) for every continuous possession."""
    runs = []
    for i, h in enumerate(held):
        if h is None:
            continue
        if runs and runs[-1][0] == h and runs[-1][2] == i - 1:
            runs[-1] = (h, runs[-1][1], i)
        else:
            runs.append((h, i, i))
    return runs


def detect_events(states: list[FrameState], teams: dict[int, int], rim_px: Point | None,
                  rim_radius_px: float, fps: float, cfg: EventConfig | None = None) -> list[Event]:
    cfg = cfg or EventConfig()
    held = smooth_possession([ball_holder(s, cfg) for s in states], cfg)
    for s, h in zip(states, held):
        s.holder = h
    runs = possession_runs(held)
    max_flight = int(cfg.max_flight_s * fps)
    events = []
    for k, (holder, _, end) in enumerate(runs):
        nxt = runs[k + 1] if k + 1 < len(runs) else None
        flight_end = min(nxt[1] if nxt else len(states), end + max_flight + 1)
        reached_rim = rim_px is not None and any(
            states[i].ball is not None and math.dist(states[i].ball, rim_px) <= rim_radius_px
            for i in range(max(end - 2, 0), flight_end)
        )
        if reached_rim:
            events.append(Event("shot", end, states[end].time, holder))
        elif nxt and nxt[0] != holder and nxt[1] - end <= max_flight and teams.get(nxt[0]) == teams.get(holder):
            events.append(Event("pass", end, states[end].time, holder, target=nxt[0]))
    return events


def load_manual_events(path: str | Path, states: list[FrameState]) -> list[Event]:
    """Events typed by hand: [{"time": 4.2, "type": "shot"|"pass", "player": 7, "target": 11}].

    "player" may be left out when the ball holder is detected at that moment.
    """
    events = []
    for item in json.loads(Path(path).read_text()):
        kind = item["type"]
        if kind not in ("shot", "pass"):
            raise ValueError(f"Event type must be 'shot' or 'pass', got {kind!r}")
        frame = min(range(len(states)), key=lambda i: abs(states[i].time - item["time"]))
        player = item.get("player")
        if player is None:
            nearby = [states[i].holder for i in range(max(frame - 3, 0), min(frame + 4, len(states)))]
            player = next((h for h in nearby if h is not None), None)
            if player is None:
                raise ValueError(f"No ball holder found at {item['time']}s; add \"player\" to the event")
        if player not in states[frame].players:
            raise ValueError(f"Player {player} is not on the court map at {item['time']}s")
        events.append(Event(kind, frame, states[frame].time, int(player), item.get("target"), source="manual"))
    return events
