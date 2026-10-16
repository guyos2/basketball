"""Split tracked players into two teams by jersey color."""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans

# Lightness depends on shadows and gym lighting more than on the jersey, so it counts for less.
LAB_WEIGHTS = np.array([0.4, 1.0, 1.0])


def assign_teams(track_colors: dict[int, list[np.ndarray]]) -> dict[int, int]:
    """Return team 0 or 1 for every track id that has at least one color sample."""
    ids = [t for t, colors in track_colors.items() if colors]
    if len(ids) < 2:
        return {t: 0 for t in ids}
    features = np.array([np.median(np.stack(track_colors[t]), axis=0) for t in ids]) * LAB_WEIGHTS
    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(features)
    return {t: int(label) for t, label in zip(ids, labels)}


def team_display_colors(track_colors: dict[int, list[np.ndarray]], teams: dict[int, int]) -> dict[int, tuple]:
    """BGR jersey color of each team, for drawing."""
    out = {}
    for team in set(teams.values()):
        samples = [c for t, colors in track_colors.items() if teams.get(t) == team for c in colors]
        if samples:
            lab = np.clip(np.median(np.stack(samples), axis=0), 0, 255).astype(np.uint8).reshape(1, 1, 3)
            out[team] = tuple(int(v) for v in cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0])
    return out
