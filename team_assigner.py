"""
Assigns each detected player to Team A or Team B based on jersey color,
using KMeans clustering — same core idea as the tutorial video.
"""
import numpy as np
import cv2
from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}          # {1: rgb, 2: rgb}
        self.player_team_cache = {}    # {player_track_id: team_number}
        self._kmeans = None

    def _get_player_jersey_color(self, frame, bbox):
        """Crop the top half of a player's bounding box (torso/jersey area)
        and find its dominant color, ignoring the background/pitch."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(x1, 0), max(y1, 0)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        top_half = crop[0: int(crop.shape[0] / 2), :]
        if top_half.size == 0:
            return None

        image_2d = top_half.reshape(-1, 3)
        # 2 clusters: jersey color vs everything else (pitch/background sliver)
        try:
            km = KMeans(n_clusters=2, n_init=1, random_state=0)
            km.fit(image_2d)
        except Exception:
            return None

        labels = km.labels_.reshape(top_half.shape[0], top_half.shape[1])
        corner_labels = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        background_cluster = max(set(corner_labels), key=corner_labels.count)
        jersey_cluster = 1 - background_cluster
        return km.cluster_centers_[jersey_cluster]

    def assign_team_colors(self, frame, player_bboxes):
        """Call this once, on a frame with several clearly-visible players,
        to establish the two team colors."""
        colors = []
        for bbox in player_bboxes:
            color = self._get_player_jersey_color(frame, bbox)
            if color is not None:
                colors.append(color)

        if len(colors) < 2:
            return False

        km = KMeans(n_clusters=2, n_init=10, random_state=0)
        km.fit(colors)
        self._kmeans = km
        self.team_colors[1] = km.cluster_centers_[0]
        self.team_colors[2] = km.cluster_centers_[1]
        return True

    def get_player_team(self, frame, bbox, player_track_id):
        if player_track_id in self.player_team_cache:
            return self.player_team_cache[player_track_id]

        if self._kmeans is None:
            return None

        color = self._get_player_jersey_color(frame, bbox)
        if color is None:
            return None

        team_id = int(self._kmeans.predict([color])[0]) + 1
        self.player_team_cache[player_track_id] = team_id
        return team_id