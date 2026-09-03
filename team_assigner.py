"""
Assigns each detected player to Team A or Team B based on bib/jersey color,
using hue-based clustering (robust to shadows/lighting, unlike raw RGB).
"""
from collections import deque, Counter
import numpy as np
import cv2
from sklearn.cluster import KMeans

VOTES_PER_PLAYER = 10    # how many recent classifications to remember per player
MIN_VOTES_TO_DECIDE = 4  # don't commit to a team color until we've seen this many confident samples

MIN_UNIQUE_PLAYERS_TO_INIT = 8   # prefer to see this many distinct players before deciding team colors
FALLBACK_FRAME_LIMIT = 300       # ...but don't wait forever — finalize with whatever we have by this frame

MIN_HUE_CONCENTRATION = 0.55  # 0-1: how "pure"/consistent the sampled color needs to be. Low = mixed/ambiguous colors (not a clean bib), reject.
MAX_HUE_DIST_FOR_TEAM = 0.4   # max distance (on unit circle, 0-2 scale) to count as a confident match to a team's hue — tightened from 0.55 to reject more borderline/coincidental color matches (e.g. a spectator's similar-ish clothing)


def _hue_to_unit_vector(hue_channel):
    """Convert an array of OpenCV hue values (0-179) into unit vectors on a
    circle. This makes 'how similar are these colors' a proper circular
    distance instead of naive subtraction, which breaks near the red/wrap
    boundary and also lets us average colors correctly."""
    theta = hue_channel.astype(np.float32) * (np.pi / 90.0)  # hue*2 -> degrees, -> radians
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1)


class TeamAssigner:
    def __init__(self):
        self.team_hue_vectors = {}     # {1: unit_vector, 2: unit_vector} — for reference/debugging
        self.player_team_votes = {}    # {player_track_id: deque of recent team guesses}
        self._color_pool = {}          # {player_track_id: unit_vector}
        self._kmeans = None

    def _get_player_hue_vector(self, frame, bbox):
        """Sample the torso/bib region of a player's bounding box and return
        a single representative hue (as a unit vector), filtering out pitch
        green, skin, and shadow — or None if the crop doesn't have a clean,
        consistent color (e.g. not actually a bib)."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(x1, 0), max(y1, 0)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        h, w = crop.shape[:2]
        ty1, ty2 = int(h * 0.15), int(h * 0.55)
        tx1, tx2 = int(w * 0.2), int(w * 0.8)
        torso = crop[ty1:ty2, tx1:tx2]
        if torso.size == 0:
            torso = crop

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Exclude pitch green, low-saturation (skin/shadow/white lines/grey),
        # and very dark or very bright/blown-out pixels.
        is_green = (hue >= 35) & (hue <= 95)
        is_low_sat = sat < 45
        is_bad_val = (val < 30) | (val > 250)
        keep_mask = ~(is_green | is_low_sat | is_bad_val)

        kept_hues = hue[keep_mask]
        if kept_hues.shape[0] < 20:
            return None  # not enough distinctive pixels — likely not wearing a bib

        vectors = _hue_to_unit_vector(kept_hues)
        mean_vec = vectors.mean(axis=0)
        concentration = np.linalg.norm(mean_vec)  # 1.0 = all pixels same hue, 0 = totally mixed/random

        if concentration < MIN_HUE_CONCENTRATION:
            return None  # colors too mixed/inconsistent to trust — likely not a clean bib

        return mean_vec / concentration  # normalized direction = "the color"

    def collect_sample(self, frame, bbox, player_track_id):
        """Call every frame, for every tracked player, before teams are
        finalized. Keeps one sample per unique player so the pool stays
        balanced rather than dominated by whoever's on screen most."""
        if self._kmeans is not None:
            return
        if player_track_id in self._color_pool:
            return
        vec = self._get_player_hue_vector(frame, bbox)
        if vec is not None:
            self._color_pool[player_track_id] = vec

    def try_finalize_teams(self, frame_num):
        if self._kmeans is not None:
            return True

        have_enough = len(self._color_pool) >= MIN_UNIQUE_PLAYERS_TO_INIT
        past_deadline = frame_num >= FALLBACK_FRAME_LIMIT and len(self._color_pool) >= 2
        if not (have_enough or past_deadline):
            return False

        vectors = np.array(list(self._color_pool.values()))
        km = KMeans(n_clusters=2, n_init=10, random_state=0)
        km.fit(vectors)
        self._kmeans = km
        self.team_hue_vectors[1] = km.cluster_centers_[0]
        self.team_hue_vectors[2] = km.cluster_centers_[1]
        print(f"Team colors finalized at frame {frame_num} using {len(vectors)} unique player samples.")
        return True

    def get_player_team(self, frame, bbox, player_track_id):
        if self._kmeans is None:
            return None

        vec = self._get_player_hue_vector(frame, bbox)
        if vec is not None:
            centers = self._kmeans.cluster_centers_
            dists = [np.linalg.norm(vec - c) for c in centers]
            nearest_idx = int(np.argmin(dists))
            if dists[nearest_idx] <= MAX_HUE_DIST_FOR_TEAM:
                guess = nearest_idx + 1
                votes = self.player_team_votes.setdefault(player_track_id, deque(maxlen=VOTES_PER_PLAYER))
                votes.append(guess)
            # else: doesn't confidently match either bib color — skip this
            # frame's sample rather than force a guess.

        votes = self.player_team_votes.get(player_track_id)
        if not votes or len(votes) < MIN_VOTES_TO_DECIDE:
            return None

        most_common_team, _ = Counter(votes).most_common(1)[0]
        return most_common_team