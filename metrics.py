"""Scanpath metrics used in GazeLNN, section V-A, and Fahimi & Bruce (2021).

Conventions follow https://github.com/rAm1n/saliency/tree/master/metrics:
raw character-level Levenshtein on row/column strings, spatial ScanMatch
(no duration expansion), discrete Frechet, FastDTW with Euclidean costs,
and directed mean TDE with k=3. Spatial distances use input coordinate units.
"""

import numpy as np
from scipy.spatial.distance import directed_hausdorff, euclidean
import Levenshtein
from fastdtw import fastdtw
import similaritymeasures


def _as_scanpath(scanpath, name):
    """Validate coordinates and convert before subtraction (including uint inputs)."""
    points = np.asarray(scanpath, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2) with (x, y) coordinates")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return points


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def scanpath_to_string(scanpath, height=256, width=384, Xbins=12, Ybins=8):
    """
        Convert scanpath to 2-char-per-fixation string and numeric cell indices,
    per Fahimi et al. (2021) reference implementation (rAm1n/saliency).
    Each fixation is encoded as 2 characters (row_char + col_char).
    Also returns a list of integer cell indices for ScanMatch.
    Args:
        scanpath: (N, 2) array of (x, y) coordinates
        height, width: coordinate space dimensions
        Xbins, Ybins: grid divisions (12x8 for GazeLNN)
    Returns:
        string: 2-char-per-fixation string for Levenshtein
        num: list of integer cell indices for ScanMatch
    """
    scanpath = _as_scanpath(scanpath, "scanpath")
    Xbins = _positive_int(Xbins, "Xbins")
    Ybins = _positive_int(Ybins, "Ybins")
    if Xbins > 26 or Ybins > 26:
        raise ValueError("Xbins and Ybins must be at most 26 for alphabetic encoding")
    if not np.isfinite(height) or not np.isfinite(width) or height <= 0 or width <= 0:
        raise ValueError("height and width must be finite positive numbers")

    # Equal-width cells also when the image dimensions are not divisible by bins.
    height_step = height / Ybins
    width_step = width / Xbins
    string = ''
    num = []
    for pt in scanpath:
        x, y = pt[0], pt[1]
        # Retain the reference's integer-pixel and inverted-y conventions.
        # Clip only for quantization; continuous metrics use the original points.
        x = int(np.clip(x, 0, width))
        y = int(np.clip(y, 0, height))
        xbin = min(int(x // width_step), Xbins - 1)
        ybin = min(int((height - y) // height_step), Ybins - 1)
        corrs_x = chr(65 + xbin)    # 'A'-'L' (column)
        corrs_y = chr(97 + ybin)    # 'a'-'h' (row)
        string += (corrs_y + corrs_x)
        num.append(ybin * Xbins + xbin)
    return string, num

def cell_distance(idx1, idx2, Xbins=12):
    """Euclidean grid distance between two cell indices."""
    col1, row1 = idx1 % Xbins, idx1 // Xbins
    col2, row2 = idx2 % Xbins, idx2 // Xbins
    return np.sqrt((col1 - col2)**2 + (row1 - row2)**2)

def scanmatch_score(P_num, Q_num, threshold=3.5, gap_penalty=0.0, *, Xbins=12, Ybins=8):
    """
    ScanMatch score using Needleman-Wunsch alignment on cell indices.
    Substitution score = threshold - grid_distance (can be negative,
    matching the reference SubMatrix from Fahimi et al.).
    Normalized by threshold * max(len(P), len(Q)).
    Xbins and Ybins must match the grid used to encode both sequences.
    With the default zero gap penalty the score is in [0, 1].
    """
    Xbins = _positive_int(Xbins, "Xbins")
    Ybins = _positive_int(Ybins, "Ybins")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    if not np.isfinite(gap_penalty):
        raise ValueError("gap_penalty must be finite")
    for sequence in (P_num, Q_num):
        for cell in sequence:
            if not isinstance(cell, (int, np.integer)) or not 0 <= cell < Xbins * Ybins:
                raise ValueError("cell indices must be integers within the specified grid")
    n, m = len(P_num), len(Q_num)
    if n == 0 or m == 0:
        return 0.0
        
    dp = np.zeros((n + 1, m + 1))
    
    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + gap_penalty
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] + gap_penalty
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Substitution score: threshold - grid distance (can be negative)
            match_score = threshold - cell_distance(P_num[i-1], Q_num[j-1], Xbins)
            dp[i][j] = max(
                dp[i-1][j-1] + match_score,
                dp[i-1][j] + gap_penalty,
                dp[i][j-1] + gap_penalty
            )

    # Normalize by max possible score (per Fahimi reference)
    scale = threshold * max(n, m)

    return dp[n][m] / scale if scale > 0 else 0.0

def time_delay_embedding(P, Q, k=3, distance_mode='Mean'):
    """
    Time Delay Embedding (TDE) distance per Wang et al. (2011),
    following the FixaTons reference cited by Fahimi et al.
    
    For each k-vector from Q, finds the spatially nearest P k-vector,
    divides that distance by k, then aggregates (Mean or Hausdorff).
    
    Args:
        P: ground truth scanpath, shape (N, 2)
        Q: predicted scanpath, shape (M, 2)
        k: embedding dimension (default 3)
        distance_mode: 'Mean' or 'Hausdorff'
    """
    P = _as_scanpath(P, "P")
    Q = _as_scanpath(Q, "Q")
    k = _positive_int(k, "k")
    if distance_mode not in ('Mean', 'Hausdorff'):
        raise ValueError("distance_mode must be 'Mean' or 'Hausdorff'")
    if len(P) < k or len(Q) < k:
        raise ValueError("both scanpaths must contain at least k fixations for TDE")

    # Create time-embedding vectors (k-subsequences)
    P_vectors = [P[i:i + k] for i in range(len(P) - k + 1)]
    Q_vectors = [Q[i:i + k] for i in range(len(Q) - k + 1)]

    # For each Q vector, find nearest P vector
    distances = []
    for s_k_vec in Q_vectors:
        norms = []
        for h_k_vec in P_vectors:
            # Euclidean distance between flattened k-vectors
            d = np.linalg.norm(s_k_vec - h_k_vec)
            norms.append(d)
        distances.append(min(norms) / k)

    if distance_mode == 'Mean':
        return sum(distances) / len(distances)
    return max(distances)


def evaluate_metrics(pred_scanpath, gt_scanpath, height=256, width=384):
    """
    Evaluates the 6 metrics given predicted and ground truth scanpaths.
    Both should be numpy arrays of shape (N, 2) in the same coordinate space.
    Empty or non-finite scanpaths raise ValueError. TDE is NaN when either
    path has fewer than three fixations; exclude those values when aggregating
    TDE (e.g. with np.nanmean). Other computation errors propagate.
    No durations are accepted, so ScanMatch uses spatial symbols only.
    """

    pred_scanpath = _as_scanpath(pred_scanpath, "pred_scanpath")
    gt_scanpath = _as_scanpath(gt_scanpath, "gt_scanpath")
    if len(pred_scanpath) == 0 or len(gt_scanpath) == 0:
        raise ValueError("both scanpaths must contain at least one fixation")

    # Discretize scanpaths (2-char strings + cell indices)
    pred_str, pred_num = scanpath_to_string(pred_scanpath, height, width)
    gt_str, gt_num = scanpath_to_string(gt_scanpath, height, width)

    # 1. Levenshtein Distance (raw edit distance on 2-char strings)
    lev_dist = Levenshtein.distance(pred_str, gt_str)

    # 2. ScanMatch (Needleman-Wunsch on cell indices)
    sm_score = scanmatch_score(pred_num, gt_num)
    
    # 3. Hausdorff Distance
    hausdorff = max(directed_hausdorff(pred_scanpath, gt_scanpath)[0],
                    directed_hausdorff(gt_scanpath, pred_scanpath)[0])
                    
    # 4. Frechet Distance
    frechet = similaritymeasures.frechet_dist(pred_scanpath, gt_scanpath)
        
    # 5. fast DTW
    fdtw, _ = fastdtw(pred_scanpath, gt_scanpath, dist=euclidean)
    
    # 6. Time Delay Embedding (one-directional nearest-neighbor / k)
    # TDE is directional: every predicted window finds its nearest GT window.
    tde = (time_delay_embedding(gt_scanpath, pred_scanpath, k=3, distance_mode='Mean')
           if min(len(pred_scanpath), len(gt_scanpath)) >= 3 else float('nan'))
    
    return {
        "Levenshtein": lev_dist,
        "ScanMatch": sm_score,
        "Hausdorff": hausdorff,
        "Frechet": frechet,
        "FastDTW": fdtw,
        "TDE": tde
    }
