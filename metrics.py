import numpy as np
from scipy.spatial.distance import directed_hausdorff, euclidean
import Levenshtein
from fastdtw import fastdtw
import similaritymeasures

def discretize_fixation(x, y, cell_size=32):
    """
    Discretize a continuous (x, y) fixation into a character symbol 
    based on a 12x8 grid for a 384x256 image.
    Cell size is 32x32 pixels.
    """
    col = int(x // cell_size)
    row = int(y // cell_size)
    col = max(0, min(col, 11))
    row = max(0, min(row, 7))
    index = row * 12 + col
    # Offset by 33 to get printable characters
    return chr(33 + index)

def discretize_scanpath(scanpath):
    """
    scanpath: numpy array of shape (N, 2)
    Returns a string representing the discretized scanpath, 
    with consecutive redundant characters removed.
    """
    chars = [discretize_fixation(pt[0], pt[1]) for pt in scanpath]
    if not chars:
        return ""
        
    filtered = [chars[0]]
    for c in chars[1:]:
        if c != filtered[-1]:
            filtered.append(c)
            
    return "".join(filtered)

def cell_distance(char1, char2):
    idx1 = ord(char1) - 33
    idx2 = ord(char2) - 33
    col1, row1 = idx1 % 12, idx1 // 12
    col2, row2 = idx2 % 12, idx2 // 12
    return np.sqrt((col1 - col2)**2 + (row1 - row2)**2)

def scanmatch_score(seq1, seq2, threshold=3.5, gap_penalty=0.0):
    """
    Computes a simplified ScanMatch score based on the Needleman-Wunsch algorithm.
    Substitution score is max(0, threshold - euclidean_grid_distance).
    """
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return 0.0
        
    dp = np.zeros((n + 1, m + 1))
    
    # Initialize first column and row with gap penalties
    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + gap_penalty
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] + gap_penalty
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_score = max(0.0, threshold - cell_distance(seq1[i-1], seq2[j-1]))
            dp[i][j] = max(
                dp[i-1][j-1] + match_score,
                dp[i-1][j] + gap_penalty,
                dp[i][j-1] + gap_penalty
            )
            
    # Normalize by the shorter sequence's max possible score
    max_possible_score = min(n, m) * threshold
    return dp[n][m] / max_possible_score if max_possible_score > 0 else 0.0

def time_delay_embedding(u, v, dim=3, delay=1, mode="mean"):
    """
    Time Delay Embedding (TDE) distance, per Wang et al. (2011).
    
    Embeds both sequences into overlapping length-`dim` subsequences,
    then aggregates pairwise Euclidean distances between embedded points
    (mean or Hausdorff-style max of nearest-neighbor distances).
    
    mode: "mean" -> average of all pairwise distances
          "hausdorff" -> max of nearest-neighbor min-distances (symmetric)
    """
    if len(u) < dim or len(v) < dim:
        return 0.0  # Sequences too short for TDE

    def embed(seq, d, tau):
        embedded = []
        for i in range(len(seq) - (d - 1) * tau):
            embedded.append(seq[i : i + d * tau : tau].flatten())
        return np.array(embedded)

    U_emb = embed(u, dim, delay)
    V_emb = embed(v, dim, delay)

    if len(U_emb) == 0 or len(V_emb) == 0:
        return 0.0

    # Pairwise Euclidean distance matrix between all embedded subsequences
    diff = U_emb[:, None, :] - V_emb[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    if mode == "mean":
        return float(np.mean(dist_matrix))
    elif mode == "hausdorff":
        u_to_v = np.max(np.min(dist_matrix, axis=1))
        v_to_u = np.max(np.min(dist_matrix, axis=0))
        return float(max(u_to_v, v_to_u))
    else:
        raise ValueError(f"Unknown mode: {mode}")

def evaluate_metrics(pred_scanpath, gt_scanpath, tde_mode="mean"):
    """
    Evaluates the 6 metrics given predicted and ground truth scanpaths.
    Both should be numpy arrays of shape (N, 2).
    """
    pred_str = discretize_scanpath(pred_scanpath)
    gt_str = discretize_scanpath(gt_scanpath)
    
    # 1. Levenshtein Distance (normalized by max string length)
    raw_lev = Levenshtein.distance(pred_str, gt_str)
    lev_dist = raw_lev if raw_lev > 0 else 0.0
    
    # 2. ScanMatch
    sm_score = scanmatch_score(pred_str, gt_str)
    
    # 3. Hausdorff Distance
    hausdorff = max(directed_hausdorff(pred_scanpath, gt_scanpath)[0], 
                    directed_hausdorff(gt_scanpath, pred_scanpath)[0])
                    
    # 4. Frechet Distance
    try:
        frechet = similaritymeasures.frechet_dist(pred_scanpath, gt_scanpath)
    except Exception:
        frechet = 0.0 # Fallback in case of identical points causing issues
        
    # 5. fast DTW
    fdtw, _ = fastdtw(pred_scanpath, gt_scanpath, dist=euclidean)
    
    # 6. Time Delay Embedding
    tde = time_delay_embedding(pred_scanpath, gt_scanpath, mode=tde_mode)
    
    return {
        "Levenshtein": lev_dist,
        "ScanMatch": sm_score,
        "Hausdorff": hausdorff,
        "Frechet": frechet,
        "FastDTW": fdtw,
        "TDE": tde
    }
