import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """
    

    # INSERT YOUR CODE
    ed_dist = np.sqrt(np.sum((ts1 - ts2) ** 2))

    return ed_dist


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    norm_ed_dist = 0

    # INSERT YOUR CODE

    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW (Dynamic Time Warping) distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size (default is 1)
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """
    
    dtw_dist = 0
    r = int(np.floor(r * len(ts1)))
    
    n, m = len(ts1), len(ts2)
    matrix = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        for j in range(m + 1):
            matrix[i, j] = np.inf
    matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(max(1, i - r), min(m, i + r) + 1):
            cost = (ts1[i - 1] - ts2[j - 1])**2
            
            last_min = np.min([matrix[i - 1, j], matrix[i, j - 1], matrix[i - 1, j - 1]])
        
            matrix[i, j] = cost + last_min
            
    dtw_dist = matrix[-1][-1]

    return dtw_dist

