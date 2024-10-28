import numpy as np

from modules.utils import *
import stumpy


def top_k_motifs(ts, matrix_profile, top_k=3):
    """
    Find the top-k motifs based on matrix profile.

    Parameters
    ---------
    matrix_profile : dict
        The matrix profile structure.

    top_k : int
        Number of motifs.

    Returns
    --------
    motifs : dict
        Top-k motifs (left and right indices and distances).
    """

    motifs = stumpy.motifs(ts, matrix_profile['mp'], cutoff=np.nanmean(matrix_profile['mp']), max_motifs=top_k) 

    return {
        "distances" : motifs[0],
        "indices" : motifs[1]
        }
