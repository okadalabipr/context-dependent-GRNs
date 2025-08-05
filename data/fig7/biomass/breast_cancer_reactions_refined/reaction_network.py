from typing import Dict, List

from .name2idx import C, V


class ReactionNetwork(object):
    """
    Reaction indices grouped according to biological processes.
    This is used for sensitivity analysis (``target``='reaction').
    """

    def __init__(self) -> None:
        super(ReactionNetwork, self).__init__()
        self.reactions: Dict[str, List[int]] = {}

    @staticmethod
    def flux(t, y, x):
        """
        Flux vector.
        """

        v = {}
        v[1] = x[C.V_Heregulin_HER3_HER2_p_RAS] * y[V.Heregulin_HER3_HER2_p] * y[V.RAS_inact] / ( x[C.K_Heregulin_HER3_HER2_p_RAS] + y[V.RAS_inact] )
        v[2] = x[C.V_Heregulin_HER3_HER2_p_PI3K] * y[V.Heregulin_HER3_HER2_p] * y[V.PI3K_inact] / ( x[C.K_Heregulin_HER3_HER2_p_PI3K] + y[V.PI3K_inact] )
        v[3] = x[C.V_EGF_EGFR_p_RAS] * y[V.EGF_EGFR_p] * y[V.RAS_inact] / ( x[C.K_EGF_EGFR_p_RAS] + y[V.RAS_inact] )
        v[4] = x[C.V_EGFR_EGFR_p_RAS] * y[V.EGFR_EGFR_p] * y[V.RAS_inact] / ( x[C.K_EGFR_EGFR_p_RAS] + y[V.RAS_inact] )
        v[5] = x[C.V_EGFR_EGFR_p_PI3K] * y[V.EGFR_EGFR_p] * y[V.PI3K_inact] / ( x[C.K_EGFR_EGFR_p_PI3K] + y[V.PI3K_inact] )
        v[6] = x[C.V_EGFR_HER2_p_RAS] * y[V.EGFR_HER2_p] * y[V.RAS_inact] / ( x[C.K_EGFR_HER2_p_RAS] + y[V.RAS_inact] )
        v[7] = x[C.V_EGFR_HER2_p_PI3K] * y[V.EGFR_HER2_p] * y[V.PI3K_inact] / ( x[C.K_EGFR_HER2_p_PI3K] + y[V.PI3K_inact] )
        v[8] = x[C.V_Shc_p_RAS] * y[V.Shc_p] * y[V.RAS_inact] / ( x[C.K_Shc_p_RAS] + y[V.RAS_inact] )
        v[9] = x[C.V_RAS_RAF1] * y[V.RAS_act] * y[V.RAF1_inact] / ( x[C.K_RAS_RAF1] + y[V.RAF1_inact] )
        v[10] = x[C.V_PI3K_AKT] * y[V.PI3K_act] * y[V.AKT_inact] / ( x[C.K_PI3K_AKT] + y[V.AKT_inact] )
        v[11] = x[C.V_PIP3_AKT] * y[V.PIP3] * y[V.AKT_inact] / ( x[C.K_PIP3_AKT] + y[V.AKT_inact] )
        v[12] = x[C.V_SPRYiRAF1] * y[V.SPRY] * y[V.RAF1_act] / ( x[C.K_RAF1i] + y[V.RAF1_act] )
        v[13] = x[C.kf13] * y[V.Heregulin] * y[V.HER3] - x[C.kr13] * y[V.Heregulin_HER3]
        v[14] = x[C.kf14] * y[V.Heregulin_HER3] * y[V.HER2] - x[C.kr14] * y[V.Heregulin_HER3_HER2]
        v[15] = x[C.kf15] * y[V.Heregulin_HER3_HER2] - x[C.kr15] * y[V.Heregulin_HER3_HER2_p]
        v[16] = x[C.kf16] * y[V.EGF] * y[V.EGFR] - x[C.kr16] * y[V.EGF_EGFR]
        v[17] = x[C.kf17] * y[V.EGF_EGFR] * y[V.EGF_EGFR] - x[C.kr17] * y[V.EGF_EGFR_EGF_EGFR]
        v[18] = x[C.kf18] * y[V.EGF_EGFR_EGF_EGFR] - x[C.kr18] * y[V.EGF_EGFR_p]
        v[19] = x[C.kf19] * y[V.EGFR] * y[V.EGFR] - x[C.kr19] * y[V.EGFR_EGFR]
        v[20] = x[C.kf20] * y[V.EGFR_EGFR] - x[C.kr20] * y[V.EGFR_EGFR_p]
        v[21] = x[C.kf21] * y[V.EGFR] * y[V.HER2] - x[C.kr21] * y[V.EGFR_HER2]
        v[22] = x[C.kf22] * y[V.EGFR_HER2] - x[C.kr22] * y[V.EGFR_HER2_p]
        v[23] = x[C.V23] * y[V.EGFR_EGFR_p] * y[V.Shc] / (x[C.K23] + y[V.Shc])
        v[24] = x[C.V24] * y[V.RAF1_act] * y[V.MEK] / (x[C.K24] + y[V.MEK])
        v[25] = x[C.V25] * y[V.MEK_p] * y[V.ERK] / (x[C.K25] + y[V.ERK])
        v[26] = x[C.kf26] * y[V.ERK_p_cytoplasm] - x[C.kr26] * y[V.ERK_p_nucleus]
        v[27] = x[C.V27] * y[V.ERK_p_cytoplasm] * y[V.RSK] / (x[C.K27] + y[V.RSK])
        v[28] = x[C.V28] * y[V.RSK_p] * y[V.cFOS] / (x[C.K28] + y[V.cFOS])
        v[29] = x[C.kf29] * y[V.cFOS_p]
        v[30] = x[C.V30] * y[V.PI3K_act] * y[V.PIP2] / (x[C.K30] + y[V.PIP2])
        v[31] = x[C.V31] * y[V.PTEN] * y[V.PIP3] / (x[C.K31] + y[V.PIP3])
        v[32] = x[C.V32] * y[V.PI3K_act] * y[V.AKT_inact] / (x[C.K32] + y[V.AKT_inact])
        v[33] = x[C.V33] * y[V.AKT_p] * y[V.GSK3B] / (x[C.K33] + y[V.GSK3B])
        v[34] = x[C.kf34] * y[V.AKT_p] * y[V.CDKN1A]
        v[35] = x[C.kf35] * y[V.EGFR]
        v[36] = x[C.V36] * y[V.ERK_p_nucleus] ** x[C.n36] / (x[C.K36] ** x[C.n36] + y[V.ERK_p_nucleus] ** x[C.n36])
        v[37] = x[C.V37] * y[V.DUSP] * y[V.ERK_p_cytoplasm] / (x[C.K37] + y[V.ERK_p_cytoplasm])
        v[38] = x[C.V38] * y[V.ERK_p_nucleus] ** x[C.n38] / (x[C.K38] ** x[C.n38] + y[V.ERK_p_nucleus] ** x[C.n38])
        v[39] = x[C.V39] * y[V.AKT_p] ** x[C.n39] / (x[C.K39] ** x[C.n39] + y[V.AKT_p] ** x[C.n39])
        v[40] = x[C.kf40] * y[V.GSK3B] * y[V.CDKN1A]
        v[41] = x[C.kf41] * y[V.Shc_p]
        v[42] = x[C.kf42] * y[V.MEK_p]
        v[43] = x[C.kf43] * y[V.RSK_p]
        v[44] = x[C.kf44] * y[V.AKT_p]
        v[45] = x[C.kf45] * y[V.GSK3B_p]
        v[46] = x[C.kf46] * y[V.DUSP]
        v[47] = x[C.kf47] * y[V.SPRY]
        v[50] = x[C.V50] * y[V.ERK_p_nucleus] ** x[C.n50] / (x[C.K50] ** x[C.n50] + y[V.ERK_p_nucleus] ** x[C.n50])
        v[51] = x[C.kf51] * y[V.cFOS_mRNA]
        v[52] = x[C.kf52] * y[V.cFOS]
        v[53] = x[C.V53] * y[V.cFOS_p] / (x[C.K53] + y[V.cFOS_p])

        return v
