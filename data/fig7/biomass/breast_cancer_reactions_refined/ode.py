from .name2idx import C, V
from .reaction_network import ReactionNetwork


class DifferentialEquation(ReactionNetwork):
    def __init__(self, perturbation):
        super(DifferentialEquation, self).__init__()
        self.perturbation = perturbation

    def diffeq(self, t, y, *x):
        """Kinetic equations"""
        v = self.flux(t, y, x)

        if self.perturbation:
            for i, dv in self.perturbation.items():
                v[i] = v[i] * dv

        dydt = [0] * V.NUM
        dydt[V.RAS_inact] = - v[1] - v[3] - v[4] - v[6] - v[8]
        dydt[V.RAS_act] = + v[1] + v[3] + v[4] + v[6] + v[8]
        dydt[V.PI3K_inact] = - v[2] - v[5] - v[7]
        dydt[V.PI3K_act] = + v[2] + v[5] + v[7]
        dydt[V.RAF1_inact] = - v[9] + v[12]
        dydt[V.RAF1_act] = + v[9] - v[12]
        dydt[V.AKT_inact] = - v[10] - v[11] - v[32]
        dydt[V.AKT_p] = + v[10] + v[11] + v[32] - v[44]
        dydt[V.Heregulin] = - v[13]
        dydt[V.HER3] = - v[13]
        dydt[V.Heregulin_HER3] = + v[13] - v[14]
        dydt[V.HER2] = - v[14] - v[21]
        dydt[V.Heregulin_HER3_HER2] = + v[14] - v[15]
        dydt[V.Heregulin_HER3_HER2_p] = + v[15]
        dydt[V.EGF] = - v[16]
        dydt[V.EGFR] = - v[16] - 2 * v[19] - v[21] - v[35]
        dydt[V.EGF_EGFR] = + v[16] - 2 * v[17]
        dydt[V.EGF_EGFR_EGF_EGFR] = + v[17] - v[18]
        dydt[V.EGF_EGFR_p] = + v[18]
        dydt[V.EGFR_EGFR] = + v[19] - v[20]
        dydt[V.EGFR_EGFR_p] = + v[20]
        dydt[V.EGFR_HER2] = + v[21] - v[22]
        dydt[V.EGFR_HER2_p] = + v[22]
        dydt[V.Shc] = - v[23]
        dydt[V.Shc_p] = + v[23] - v[41]
        dydt[V.MEK] = - v[24]
        dydt[V.MEK_p] = + v[24] - v[42]
        dydt[V.ERK] = - v[25] + v[37]
        dydt[V.ERK_p_cytoplasm] = + v[25] - v[26] - v[37]
        dydt[V.ERK_p_nucleus] = + v[26]
        dydt[V.RSK] = - v[27]
        dydt[V.RSK_p] = + v[27] - v[43]
        dydt[V.cFOS] = - v[28] + v[51] - v[52] + v[53]
        dydt[V.cFOS_p] = + v[28] - v[29] - v[53]
        dydt[V.PIP2] = - v[30] + v[31]
        dydt[V.PIP3] = + v[30] - v[31]
        dydt[V.GSK3B] = - v[33]
        dydt[V.GSK3B_p] = + v[33] - v[45]
        dydt[V.CDKN1A] = - v[34] + v[39] - v[40]
        dydt[V.DUSP] = + v[36] - v[46]
        dydt[V.SPRY] = + v[38] - v[47]
        dydt[V.cFOS_mRNA] = + v[50]

        return dydt


def param_values():
    """Parameter values"""
    x = [1] * C.NUM
    x[C.kf29] = 1e-03
    x[C.kf41] = 1e-03
    x[C.kf42] = 1e-03
    x[C.kf43] = 1e-03
    x[C.kf44] = 1e-03
    x[C.kf45] = 1e-03
    x[C.kf46] = 1e-03
    x[C.kf47] = 1e-03
    x[C.kf52] = 1e-02
    x[C.V53] = 1e-03

    return x


def initial_values():
    """Values of the initial condition"""
    y0 = [0] * V.NUM
    y0[V.RAS_inact] = 1.00e02
    y0[V.PI3K_inact] = 1.00e02
    y0[V.RAF1_inact] = 1.00e02
    y0[V.AKT_inact] = 1.00e02
    y0[V.HER3] = 1.00e02
    y0[V.HER2] = 1.00e02
    y0[V.EGFR] = 1.00e02
    y0[V.Shc] = 1.00e02
    y0[V.MEK] = 1.00e02
    y0[V.ERK] = 9.60e02
    y0[V.RSK] = 3.53e02
    y0[V.cFOS] = 1.00e02
    y0[V.PIP2] = 1.00e02
    y0[V.PTEN] = 1.00e02
    y0[V.GSK3B] = 1.00e02

    return y0
