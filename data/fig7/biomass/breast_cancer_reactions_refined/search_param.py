import numpy as np

from biomass.estimation import convert_scale, initialize_search_param

from .name2idx import C, V
from .ode import initial_values, param_values


class SearchParam(object):
    """Specify model parameters and/or initial values to optimize."""

    def __init__(self):
        # parameters
        self.idx_params = [
            C.V_Heregulin_HER3_HER2_p_RAS,
            C.K_Heregulin_HER3_HER2_p_RAS,
            C.V_Heregulin_HER3_HER2_p_PI3K,
            C.K_Heregulin_HER3_HER2_p_PI3K,
            C.V_EGF_EGFR_p_RAS,
            C.K_EGF_EGFR_p_RAS,
            C.V_EGFR_EGFR_p_RAS,
            C.K_EGFR_EGFR_p_RAS,
            C.V_EGFR_EGFR_p_PI3K,
            C.K_EGFR_EGFR_p_PI3K,
            C.V_EGFR_HER2_p_RAS,
            C.K_EGFR_HER2_p_RAS,
            C.V_EGFR_HER2_p_PI3K,
            C.K_EGFR_HER2_p_PI3K,
            C.V_Shc_p_RAS,
            C.K_Shc_p_RAS,
            C.V_RAS_RAF1,
            C.K_RAS_RAF1,
            C.V_PI3K_AKT,
            C.K_PI3K_AKT,
            C.V_PIP3_AKT,
            C.K_PIP3_AKT,
            C.V_SPRYiRAF1,
            C.K_RAF1i,
            C.kf13,
            C.kr13,
            C.kf14,
            C.kr14,
            C.kf15,
            C.kr15,
            C.kf16,
            C.kr16,
            C.kf17,
            C.kr17,
            C.kf18,
            C.kr18,
            C.kf19,
            C.kr19,
            C.kf20,
            C.kr20,
            C.kf21,
            C.kr21,
            C.kf22,
            C.kr22,
            C.V23,
            C.K23,
            C.V24,
            C.K24,
            C.V25,
            C.K25,
            C.kf26,
            C.kr26,
            C.V27,
            C.K27,
            C.V28,
            C.K28,
            C.kf29,
            C.V30,
            C.K30,
            C.V31,
            C.K31,
            C.V32,
            C.K32,
            C.V33,
            C.K33,
            C.kf34,
            C.kf35,
            C.V36,
            C.K36,
            C.n36,
            C.V37,
            C.K37,
            C.V38,
            C.K38,
            C.n38,
            C.V39,
            C.K39,
            C.n39,
            C.kf40,
            C.kf41,
            C.kf42,
            C.kf43,
            C.kf44,
            C.kf45,
            C.kf46,
            C.kf47,
            C.V50,
            C.K50,
            C.n50,
            C.kf51,
            C.kf52,
            C.V53,
            C.K53,
        ]

        # initial values
        self.idx_initials = []

    def get_region(self):
        x = param_values()
        y0 = initial_values()

        search_param = initialize_search_param(
            parameters=C.NAMES,
            species=V.NAMES,
            param_values=x,
            initial_values=y0,
            estimated_params=self.idx_params,
            estimated_initials=self.idx_initials,
        )

        search_rgn = np.zeros((2, len(x) + len(y0)))
        # Default: 0.1 ~ 10
        for i, j in enumerate(self.idx_params):
            search_rgn[0, j] = search_param[i] * 0.1  # lower bound
            search_rgn[1, j] = search_param[i] * 10.0  # upper bound
        # Default: 0.5 ~ 2
        for i, j in enumerate(self.idx_initials):
            search_rgn[0, j + len(x)] = search_param[i + len(self.idx_params)] * 0.5  # lower bound
            search_rgn[1, j + len(x)] = search_param[i + len(self.idx_params)] * 2.0  # upper bound

        # search_rgn[:,C.parameter] = [lower_bound, upper_bound]
        # search_rgn[:,V.specie+len(x)] = [lower_bound, upper_bound]

        search_rgn = convert_scale(
            region=search_rgn,
            parameters=C.NAMES,
            species=V.NAMES,
            estimated_params=self.idx_params,
            estimated_initials=self.idx_initials,
        )

        return search_rgn

    def update(self, indiv):
        x = param_values()
        y0 = initial_values()

        for i, j in enumerate(self.idx_params):
            x[j] = indiv[i]
        for i, j in enumerate(self.idx_initials):
            y0[j] = indiv[i + len(self.idx_params)]

        # parameter constraints
        

        return x, y0
