from typing import Dict, List

import numpy as np

from biomass.dynamics.solver import *

from .name2idx import C, V
from .ode import DifferentialEquation


class Observable(DifferentialEquation):
    """
    Correlating model simulations and experimental measurements.

    Attributes
    ----------
    obs_names : list of strings
        Names of model observables.

    t : range
        Simulation time span.

    conditions : list of strings
        Experimental conditions.

    simulations : numpy.ndarray
        The numpy array to store simulation results.

    normalization : nested dict
        * 'timepoint' : Optional[int]
            The time point at which simulated values are normalized.
            If :obj:`None`, the maximum value will be used for normalization.

        * 'condition' : list of strings
            The experimental conditions to use for normalization.
            If empty, all conditions defined in ``sim.conditions`` will be used.

    experiments : list of dict
        Time series data.

    error_bars : list of dict
        Error bars to show in figures.

    """

    def __init__(self):
        super(Observable, self).__init__(perturbation={})
        self.obs_names: list = [
            "Phosphorylated_EGFR",
            "Phosphorylated_HER2",
            "Phosphorylated_Shc",
            "Phosphorylated_MEK",
            "Phosphorylated_ERK",
            # observables below were not used for parameter estimation;
            # if you wish to reproduce the results, comment out the lines below
            # when running the parameter estimation
            "Phosphorylated_HER3",
            "Phosphorylated_RSK",
            "Phosphorylated_FOS",
        ]
        self.t: range = range(0, 1800 + 1)
        self.conditions: list = [
            "EGF",
            "HRG",
        ]
        self.simulations: np.ndarray = np.empty(
            (len(self.obs_names), len(self.conditions), len(self.t))
        )
        self.normalization: dict = {}
        self.experiments: list = [None] * len(self.obs_names)
        self.error_bars: list = [None] * len(self.obs_names)
        for observable in self.obs_names:
            self.normalization[observable] = {"timepoint": None, "condition": []}

    def simulate(self, x: list, y0: list, _perturbation: dict = {}):
        if _perturbation:
            self.perturbation = _perturbation
        # unperturbed steady state

        for i, condition in enumerate(self.conditions):
            if condition == "EGF":
                y0[V.EGF] = 1.0e01
            elif condition == "HRG":
                y0[V.Heregulin] = 1.0e01

            sol = solve_ode(self.diffeq, y0, self.t, tuple(x))

            if sol is None:
                return False
            else:
                self.simulations[self.obs_names.index("Phosphorylated_EGFR"), i] = (
                    2 * sol.y[V.EGF_EGFR_p]
                    + 2 * sol.y[V.EGFR_EGFR_p]
                    + sol.y[V.EGFR_HER2_p]
                ) / (
                    sol.y[V.EGFR]
                    + sol.y[V.EGF_EGFR]
                    + 2 * sol.y[V.EGF_EGFR_EGF_EGFR]
                    + 2 * sol.y[V.EGF_EGFR_p]
                    + 2 * sol.y[V.EGFR_EGFR]
                    + 2 * sol.y[V.EGFR_EGFR_p]
                    + sol.y[V.EGFR_HER2]
                    + sol.y[V.EGFR_HER2_p]
                )
                self.simulations[self.obs_names.index("Phosphorylated_HER2"), i] = (
                    sol.y[V.Heregulin_HER3_HER2_p] + sol.y[V.EGFR_HER2_p]
                ) / (
                    sol.y[V.HER2]
                    + sol.y[V.Heregulin_HER3_HER2]
                    + sol.y[V.Heregulin_HER3_HER2_p]
                    + sol.y[V.EGFR_HER2]
                    + sol.y[V.EGFR_HER2_p]
                )
                self.simulations[self.obs_names.index("Phosphorylated_Shc"), i] = sol.y[
                    V.Shc_p
                ] / (sol.y[V.Shc] + sol.y[V.Shc_p])
                self.simulations[self.obs_names.index("Phosphorylated_MEK"), i] = sol.y[
                    V.MEK_p
                ] / (sol.y[V.MEK] + sol.y[V.MEK_p])
                self.simulations[self.obs_names.index("Phosphorylated_ERK"), i] = (
                    sol.y[V.ERK_p_cytoplasm] + sol.y[V.ERK_p_nucleus]
                ) / (sol.y[V.ERK] + sol.y[V.ERK_p_cytoplasm] + sol.y[V.ERK_p_nucleus])
                self.simulations[
                    self.obs_names.index("Phosphorylated_HER3"), i
                ] = sol.y[V.Heregulin_HER3_HER2_p] / (
                    sol.y[V.HER3]
                    + sol.y[V.Heregulin_HER3]
                    + sol.y[V.Heregulin_HER3_HER2]
                    + sol.y[V.Heregulin_HER3_HER2_p]
                )
                self.simulations[self.obs_names.index("Phosphorylated_RSK"), i] = sol.y[
                    V.RSK_p
                ]
                self.simulations[self.obs_names.index("Phosphorylated_FOS"), i] = sol.y[
                    V.cFOS_p
                ]

    def set_data(self):
        self.experiments[self.obs_names.index("Phosphorylated_EGFR")] = {
            "EGF": [0.001, 0.444, 0.464, 0.208, 0.129, 0.119],
            "HRG": [0.001, 1.0, 0.857, 0.813, 0.488, 0.454],
        }

        self.experiments[self.obs_names.index("Phosphorylated_HER2")] = {
            "EGF": [0.144, 0.25, 0.226, 0.176, 0.19, 0.165],
            "HRG": [0.179, 0.902, 1.0, 0.858, 0.636, 0.724],
        }

        self.experiments[self.obs_names.index("Phosphorylated_Shc")] = {
            "EGF": [0.0, 0.355, 0.389, 0.331, 0.321, 0.205],
            "HRG": [0.0, 0.479, 0.552, 0.508, 0.731, 1.0],
        }

        self.experiments[self.obs_names.index("Phosphorylated_MEK")] = {
            "EGF": [0.004, 0.111, 0.342, 0.45, 0.298, 0.091],
            "HRG": [0.007, 0.137, 0.547, 0.957, 0.913, 1.0],
        }

        self.experiments[self.obs_names.index("Phosphorylated_ERK")] = {
            "EGF": [0.0, 0.127, 0.536, 0.778, 0.945, 0.327],
            "HRG": [0.004, 0.15, 0.57, 0.98, 1.0, 0.914],
        }

        self.experiments[self.obs_names.index("Phosphorylated_HER3")] = {
            "EGF": [0.162, 0.15, 0.121, 0.15, 0.127, 0.086],
            "HRG": [0.035, 0.715, 0.75, 0.924, 0.895, 1.0],
        }
        self.experiments[self.obs_names.index("Phosphorylated_RSK")] = {
            "EGF": [0, 0.814, 0.812, 0.450, 0.151],
            "HRG": [0, 0.953, 1.000, 0.844, 0.935],
            # "EGF": [0, 0.814, 0.812, 0.450, 0.151, 0.059, 0.038, 0.030],
            # "HRG": [0, 0.953, 1.000, 0.844, 0.935, 0.868, 0.779, 0.558],
        }
        self.experiments[self.obs_names.index("Phosphorylated_FOS")] = {
            "EGF": [0, 0.1, 0.182, 0.139, 0.114],
            "HRG": [0, 0.242, 0.296, 0.264, 1.000],
            # "EGF": [0, 0.060, 0.109, 0.083, 0.068, 0.049, 0.027, 0.017],
            # "HRG": [0, 0.145, 0.177, 0.158, 0.598, 1.000, 0.852, 0.431],
        }

    def get_timepoint(self, obs_name) -> Dict[str, List[int]]:
        if obs_name in [
            "Phosphorylated_EGFR",
            "Phosphorylated_HER2",
            "Phosphorylated_Shc",
            "Phosphorylated_MEK",
            "Phosphorylated_ERK",
            "Phosphorylated_HER3",
        ]:
            return {
                condition: [0, 60, 120, 300, 600, 1800]  # (Unit: sec.)
                for condition in self.conditions
            }
        elif obs_name in [
            "Phosphorylated_RSK",
            "Phosphorylated_FOS",
        ]:
            return {
                condition: [0, 300, 600, 900, 1800]  # (Unit: sec.)
                # condition: [0, 300, 600, 900, 1800, 2700, 3600, 5400]  # (Unit: sec.)
                for condition in self.conditions
            }
        assert False

    def simulate_all(self, x: list, y0: list, _perturbation: dict = {}):
        simulations = np.empty((len(self.conditions), V.NUM, len(self.t)))
        if _perturbation:
            self.perturbation = _perturbation
        # unperturbed steady state

        for i, condition in enumerate(self.conditions):
            if condition == "EGF":
                y0[V.EGF] = 1.0e01
            elif condition == "HRG":
                y0[V.Heregulin] = 1.0e01

            sol = solve_ode(self.diffeq, y0, self.t, tuple(x))

            if sol is None:
                return False
            else:
                simulations[i] = sol.y
        return simulations
