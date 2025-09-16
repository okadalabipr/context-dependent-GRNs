# Manual Changes Made to the Breast Cancer Model

Below is the summary of the manual changes made to the generated breast cancer model file:

- Convert `-` in species names to `_` to avoid issues with Python variable names
- Convert `AKT_act` to `AKT_p` (deduplication of activated AKT species)
- Delete EGF and ERK degradation reactions
- Add the following reactions:
  - `cFOS_mRNA` transcription by `ERK_p_nucleus`
  - `cFOS` translation from `cFOS_mRNA`
  - `cFOS` degradation
  - `cFOS_p` dephosphorylation
- Define observables based on the following papers:
  - [Nagashima et al. 2007](https://doi.org/10.1074/jbc.M608653200)
    - phosphorylated EGFR and HER2
  - [Birtwistle et al. 2007](https://doi.org/10.1038/msb4100188)
    - phosphorylated SHC, MEK, ERK, AKT
  - [Nakakuki et al. 2011](https://doi.org/10.1016/j.cell.2010.03.054)
    - Phosphorylated RSK and cFOS
- Define the following:
  - simulation time (30 minutes) and simulation conditions (EGF and HRG stimulations)
  - initial concentrations for the inactive states of the species
  - initial values for the rate constants of the reactions
