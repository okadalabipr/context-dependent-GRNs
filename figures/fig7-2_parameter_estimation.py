import os
import sys

from biomass import Text2Model, create_model, optimize, run_simulation


#### Convert the Text2Model file to a BioMASS model
# t2m_path = "../data/fig7/biomass/breast_cancer_reactions_refined.txt"
# model = Text2Model(t2m_path)
# model.convert()

# The BioMASS model is written to "../data/fig7/biomass/breast_cancer_reactions_refined"
# After conversion, the experimental data to use for parameter estimation must be
# defined in the "observable.py" file under the model directory.
# The model directory with the correct structure is already provided in this repository.

#### Load the BioMASS model
model_file = "../data/fig7/biomass/breast_cancer_reactions_refined"

# Due to how BioMASS loads model files,
# the model folder needs to be directly under the current working directory
model_dir = os.path.dirname(os.path.abspath(model_file))
os.chdir(model_dir)

# BioMASS needs to import the model as a module,
# thus we need to add the model directory to the Python path
cur_dir = os.path.dirname(os.path.abspath(__file__))
relpath = os.path.relpath(model_dir, cur_dir)
sys.path.append(os.path.join(cur_dir, relpath))

model_pkg = os.path.basename(model_file)
model = create_model(model_pkg)

#### Run parameter estimation
# n_paramsets = 30
# for x_id in range(1, n_paramsets + 1):
#     optimize(model, x_id=x_id, disp_here=False, optimizer_options={"workers": -1})

# The results will be saved to the "out/" directory  under the model directory.
# The optimization results used in the paper are already provided in this repository.

#### Plot the simulation results
# The following line will run the simulation with the optimized parameters
# and save the plots to the "figure/" directory under the model directory.
run_simulation(model, viz_type="average", stdev=True)
