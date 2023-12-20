import os
from datetime import datetime
import numpy as np
import matplotlib.pylab as plt

from solverNominalMPC import solverNominalMPC
from solverClosedLoopRMPC_SIRO import solverClosedLoopRMPC_SIRO
from kite_probdef import dyn_rk4, cost_stage, cost_end, get_params, ineq_constr_stage, ineq_constr_end
from kite_plotutils import plotPsiControlThrustOverTime, plotKitePositionInAngleSpace

# set up folders for saving plots and results file
save_plots = True
plot_folder = 'kite_plots_lastrun/'
if not os.path.exists(plot_folder): os.makedirs(plot_folder)

save_results = True
res_folder = 'kite_results/'
if not os.path.exists(res_folder): os.makedirs(res_folder)

params = get_params()
def dyn_rk4_nom(x, u, params):
    return dyn_rk4(x, u, params['nw']*[0 ], params)

# some extra parameters
sigma = 1                                   # scale the overall level of uncertainty
eps_x0 = 0                                  # standard deviation of initial states
eps_x = 3 * [sigma * 1e-4]                  # standard deviation of process noise on every state
wind_delta = sigma * 1                      # possible deviation of noisy wind from true wind
x0 = params['x0']                           # initial state
P0 = eps_x0**2 * np.eye(params['nx'])       # initial state uncertainty
W =  np.diag( eps_x + [wind_delta] ) **2    # noise ellipsoid
N = params['N']                             # discrete OCP horizon
X0 = np.tile(x0[:,None], (1, N+1))          # simple initial trajectory guess 

# %% nominal OCP
solver_nom = solverNominalMPC(dyn_rk4_nom, cost_stage, cost_end, params, ineq_constr_stage, ineq_constr_end  )
solver_nom.create_solver()
solver_nom.set_initial_all( X0, np.zeros((params['nu'], N ) ))
solver_nom.set_value_x0(x0)
solver_nom.solve()
Xnom, Unom = solver_nom.get_sol()

# plot results
plotPsiControlThrustOverTime(params, Xnom, Unom)
if save_plots:
    plt.tight_layout()
    plt.savefig(plot_folder + 'kite_ocp_nom_rest.pdf')
plotKitePositionInAngleSpace(params, Xnom, Unom)
if save_plots:
    plt.tight_layout()
    plt.savefig(plot_folder + 'kite_ocp_nom_pos.pdf')


#%%# SIRO 
solver_siro = solverClosedLoopRMPC_SIRO(dyn_rk4, cost_stage, cost_end, params, ineq_constr_stage, ineq_constr_end  )
solver_siro.regu_backoff = 1e-3         # regularization backoff 
solver_siro.regu_riccatiR = 1e-6        # regularization backoff 
solver_siro.create_solver()
solver_siro.set_value_all(x0, P0, W)
solver_siro.initialize_to_nominal(X=X0)
solver_siro.solve()
X_siro, U_siro, P_siro, K_siro = solver_siro.get_sol()
P_siro = [ (p + p.T) / 2 for p in P_siro ]

# plots SIRO
plotPsiControlThrustOverTime(params, X_siro, U_siro)
if save_plots:
    plt.tight_layout()
    plt.savefig(plot_folder + 'kite_ocp_cl_rest.pdf')
plotKitePositionInAngleSpace(params, X_siro, U_siro, P=P_siro)
if save_plots:
    plt.tight_layout()
    plt.savefig(plot_folder + 'kite_ocp_cl_pos.pdf')

it_hist_SIRO = solver_siro.get_iteration_history()


# %% SIRO open loop

# solve open loop
solver_siro.closed_loop = False
solver_siro.initialize_to_nominal(X=Xnom, U=Unom)
solver_siro.solve()
X_ol, U_ol, P_ol, K_ol = solver_siro.get_sol()
P_ol = [ (p + p.T) / 2 for p in P_ol ]

plotPsiControlThrustOverTime(params, X_ol, U_ol)
if save_plots:
    plt.tight_layout()
    plt.savefig(plot_folder + 'kite_ocp_ol_rest.pdf')
plotKitePositionInAngleSpace(params, X_ol, U_ol, P=P_ol)
if save_plots:
    plt.tight_layout()
    plt.savefig(plot_folder + 'kite_ocp_ol_pos.pdf')


# %% save results
if save_results:

    today_str = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    outfile = res_folder + 'kite_res_' + today_str

    res_dict = {}
    res_dict['sigma'] = sigma
    res_dict['params'] = params
    res_dict['traj_nom'] = {}
    res_dict['traj_nom']['X'] = Xnom
    res_dict['traj_nom']['U'] = Unom
    res_dict['traj_rol'] = {}
    res_dict['traj_rol']['X'] = X_ol
    res_dict['traj_rol']['U'] = U_ol
    res_dict['traj_rol']['P'] = P_ol
    res_dict['traj_rcl'] = {}
    res_dict['traj_rcl']['X'] = X_siro
    res_dict['traj_rcl']['U'] = U_siro
    res_dict['traj_rcl']['P'] = P_siro
    res_dict['traj_rcl']['K'] = K_siro
    res_dict['SIRO_hist'] = it_hist_SIRO
    np.save(outfile, res_dict)

