#%%
import numpy as np
import matplotlib.pylab as plt
import casadi as ca

from solverNominalMPC import solverNominalMPC
from solverClosedLoopRMPC_SIRO import solverClosedLoopRMPC_SIRO

from robot2D_simple_probdef import dyn_discr, cost_stage, cost_end, get_params, ineq_constr_stage, ineq_constr_end
from robot2D_simple_plotutils import plotTrajectoryInSpace, plotTrajectoryInTime

params = get_params()
params['N'] = 10
params['T'] = params['N'] / 3

N = params['N']
nx = params['nx']
nu = params['nu']
nw = params['nw']

def dyn_discr_nom(x, u, params):
    return dyn_discr(x, u, 0, params)

eps = 1e-1
x0 = [9, 10, 0, 0]                  # initial state
P0 = 0 * np.eye(params['nx'])
W = eps**2 * np.eye(params['nw'])

# mpc_mode = 'nominal'
mpc_mode = 'SIRO'
# closed_loop_robust = False
closed_loop_robust = True

# nominal
if mpc_mode == 'nominal':
    solver_nom = solverNominalMPC(dyn_discr_nom, cost_stage, cost_end, params, ineq_constr_stage, ineq_constr_end  )
    solver_nom.create_solver()
    # solver_nom.set_initial_X(X0)
    solver_nom.set_value_x0(x0)
    solver_nom.solve()
    solver_mpc = solver_nom

# SIRO
if mpc_mode == 'SIRO':
    solver_mpc = solverClosedLoopRMPC_SIRO(dyn_discr, cost_stage, cost_end, params, ineq_constr_stage, ineq_constr_end  )
    solver_mpc.closed_loop = closed_loop_robust
    solver_mpc.tol = 1e-4
    solver_mpc.create_solver()
    solver_mpc.set_value_all(x0, P0, W)
    solver_mpc.initialize_to_nominal()

# %% closed loop MPC
DT_sim = params['T'] / params['N']
N_sim = 30
T_sim = N_sim * DT_sim

def simulator(x, u, w):
    return dyn_discr(x, u, w)

#%%
X_sim = np.zeros((nx, N_sim))   * np.nan
U_sim = np.zeros((nu, N_sim-1)) * np.nan
TF_sim = np.zeros((1, N_sim-1))
X_sim[:, 0] = x0


for k in range(N_sim-1):

    if mpc_mode == 'nominal':
        X, U = solver_mpc.get_sol()
        solver_mpc.set_initial_all(X=X, U=U)

    xcurrent = X_sim[:, k]
    # solve mpc problem
    solver_mpc.set_value_x0(xcurrent)

    # TODO: USE THIS TO CHANGE UNCERTAINTY PARAMETERS
    # solver_mpc.set_value_all(x0, P0, W)

    # if k == 4:
    #     import pdb; pdb.set_trace()
    solver_mpc.solve()

    if mpc_mode == 'nominal' and not solver_mpc.solver.stats()['success']:
        print("Solve not successfull")
        break

    if mpc_mode == 'SIRO' and not solver_mpc.success:
        print("Solve not successfull")
        break


    # mpc control
    u0 = solver_mpc.get_value_u0()

    xplus = simulator(xcurrent, u0, 0)

    # save values
    U_sim[:, k] = u0
    X_sim[:, k+1] = xplus.full().flatten()

plotTrajectoryInTime(params, X_sim, U_sim)
plotTrajectoryInSpace(params, X_sim)
plt.show()