import numpy as np
import matplotlib.pylab as plt

from solverNominalMPC import solverNominalMPC
from solverClosedLoopRMPC_SIRO import solverClosedLoopRMPC_SIRO
from solverOpenLoopRMPC import solverOpenLoopRMPC

from robot2D_simple_probdef import dyn_discr, cost_stage, cost_end, get_params, ineq_constr_stage, ineq_constr_end
from robot2D_simple_plotutils import plotTrajectoryInSpace, plotTrajectoryInTime

params = get_params()

def dyn_discr_nom(x, u, params):
    return dyn_discr(x, u, 0, params)

eps = 1e-1                          # scale noise uncertainty
x0 = [9, 10, 0, 0]                  # initial state
P0 = 1e-2 * np.eye(params['nx'])       # initial state uncertainty ellipsoid
W = eps**2 * np.eye(params['nw'])   # noise uncertainty ellipsoid

# nominal
solver_nom = solverNominalMPC(dyn_discr_nom, cost_stage, cost_end, params, ineq_constr_stage, ineq_constr_end  )
solver_nom.create_solver()
solver_nom.set_value_x0(x0)
solver_nom.solve()
Xnom, Unom = solver_nom.get_sol()
plotTrajectoryInTime(params, Xnom, Unom)
plt.suptitle('nominal')
plotTrajectoryInSpace(params, Xnom)
plt.title('nominal')


# SIRO 
solver_siro = solverClosedLoopRMPC_SIRO(dyn_discr, cost_stage, cost_end, params, ineq_constr_stage, ineq_constr_end  )
# solver_siro.regu_riccatiR = 1e0
# solver_siro.regu_riccatiQ = 1e-5
# solver_siro.closed_loop = False
solver_siro.create_solver()
solver_siro.set_value_all(x0, P0, W)
solver_siro.initialize_to_nominal()
solver_siro.solve()
X_siro, U_siro, P_siro, K_siro = solver_siro.get_sol()
P_siro = [ (p + p.T) / 2 for p in P_siro ]
plotTrajectoryInTime(params, X_siro, U_siro)
plt.suptitle('closed loop robust')
plotTrajectoryInSpace(params, X_siro, P=P_siro)
plt.title('closed loop robust')


# solve again, but open loop
solver_siro.closed_loop = False
solver_siro.solve()
X_siro_ol, U_siro_ol, P_siro_ol, K_siro_ol = solver_siro.get_sol()
P_siro_ol = [ (p + p.T) / 2 for p in P_siro_ol ]
plotTrajectoryInTime(params, X_siro_ol, U_siro_ol)
plt.suptitle('open loop robust')
plotTrajectoryInSpace(params, X_siro_ol, P=P_siro_ol)
plt.title('open loop robust')


# plt.show()


#%% open loop with IPOPT

solver_ol = solverOpenLoopRMPC(dyn_discr, cost_stage, cost_end, params, ineq_constr_stage, ineq_constr_end  )
solver_ol.create_solver()
solver_ol.set_value_all(x0, P0, W)
solver_ol.set_initial_all(Xnom, Unom, P=P0)
solver_ol.solve()
X_ol, U_ol, P_ol = solver_ol.get_sol()
P_ol = [ (p + p.T) / 2 for p in P_ol ]

plotTrajectoryInTime(params, X_ol, U_ol)
plt.suptitle('open loop robust')
plotTrajectoryInSpace(params, X_ol, P=P_ol)
plt.title('open loop robust')
plt.show()
