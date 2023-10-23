import numpy as np
import matplotlib.pylab as plt

from solverNominalMPC import solverNominalMPC
from solverClosedLoopRMPC_SIRO import solverClosedLoopRMPC_SIRO
from solverOpenLoopRMPC import solverOpenLoopRMPC
from simple1Dsystem_probdef import dyn_discr, cost_stage, cost_end, get_params, ineq_constr_stage, ineq_constr_end
from simple1Dsystem_plotutils import plotTrajectory

# parameters defining system and OCP
params = get_params()

x0 = 0.4                # initial state
P0bar = 1e-2**2         # initial state uncertainty ellipsoid
W = .1**2               # noise ellipsoid

def dyn_discr_nom(x, u, params):
    '''
    nominal dynamics
    '''
    return dyn_discr(x, u, 0, params)


# %% solver of nominal OCP problem
solver_nom = solverNominalMPC(dyn_discr_nom, cost_stage, cost_end, params, ineq_constr_stage, ineq_constr_end  )
solver_nom.create_solver()
solver_nom.set_value_x0(x0)
solver_nom.solve()
Xnom, Unom = solver_nom.get_sol()
plotTrajectory(params, Xnom, Unom, title='nominal')


#%%solver for open loop robust OCP

# obtained by enforcing no feedback, K = 0, in closed loop problem
solver_ol = solverClosedLoopRMPC_SIRO(dyn_discr, cost_stage, cost_end, params, ineq_constr_stage, ineq_constr_end  )
# solver_ol.regu_riccatiR = 1e0
solver_ol.closed_loop = False
solver_ol.create_solver()
solver_ol.set_value_all(x0, P0bar, W)
solver_ol.initialize_to_nominal()
solver_ol.solve()
X_ol, U_ol, P_ol, K_ol = solver_ol.get_sol()
P_ol = [P_.squeeze() for P_ in P_ol]
K_ol[0] = np.array([[0]])
K_ol = [K_.squeeze() for K_ in K_ol]

plotTrajectory(params, X_ol, U_ol, P=P_ol, K=K_ol, title='open loop robust')

#%%solver for closed loop robust OCP

solver_cl = solverClosedLoopRMPC_SIRO(dyn_discr, cost_stage, cost_end, params, ineq_constr_stage, ineq_constr_end  )
# solver_cl.regu_riccatiR = 1e0
solver_cl.create_solver()
solver_cl.set_value_all(x0, P0bar, W)
solver_cl.initialize_to_nominal()
solver_cl.solve()
X_cl, U_cl, P_cl, K_cl = solver_cl.get_sol()
P_cl = [P_.squeeze() for P_ in P_cl]
K_cl[0] = np.array([[0]])
K_cl = [K_.squeeze() for K_ in K_cl]

plotTrajectory(params, X_cl, U_cl, P=P_cl, K=K_cl, title='closed loop')
# plt.show()


#%% open loop with IPOPT

solver_ol2 = solverOpenLoopRMPC(dyn_discr, cost_stage, cost_end, params, ineq_constr_stage, ineq_constr_end  )
solver_ol2.create_solver()
solver_ol2.set_value_all(x0, P0bar, W)
solver_ol2.set_initial_all(Xnom, Unom, P=P0bar)
solver_ol2.solve()
X_ol2, U_ol2, P_ol2 = solver_ol2.get_sol()
# P_ol2 = [P_.squeeze() for P_ in P_ol2]

plotTrajectory(params, X_ol2, U_ol2, P=P_ol2, title='open loop w IPOPT')
plt.show()



