import numpy as np
import casadi as ca
# from utils import rk4_step

def ode(x, u, w, params=None):
    '''
    ode of system
    '''
    return ca.vertcat(x[2:], u) + w


def dyn_discr(x, u, w, params=None):
    if params is None:
        params = get_params()

    h = params['T']/params['N']
    k1       = ode(x,            u, w, params=params)
    k2       = ode(x + h/2 * k1, u, w, params=params)
    k3       = ode(x + h/2 * k2, u, w, params=params)
    k4       = ode(x + h * k3,   u, w, params=params)
    return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)


def cost_stage(x, u, params=None):
    if params is None:
        params = get_params()
    return 10 * x[:2].T @ x[:2] + u.T @ u


def cost_end(x, params=None):
    return 10 * x[:2].T @ x[:2]


def ineq_constr_stage(x, u, params=None):
    """
    stage constr function h(x,u) <= 0
    """
    if params is None:
        params = get_params()

    c = params['obstacle_center']
    R = params['obstacle_radius']

    h1 = R**2 - (x[0] - c[0])**2 - (x[1] - c[1])**2
    h2 = u.T @ u - params['umax']**2
    h3 = params['rxmin'] - x[0]
    h4 = params['rymin'] - x[1]
    return h1, h2, h3, h4
    # return h1, h2,
    # return h2,


def ineq_constr_end(x, params=None):
    """
    end constr function h(x_N) <= 0
    """
    if params is None:
        params = get_params()

    c = params['obstacle_center']
    R = params['obstacle_radius']
    h1 = R**2 - (x[0] - c[0])**2 - (x[1] - c[1])**2
    h3 = params['rxmin'] - x[0]
    h4 = params['rymin'] - x[1]

    return h1, h3, h4


def get_params():

    params = dict()
    params['T'] = 10                    # cont time horizon
    params['N'] = 30                    # time steps
    params['nx'] = 4                    # state dim
    params['nu'] = 2                    # control dim
    params['nw'] = 4                    # noise dim
    
    params['umax'] = 3                 # control box constr, max
    params['rxmin'] = -1                # minimal x position
    params['rymin'] = -1                # minimal y position
    params['obstacle_center'] = [5, 5]
    params['obstacle_radius'] = 3
    return params
