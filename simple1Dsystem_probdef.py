def ode(x, u, w, params=None):
    '''
    ode of system
    '''
    return x**3 - u + w


def dyn_discr(x, u, w, params=None):
    '''
    discretized dynamics
    '''

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
    return 10 * x**2 + u**2


def cost_end(x, params=None):
    return 10 * x ** 2


def ineq_constr_stage(x, u, params=None):
    """
    stage constr function h(x,u) <= 0
    """
    if params is None:
        params = get_params()
    h = []
    h.append(params['xmin'] - x)
    h.append(  params['umin'] - u )
    h.append( -params['umax'] + u )
    return h


def ineq_constr_end(x, params=None):
    """
    end constr function h(x_N) <= 0
    """
    if params is None:
        params = get_params()
    # x >= x_min
    return params['xmin'] - x, 


def get_params():
    """
    returns dictionary of parameters defining the system and OCP
    """
    params = dict()
    params['T'] = 1.5                   # cont time horizon
    params['N'] = 30                    # time steps
    params['nx'] = 1                    # state dim
    params['nu'] = 1                    # control dim
    params['nw'] = 1                    # noise dim    
    params['umin'] = -1                 # control box constr, min
    params['umax'] = 1                  # control box constr, max
    params['xmin'] = .2                 # distance from origin
    return params
