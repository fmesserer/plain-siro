import numpy as np
import casadi as ca

def ode(x, u, w, params=None):
    '''
    ode of system
    '''
    if params is None:
        params = get_params()

    return ca.vertcat(kite_model_simple(x, u, w, params))

def dyn_rk4(x, u, w, params=None):
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
    # evaluate cost at nominal traj
    cost = -kite_thrust(x, u, params)
    # average thrust contribution
    cost /= params['N']

    return cost

def cost_end(x, params=None):
    # no terminal cost (only constraint)
    return 0

def ineq_constr_stage(x, u, params=None):
    """
    stage constr function h(x,u) <= 0
    """
    if params is None:
        params = get_params()

    h = []
    # minimal height
    h.append( params['hmin']  - kite_height( x, params["L"]) )
    # bounded controls
    h.append( -u + params['umin'] ) 
    h.append(  u - params['umax']  ) 
    return h

def ineq_constr_end(x, params=None):
    """
    end constr function h(x_N) <= 0
    """
    if params is None:
        params = get_params()

    h = []
    # minimal height
    h.append( params['hmin']  - kite_height( x, params["L"]) )
    return h

def get_params():

    params = dict()
    params['N'] = 80                    # time steps
    # params['N'] = 40                    # time steps
    params['T'] = params['N'] * .3      # cont time horizon
    params['nx'] = 3                    # state dim
    params['nu'] = 1                    # control dim
    params['nw'] = 4                    # noise dim
    
    # physical parameters
    params["E0"] = 5            # glide ratio in absence of steering deflection (non-dimensional)
    params["v0"] = 10           # apparent windspeed (m/s)
    params["ctilde"] = 0.028    # coefficient for glide ratio in dependence of steering angle (non-dimensional)
    params["rho"] = 1           # air density (kg / m^3)
    params["L"] = 400           # tether length (m)
    params["A"] = 300           # kite area (m^2)
    params["beta"] = 0          # related to angle between wind and boat (rad)

    # design parameters
    params["hmin"] = 100        # minimal height (m)
    params["umin"] = -10        # lower control constraint (non-dimensional)
    params["umax"] = 10         # upper control constraint (non-dimensional)
    
    # initial state
    rad_per_deg = np.pi / 180   # conversion factor degree to rad
    theta0 = 20 * rad_per_deg
    phi0 = 30 * rad_per_deg
    psi0 = 0
    params["x0"] = np.array([theta0, phi0, psi0])  # initial state (rad)

    return params

def kite_model_simple(x, u, w, params):
    """
    simple model of the kite,
    
    output: xdot
    """
    nx = params['nx']
    nw = params['nw']
    # states
    theta = x[0]
    # phi  = x[1]
    psi = x[2]

    # parameters
    E0 = params["E0"] 
    v0 = params["v0"]
    ctilde = params["ctilde"]
    L = params["L"]

    # add noise
    v0 += w[nw-1]

    # intermediate 
    E = glide_ratio(E0, ctilde, u)
    va = v0 * E * np.cos(theta)

    # ode
    theta_dot = va /  L * (np.cos(psi) - np.tan(theta) / E )
    phi_dot = - va * np.sin(psi) / ( L * np.sin(theta))
    psi_dot = va / L * u + phi_dot * np.cos(theta)

    xdot = ca.vertcat(theta_dot, phi_dot, psi_dot)

    xdot += w[:nx]

    return xdot

def glide_ratio(E0, ctilde, u):
    """
    glide ratio corrected for steering deflections
    input:
        E0      glide ratio without steering deflection
        ctilde  coefficient
        u       steering deflection (control)
    """

    return E0 - ctilde * u **2

def kite_height(x, L):
    """
    height of kite given state x = (phi, theta, psi) and tether length L
    """
    theta = x[0]
    phi = x[1]
    return L * np.sin( theta ) * np.cos(phi)
 

def kite_thrust(x, u, params):
    """
    thrust currently obtained by kite given state x = (phi, theta, psi)
    and control u
    """
    nx = params['nx']
    theta = x[0]
    phi = x[1]

    v0 = params["v0"]
    A = params["A"]
    rho = params["rho"]
    beta = params["beta"]
    PD = rho * v0 ** 2 / 2
    E = glide_ratio(params["E0"], params["ctilde"], u)

    TF = PD * A * np.cos(theta)**2 * (E + 1) * np.sqrt(E**2 + 1) * \
                ( np.cos(theta) * np.cos(beta) + np.sin(theta) * np.sin(beta) * np.sin(phi) )

    return TF