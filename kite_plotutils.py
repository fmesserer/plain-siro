import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from kite_probdef import kite_thrust
from plotutils import ellipsoid_surface_2D

color_palette = sns.color_palette('muted')
color_1 = color_palette[0]
color_2 = color_palette[3]

# conversion from radian to degree
rad_per_deg = np.pi / 180


def plotKitePositionInAngleSpace(params, X, U, P=None, title=''):
    """
    plots kite position in angle (i.e. phi-theta) space.
    X[0, :] contains the time series of theta (in rad)
    X[1, :] contains the time series of phi (in rad)
    P = list of uncertainty ellipsoids
    """

    hmin = params['hmin']
    L = params['L']
    N = X.shape[1] - 1
    theta = X[0,:] / rad_per_deg
    phi = X[1,:] / rad_per_deg

    if P is not None:
        P_surf = [ X[:2,k][:,None] / rad_per_deg + ellipsoid_surface_2D(P[k][:2,:2]) / rad_per_deg for k in range(N+1) ]
    
    # obtained thrust
    TF = [kite_thrust(X[:, k], U[k], params) / 1000 for k in range(N-1) ]
    TFavg = np.mean(TF)
    
    plt.figure()
    phi_constr = np.linspace(-1, 1, 100) * 70 
    theta_constr = np.arcsin( hmin / L / np.cos(phi_constr * rad_per_deg) ) / rad_per_deg
    plt.plot( phi_constr, theta_constr, '-', color=color_2, label=r'height constraint')

    plt.plot( phi[0], theta[0], 'x', color=color_1, label= 'initial state')
    if P is not None:
        for p in P_surf:
            plt.plot(p[1,:], p[0,:], color=color_1, lw=1)
            # plt.plot(p[1,:], p[0,:], 'b', label='trajectory')
            plt.fill(p[1,:], p[0,:], color=color_1)#, alpha=.4 )
    else:
        plt.plot( phi, theta, '.', color=color_1, label='trajectory', ms=4)

    ylim = list(plt.gca().get_ylim())
    ylim[1] = 45
    plt.gca().set_ylim(ylim)

    # plt.legend()
    title += r'$,\quad\hat T_\mathrm{F} =' + '{:.2f}'.format(TFavg) + '\;\mathrm{kN}$'

    plt.xlabel(r'azimuth angle $\phi$ in  deg')
    plt.ylabel(r"zenith angle $\theta$ in  deg")
    plt.title(title)


def plotPsiControlThrustOverTime(params, X, U):
    """
    input: states in rad
    plots psi and u over time
    """    

    DT = params['T'] / params['N']
    Psi = X[2, :]
    N = Psi.size
    T = np.arange(N) * DT

    # compute thrust
    TF = [kite_thrust(X[:, k], U[k], params) / 1000 for k in range(N-1) ]

    plt.figure()
    ax = plt.subplot(3,1,1)
    plt.plot(T, Psi / rad_per_deg)
    plt.ylabel(r'$\psi$ in deg')
    ax.set_xticklabels([])

    ax = plt.subplot(3,1,2)
    plt.step(T, np.concatenate((U, [np.nan])))
    plt.ylabel('control u')
    ax.set_xticklabels([])

    plt.subplot(3,1,3)
    plt.step(T, TF +  [np.nan])
    plt.ylabel('thrust $T_F$ in kN')
    plt.xlabel('time $t$ in s')


def plotAngleSpaceCompareNom(params, Xlist):

    hmin = params['hmin']
    L = params['L']
    N = Xlist[0].shape[1] - 1
    
    plt.figure()
    for X in Xlist:
        theta = X[0,:] / rad_per_deg
        phi = X[1,:] / rad_per_deg
        plt.plot( phi, theta)

    phi_constr = np.linspace(-1, 1, 100) * 70 
    theta_constr = np.arcsin( hmin / L / np.cos(phi_constr * rad_per_deg) ) / rad_per_deg
    plt.plot( phi_constr, theta_constr, 'r', label=r'$h_{\min}$ constraint')
    plt.legend()
    plt.xlabel(r'$\phi$ in  deg')
    plt.ylabel(r"$\theta$ in  deg")

