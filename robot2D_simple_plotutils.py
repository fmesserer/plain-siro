import numpy as np
import matplotlib.pylab as plt

from plotutils import ellipsoid_surface_2D

def plotTrajectoryInTime(params, X, U):
    plt.figure()
    plt.subplot(311)
    plt.plot(X[0,:], label='r_x')
    plt.plot(X[1,:], label='r_y')
    plt.xlabel('discrete time k')
    plt.ylabel('position r')
    plt.legend()

    plt.subplot(312)
    plt.plot(X[2,:], label='v_x')
    plt.plot(X[3,:], label='v_y')
    plt.xlabel('discrete time k')
    plt.ylabel('velocity v')
    plt.legend()

    plt.subplot(313)
    plt.plot(U[0,:], label='u_x')
    plt.plot(U[1,:], label='u_y')
    plt.xlabel('discrete time k')
    plt.ylabel('control u')
    plt.legend()

def plotTrajectoryInSpace(params, X, P=None):

    N = X.shape[1] - 1

    theta = np.linspace(0, 2*np.pi, 100)
    c = params['obstacle_center']
    R = params['obstacle_radius']

    if P is not None:
        P_surf = [ X[:2,k][:,None] + ellipsoid_surface_2D(P[k][:2,:2]) for k in range(N+1) ]

    plt.figure()
    plt.plot(X[0, :], X[1,:], 'bx')
    if P is not None:
        for p in P_surf:
            plt.plot(p[0, :], p[1,:], 'b')
    plt.plot( c[0] + R * np.sin(theta), c[1] + R * np.cos(theta), 'r' )
    plt.plot( [params['rxmin']] * 2, [-1, 11 ], 'r' )
    plt.plot( [-1, 11 ], [params['rymin']] * 2, 'r' )
    plt.axis('equal')
    plt.xlabel(r'$r_x$')
    plt.ylabel(r'$r_y$')