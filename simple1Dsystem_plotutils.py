import numpy as np
import matplotlib.pylab as plt


def plotTrajectory(params, X, U, P=None, K=None, title=None):

    N = X.shape[0] - 1


    if P is not None:
        tube_lower = [ X[k] - np.sqrt(P[k]) for k in range(N+1) ]
        tube_upper = [ X[k] + np.sqrt(P[k]) for k in range(N+1) ]
    
    if K is not None:
        Uellipsoid = [ Kk * Pk * Kk for Kk, Pk in zip(K, P) ]
        Utube_lower = [ uk - np.sqrt(Uk) for uk, Uk in zip(U, Uellipsoid) ]
        Utube_upper = [ uk + np.sqrt(Uk) for uk, Uk in zip(U, Uellipsoid) ]


    plt.figure()
    plt.subplot(211)
    if title is not None:
        plt.title(title)
    plt.plot(X, 'b')
    if P is not None:
        plt.fill_between(list(range(N+1)), tube_lower, tube_upper, alpha=.3, facecolor='b')
    plt.plot([0, N],  2 * [params['xmin']], 'r')
    # plt.xlabel('discrete time k')
    plt.gca().xaxis.set_ticklabels([])
    plt.ylabel('state x')

    plt.subplot(212)
    plt.plot(U, 'b')
    if K is not None:
        # import pdb; pdb.set_trace()
        plt.fill_between(list(range(N)), Utube_lower, Utube_upper, alpha=.3, facecolor='b')
    plt.xlabel('discrete time k')
    plt.ylabel('control u')

    # return plt.gcf()