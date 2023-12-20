import numpy as np
import matplotlib.pylab as plt

def plotContractionCompare(it_hists, ls=None, labels=None):

    norm_order = 2
    n = len(it_hists)
    if ls is None:
        ls = ['-'] * n
    if labels is None:
        labels = [''] * n

    fig, (ax1, ax2) = plt.subplots(2, 1)

    for i in range(n):
        it_history = it_hists[i]

        y = [ entry['y'] for entry in it_history  ]
        lam = [ entry['lam'] for entry in it_history  ]
        mu = [ entry['mu'] for entry in it_history  ]
        beta = [ entry['beta'] for entry in it_history  ]
        eta = [ entry['eta'] for entry in it_history  ]

        # convergence in z
        z = []
        for k in range(len(y)):
            z.append(np.concatenate((y[k], lam[k], mu[k], beta[k], eta[k])))

        dz = np.linalg.norm(np.array(z[:-1]) - z[-1], axis=1, ord=norm_order)
        dz /= np.linalg.norm(z[-1], ord=norm_order)
        
        # convergence in M
        K = [ entry['K'] for entry in it_history  ]
        M = []
        for K_ in K:
            M.append( np.concatenate( [ Kk[:].flatten() for Kk in K_ ] ) )
        dM = np.linalg.norm(np.array(M[:-1]) - M[-1], axis=1, ord=norm_order)
        dM /= np.linalg.norm( M[-1], ord=norm_order)
        
        # plt.subplot(121)
        ax1.semilogy(dz, ls=ls[i], label=labels[i])
        # plt.subplot(122)
        ax2.semilogy(dM, ls=ls[i], label=labels[i])

    ax1.set_ylabel(r'$\lVert z_k - z_*  \rVert_2$')
    ax1.xaxis.set_ticklabels([])
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.legend()
    
    ax2.set_xlabel('iteration $k$')
    ax2.set_ylabel(r'$\lVert M_k - M_*  \rVert_2$')
    ax2.autoscale(enable=True, axis='x', tight=True)

