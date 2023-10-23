import time
import casadi as ca
import numpy as np
import scipy as sp

class solverClosedLoopRMPC_SIRO:
    def __init__(self, dyn_discr, cost_stage, cost_end, params, 
                    ineq_constr_stage=None, ineq_constr_end=None):

        # algo params
        self.maxit = 100              # maximum number of SIRO iterations
        self.tolKKT = 1e-3            # convergence tolerance KKT conditions
        self.regu_backoff = 1e-6      # regularization backoff 
        self.regu_riccatiR = 0        # regularization riccati recursion
        self.regu_riccatiQ = 0        # regularization riccati recursion
        self.closed_loop = True       # include and optimize over linear feedback laws
        self.checkSOSC = False        # check second order sufficient optimality conditions (SOSC) at solution
        # self.checkSOSC = True
        self.print_timings = True     # whether timings are printed (per iterations)

        self.params = params
        self.nx = params['nx']
        self.nu = params['nu']
        self.nw = params['nw']
        self.N = params['N']

        nx = self.nx
        nu = self.nu
        nw = self.nw
        N = self.N

        x = ca.SX.sym('x', nx)
        u = ca.SX.sym('u', nu)
        w = ca.SX.sym('w', nw)

        self.f_discr = ca.Function('f_discr', [x, u, w], [dyn_discr(x, u, w, params)])
        self.l_stage = ca.Function('l_stage', [x, u], [cost_stage(x, u, params)])
        self.l_end = ca.Function('l_end', [x], [cost_end(x, params)])

        self.exist_constr_stage = ineq_constr_stage is not None
        self.exist_constr_end = ineq_constr_end is not None

        if self.exist_constr_stage:
            self.h_stage = ca.Function('h_stage', [x ,u], [ca.vertcat(*ineq_constr_stage(x, u, params))])
            nh = self.h_stage.size1_out(0)
            self.nh = nh

        if self.exist_constr_end:
            self.h_end = ca.Function('h_end', [x], [ca.vertcat(*ineq_constr_end(x, params))])
            nhN = self.h_end.size1_out(0)
            self.nhN = nhN

        # uncertainty part
        P = ca.SX.sym('P', nx, nx)                          # state deviation elipsoid matrix
        W = ca.SX.sym('W', nw, nw)                          # noise eliopsoid matrix
        K = ca.SX.sym('K', nu, nx)                          # feedback matrix
        xs = ca.SX.sym('xs', nx)                            # stochastic state (helper)
        us = u + K @ (xs - x)                               # stochastic control
        
        A = ca.jacobian( self.f_discr(x, u, w), x )         # dynamics state sensitivity
        B = ca.jacobian( self.f_discr(x, u, w), u )         # dynamics control sensitivity
        Gamma = ca.jacobian( self.f_discr(x, u, w), w)      # dynamics noise sensitivity

        self.df_dx = ca.Function('df_dx', [x, u, w], [A])
        self.df_du = ca.Function('df_du', [x, u, w], [B])

        # uncertainty dynamics
        lyap_dyn_ = ca.substitute( (A + B @ K) @ P @ (A + B @ K).T + Gamma @ W @ Gamma.T , w, 0)
        self.lyap_dyn = ca.Function('lyap_dyn', [x, u, P, K, W], [ lyap_dyn_ ]) 

        if self.exist_constr_stage:
            # tightened constr
            self.h_stage_tight = ca.Function('h_stage_tight', [x, u], [ self.h_stage(x, u) ])
            # uncertainty orthogonal to constr 
            H = ca.jacobian(self.h_stage(xs, us), xs)
            H = ca.substitute(H, xs, x)         # eval at nominal traj
            h_ = [  H[i,:] @ P @ H[i,:].T  for i in range(nh)  ]
            self.h_stage_unc = ca.Function('h_stage_unc', [x, u, P, K],  [ca.vertcat(*h_)])
            self.dh_dxu = ca.Function('dh_dxu', [x, u], [ca.jacobian(self.h_stage(x, u), ca.vertcat(x, u))])

        if self.exist_constr_end:
            # tightened constr
            self.h_end_tight = ca.Function('h_end_tight', [x], [ self.h_end(x) ])
            # uncertainty orthogonal to constr
            H = ca.jacobian(self.h_end(x), x)
            h_ = [ H[i,:] @ P @ H[i,:].T  for i in range(nhN) ]
            self.h_end_unc = ca.Function('h_end_unc', [x, P], [ca.vertcat(*h_)])
            self.dhN_dx = ca.Function('dhN_dx', [x], [H])

    def create_solver(self):
        
        nx = self.nx
        nu = self.nu
        N = self.N

        # decision vars
        X = ca.SX.sym('X', nx, N+1)
        U = ca.SX.sym('U', nu, N)
        y = ca.vertcat(ca.vec(X), ca.vec(U))
        # lby = [- ca.inf] * y.shape[0]
        # uby = [ca.inf] * y.shape[0]

        # params
        x0 = ca.SX.sym('x0', nx)                    # initial state
        grad_corr = ca.SX.sym('c', y.shape[0] )      # gradient perturb param
        pars = ca.vertcat(x0, grad_corr)
        self.pars_val = np.nan * ca.DM.ones(pars.shape[0])

        obj = 0
        for k in range(N):
            obj += self.l_stage(X[:,k], U[:,k])
        obj += self.l_end(X[:, -1])

        # gradient correction
        obj += grad_corr.T @ y

        constr_equa = []
        constr_ineq = []

        constr_equa.append( x0 - X[:,0] )
        for k in range(N):
            constr_equa.append( self.f_discr(X[:, k], U[:, k], 0) - X[:, k+1] )
            if self.exist_constr_stage:
                    constr_ineq.append(self.h_stage(X[:,k], U[:,k]))
        if self.exist_constr_end:
            constr_ineq.append(self.h_end(X[:,-1]))

        constr_equa = ca.vertcat(*constr_equa)
        constr_ineq = ca.vertcat(*constr_ineq)
        g = ca.vertcat(constr_equa, constr_ineq)
        lbg = ca.vertcat( [0] * constr_equa.shape[0], [-np.inf] * constr_ineq.shape[0]  )
        ubg = 0

        # idxes of corresponding multipliers
        idx_lam = np.arange(constr_equa.shape[0])
        idx_mu = np.arange(constr_equa.shape[0], g.shape[0] )

        nlp = {}
        nlp['x'] = y
        nlp['p'] = pars
        nlp['f'] = obj
        nlp['g'] = g

        opts = {}
        opts['ipopt'] = {}
        # opts['ipopt']['linear_solver'] = 'ma27'
        # opts['ipopt']['linear_solver'] = 'ma57'
        # opts['ipopt']['linear_solver'] = 'ma77'
        # opts['ipopt']['linear_solver'] = 'ma86'
        # opts['ipopt']['linear_solver'] = 'ma97'

        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        self.solver_nom = solver
        self.solver_nom_initialized = False
        self.ocp_nom = nlp
        self.X = X
        self.U = U
        self.idx_mu = idx_mu
        self.idx_lam = idx_lam
        self.lbg = lbg
        self.ubg = ubg

        self.KKT_prepare(prepare_SOSC=self.checkSOSC)
        self.reset_iteration_history()


    def solve_nom(self):
        '''
        solve nominal OCP (given a fixed backoff and gradient correction)
        '''
        sol = self.solver_nom( \
            x0  = self.init_guess, \
            p   = self.pars_val, \
            ubg = self.ubg, \
            lbg = self.lbg, \
        )

        Xopt = ca.DM(ca.substitute(self.X, self.ocp_nom['x'], sol['x'])).full().squeeze()
        Uopt = ca.DM(ca.substitute(self.U, self.ocp_nom['x'], sol['x'])).full().squeeze()
        
        self.sol = sol
        self.Xbar = Xopt
        self.Ubar = Uopt


    def initialize_to_nominal(self, X=None, U=None, b=None):
        """ initialize solver by computing nominal solution """

        # initialize primals
        self.set_initial_all(X, U)
        # set parameters
        self.set_value_backoff(b)       # backoff
        self.set_value_gradcorr(0)
        self.solve_nom()


    def set_initial_all(self, X=None, U=None):
        '''
        set initial guess of X and U (defaults to zero)
        '''

        if X is None:
            X = ca.DM.zeros(self.X.shape)
        if U is None:
            U = ca.DM.zeros(self.U.shape)

        init_guess = []
        init_guess.append(ca.vec(X))
        init_guess.append(ca.vec(U))
        self.init_guess = ca.vertcat(*init_guess)


    def set_value_all(self, x0, P0, W):
        '''
        set values of parameters
        x0: initial state
        P0: initial state uncertainty ellipsoid
        W:  noise uncertainty ellipsoid
        '''

        self.set_value_x0(x0)
        self.P0bar = ca.DM(P0)
        self.W = W


    def set_value_x0(self, x0):
        '''
        set value of initial state x0
        '''
        self.x0bar = x0
        self.pars_val[:self.nx] = x0


    def set_value_backoff(self, b=None):
        '''
        set value of constraint backoff
        '''
        if b is None:
            b = np.sqrt(self.regu_backoff)
        if isinstance(b, (int, float)):
            b = [ b * np.ones(self.nh) ]  * self.N 
            b.append( b[0][0] * np.ones(self.nhN))
        self.bbar = b
        bvec = ca.vertcat(*b)
        self.ubg = ca.vertcat( ca.DM.zeros(self.idx_lam.shape[0]), -bvec)


    def set_value_gradcorr(self, g):
        '''
        set value of gradient correction
        '''
        if isinstance(g, (int, float)):
            g *= np.ones(self.ocp_nom['x'].shape[0])
        self.pars_val[self.nx:] = g


    def get_nominal_traj(self):
        '''
        get current nominal trajectory
        '''
        return np.copy(self.Xbar).squeeze(), np.copy(self.Ubar).squeeze()


    def get_value_u0(self):
        '''
        get value of first control of solution (MPC law)
        '''
        return self.Ubar[:, 0]


    def get_duals_as_lists(self):
        duals_g = self.sol['lam_g']
        lam_ = ca.vertcat( duals_g[self.idx_lam])
        mu_ = duals_g[self.idx_mu]

        lam = ca.reshape(lam_, (self.nx, self.N+1))
        lam = [lam[:, k] for k in range(self.N+1)]
        mu = ca.reshape(mu_[:-self.nhN], (self.nh, self.N))
        mu = [ mu[:,k] for k in range(self.N)]
        mu.append(mu_[-self.nhN:])

        lam = [l.full().flatten() for l in lam]
        mu = [m.full().flatten() for m in mu]

        return lam, mu

    def get_sol(self):
        """
        get solution of robust OCP

        returns X [matrix], U [matrix], P [list of matrices], K [list of matrices]
        """
        P = [p.full() for p in self.P  ]
        K = [0] + [k.full() for k in self.K[1:]  ]

        return np.copy(self.Xbar.squeeze()), np.copy(self.Ubar.squeeze()), P, K

    def solve(self):
        """
        solve robust OCP
        """
        nx = self.nx
        nu = self.nu
        N = self.N
        maxit = self.maxit
        Xbar = self.Xbar
        Ubar = self.Ubar
        bbar = self.bbar

        t0 = time.perf_counter()

        for it in range(maxit):

            Xbar = np.copy(self.Xbar)
            Ubar = np.copy(self.Ubar)

            t1 = time.perf_counter()

            print('iteration: ', it)

            # expand back to 2 dim if necessary (since sol.value flattens)
            if Xbar.ndim == 1:
                Xbar = Xbar[None, :]
            if Ubar.ndim == 1:
                Ubar = Ubar[None, :]

            lambar, mubar = self.get_duals_as_lists()

            # compute lambda from mu
            etabar = [ mu_to_eta(mu_, b_) for mu_, b_ in zip(mubar, bbar)  ]
            # compute uncertainty part
            Abar = [ self.df_dx(Xbar[:, k], Ubar[:, k], 0) for k in range(N) ]
            Bbar = [ self.df_du(Xbar[:, k], Ubar[:, k], 0) for k in range(N) ]

            t1_0 = time.perf_counter()

            # stage cost matrices
            if self.exist_constr_stage:
                Jac_h = [ self.dh_dxu(Xbar[:,k], Ubar[:, k])  for k in range(N)]
                Hbar = [ Jac_h[k].T @ np.diag(etabar[k]) @ Jac_h[k] for k in range(N) ]
            else:
                Hbar = [np.zeros((nx+nu, nx+nu))] * N
            if self.exist_constr_end:
                Jac_h.append( self.dhN_dx(Xbar[:,-1]))
                Hbar.append(Jac_h[-1].T @ np.diag(etabar[-1]) @ Jac_h[-1])
            else:
                Hbar.append(np.zeros(nx, nx))

            Qbar = [ Hbar[k][:nx, :nx] for k in range(N+1) ]
            Rbar = [ Hbar[k][nx:, nx:] for k in range(N) ]
            Sbar = [ Hbar[k][nx:, :nx] for k in range(N) ]


            # regularize
            for k in range(N):
                Rbar[k] += self.regu_riccatiR * np.eye(nu)

            for k in range(N+1):
                Qbar[k] += self.regu_riccatiQ * np.eye(nx)

            t1_1 = time.perf_counter()

            # # compute optimal feedbacks for current traj
            if self.closed_loop:
                K, _ = riccati_recursion(Qbar, Rbar, Sbar, Abar, Bbar)
                # no feedback on initial control
                K[0] = 0
            else:
                K = [0] + [ ca.DM.zeros((nu, nx))  ] * (N-1)    # no feedback
            
            # forward lyapunov pass to get uncertainty ellipsoids
            t1_2 = time.perf_counter()
            
            P = [self.P0bar]
            for k in range(N):
                P += [self.lyap_dyn(Xbar[:, k], Ubar[:, k], P[-1], K[k], self.W )]       

            t1_3 = time.perf_counter()

            # compute backoffs
            betabar = []
            if self.exist_constr_stage:
                for k in range(N):
                    betabar.append( self.h_stage_unc(Xbar[:,k], Ubar[:,k], P[k], K[k]) )
            if self.exist_constr_end:
                betabar.append( self.h_end_unc(Xbar[:,-1], P[-1]))
            betabar = [b_.full().flatten() for b_ in betabar]

            bbar = [np.sqrt(b_ + self.regu_backoff) for b_ in betabar]
            self.set_value_backoff(bbar)

            # grad cor
            ybar = ca.vertcat(ca.vec(Xbar), ca.vec(Ubar))
            Kvecbar = ca.vertcat(*[ca.vec(Kk) for Kk in K[1:]])
            Gbar = self.func_grad_corr(ybar, Kvecbar, ca.vertcat(*etabar), self.P0bar, self.W)
            self.set_value_gradcorr(Gbar)

            self.write_iteration_history(Xbar, Ubar, betabar, bbar, Gbar, K, P, lambar, mubar, etabar)
            
            # check convergence
            # (check via KKT conditions only implented for closed loop robust)
            if self.closed_loop:
                if self.KKT_check(Xbar, Ubar, K, betabar, lambar, mubar, etabar, self.x0bar, self.P0bar, self.W):
                    break

            # otherwise solve again
            self.set_initial_all(Xbar, Ubar)
            t2 = time.perf_counter()
            self.solve_nom()

            if not self.solver_nom.stats()['success']:
                self.success = False
                print("")
                print("Nominal OCP could not be solved. Exiting")
                print("")
                break

            t3 = time.perf_counter()


            # HACK for open loop OCP convergence (as KKT are written only for closed loop)
            if self.closed_loop is False:
                dXbar = np.max(np.abs (Xbar[:] - self.Xbar[:]))
                dUbar = np.max(np.abs (Ubar[:] - self.Ubar[:]))
                if dXbar <= 1e-5 and dUbar <= 1e-5:
                    break


            t4 = time.perf_counter()

            if self.print_timings:
                print('')
                print('time spent in ... (in s)')
                print('evaluating dyn sensitivities:', t1_0 - t1 )
                print('evaluating cost matrices:', t1_1 - t1_0 )
                print('executing riccati:', t1_2 - t1_1 )
                print('lyapunov forward:', t1_3 - t1_2 )
                print('preparing ocp:', t2 - t1_3)
                print('solving ocp:', t3 - t2)
                print('checking KKT:', t4 -t3 )




        toc = time.perf_counter()

        print("")
        print("exiting after", it+1, "iterations")
        print("total time spent:", (toc - t0) * 1000, "ms")

        self.success = True
        if it+1 == self.maxit:
            self.success = False

        self.P = P
        self.K = K
        self.KKT_check(Xbar, Ubar, K, betabar, lambar, mubar, etabar, self.x0bar, self.P0bar, self.W, printinfo=True, checkSOSC=self.checkSOSC )


    def KKT_prepare(self, prepare_SOSC=False):
        """
        prepares function needed to evaluate KKT conditions of actually solved NLP
        """
        nx = self.nx
        nu = self.nu
        nw = self.nw
        nh = self.nh
        nhN = self.nhN
        N = self.N

        # primal variables ...
        #  ... of nominal prob      (x and y)
        X = ca.SX.sym('X', nx, N+1)
        U = ca.SX.sym('U', nu, N)
        y = ca.vertcat(ca.vec(X), ca.vec(U))
        # ... of uncertainty part (feedback matrices K as vector)
        Kvec = ca.SX.sym('Kvec', nx * nu, N-1)
        M = ca.vec(Kvec)
        K = [0] + [ca.reshape(Kvec[:, k], nu, nx ) for k in range(Kvec.shape[1])]
        # ... coupling variable
        Beta_lst = [ ca.SX.sym('beta_'+str(k), nh)  for k in range(N) ]
        Beta_lst.append( ca.SX.sym('beta_'+str(N), nhN))
        beta = ca.vertcat(*Beta_lst)

        # parameters
        x0bar = ca.SX.sym('x0bar', nx)
        P0bar = ca.SX.sym('P0bar', nx, nx)
        W = ca.SX.sym('W', nw, nw)
        rb = ca.SX.sym('rb')        # regu backoff

        # forward simulate / eliminate P
        P = [P0bar]
        for k in range(N):
            P.append(self.lyap_dyn(X[:,k], U[:,k], P[-1], K[k], W))

        # objective
        f = 0
        for k in range(N):
            f += self.l_stage(X[:,k], U[:,k])
        f += self.l_end(X[:,-1])

        # regularization
        f_regu = 0
        # for k in range(N+1):
        #     f_regu +=  self.regu_riccatiQ * ca.trace(P[k])
        for k in range(1, N):
            f_regu += self.regu_riccatiR * ca.trace(K[k] @ P[k] @ K[k].T)
        # for k in range(1,N):
        #     f_regu += self.regu_riccatiR * ca.trace(K[k] @ K[k].T) range(1, N):

        # dynamics constraint / nominal equality constr
        g = [x0bar - X[:,0]]
        for k in range(N):
            g.append(self.f_discr(X[:, k], U[:,k], 0)  - X[:,k+1])
        g = ca.vertcat(*g)

        # inequality constr with backoff var
        h = []
        for k in range(N):
            h.append( self.h_stage( X[:,k], U[:,k] ) )
        h.append( self.h_end(X[:, -1]) )
        h = ca.vertcat(*h)
        h_beta = h + ca.sqrt(beta + rb)

        # uncertainty part of constr
        H = []
        for k in range(N):
            H.append( self.h_stage_unc( X[:,k], U[:,k], P[k], K[k])  )
        H.append( self.h_end_unc(X[:,-1], P[-1]) )
        H = ca.vertcat(*H)
        H_beta = H - beta

        # Lagrange multipliers
        lam = ca.SX.sym('lam', g.shape[0])
        mu = ca.SX.sym('mu', h.shape[0])
        eta = ca.SX.sym('eta', H.shape[0])

        # Lagrangian
        Lag = f + f_regu + lam.T @ g + mu.T @ h_beta + eta.T @ H_beta
        # KKT condition funcs
        self.KKT_Lag_dy = ca.Function('KKT_Lag_dy', [y, M, lam, mu, eta, x0bar, P0bar, W], [ca.gradient( Lag, y ) ])
        self.KKT_Lag_dM = ca.Function('KKT_Lag_dM', [y, M, eta, x0bar, P0bar, W], [ca.gradient( Lag, M ) ])
        self.KKT_Lag_db = ca.Function('KKT_Lag_db', [beta, mu, eta, rb], [ca.gradient( Lag, beta ) ])
        self.KKT_g = ca.Function('KKT_g', [y, x0bar], [ g ])
        self.KKT_h = ca.Function('KKT_h', [y, beta, rb], [ h_beta ])
        self.KKT_H = ca.Function('KKT_H', [y, M, beta, x0bar, P0bar, W], [ H_beta ])

        if prepare_SOSC:
            self.KKT_Lag_hess = ca.Function('KKT_Lag_hess',  [y, M, beta, lam, mu, eta, x0bar, P0bar, W, rb], [ca.hessian(Lag, ca.vertcat(y, M, beta))[0]] )
            self.KKT_g_jac = ca.Function('KKT_g_jac', [y, x0bar], [ ca.jacobian(g, ca.vertcat(y, M, beta)) ])
            self.KKT_h_jac = ca.Function('KKT_h_jac', [y, beta, rb], [ ca.jacobian(h_beta, ca.vertcat(y, M, beta)) ])
            self.KKT_H_jac = ca.Function('KKT_H_jac', [y, M, beta, x0bar, P0bar, W], [ ca.jacobian(H_beta, ca.vertcat(y, M, beta)) ])
        


        # for gradient correction
        # make function of y only, because of the .reverse() call
        H_func = ca.Function('H_func', [y], [H])
        H_revder = H_func.reverse(1)
        etaT_jacH = H_revder(y, [], eta)
        jac_f_regu = ca.jacobian(f_regu, y)
        grad_corr = jac_f_regu.T + etaT_jacH

        self.func_grad_corr = ca.Function('func_grad_corr', [y, M, eta, P0bar, W], [grad_corr] )


    def KKT_check(self, X, U, K, beta, lam, mu, eta, x0, P0, W, printinfo=False, checkSOSC=False):

        # bring variables into correct shape
        y = ca.vertcat(ca.vec(X), ca.vec(U))
        M = ca.vertcat( *[ ca.vec(Kk) for Kk in K[1:] ] )

        beta = ca.vertcat(*beta)
        lam = ca.vertcat(*lam)
        mu = ca.vertcat(*mu)
        eta = ca.vertcat(*eta)

        # lagrange gradient
        r_dLdy = np.max(np.abs( self.KKT_Lag_dy( y, M, lam, mu, eta, x0, P0, W )  ) )
        r_dLdM = np.max(np.abs( self.KKT_Lag_dM( y, M, eta, x0, P0, W )  ) )
        r_dLdb = np.max(np.abs( self.KKT_Lag_db( beta, mu, eta, self.regu_backoff)  ) )
        # equalities
        r_g = np.max(np.abs( self.KKT_g( y, x0  ) ))
        r_H = np.max(np.abs( self.KKT_H( y, M, beta, x0, P0, W  ) ))
        # inequalities
        h = self.KKT_h( y, beta, self.regu_backoff  ) 
        r_h = np.max( h )
        r_mu = np.min(mu)
        r_comp = np.max(np.abs(mu * h))

        converged = max(r_dLdy, r_dLdM, r_dLdb, r_g, r_H, r_h, -r_mu, r_comp ) <= self.tolKKT

        # print
        if printinfo:
            print('')
            print('KKT residuals:')
            print('dLag_dy:', r_dLdy)
            print('dLag_dM:', r_dLdM)
            print('dLag_db:', r_dLdb)
            print('g:', r_g)
            print('H:', r_H)
            print('h:', r_h)
            print('mu:', r_mu)
            print('mu * h:', r_comp)
            print('')

        if checkSOSC:
            print('')
            tol_act = 1e-4
            # find active ineq
            is_active = np.abs(h) <= tol_act
            is_muzero = np.abs(mu) <= tol_act
            if np.any(np.logical_and(is_active, is_muzero)):
                print('strong complementarity does not hold')
            idx_active = is_active.nonzero()[0]

            # eval hessian and constr jac
            Lag_hess = self.KKT_Lag_hess(y, M, beta, lam, mu, eta, x0, P0, W, self.regu_backoff)
            g_jac = self.KKT_g_jac( y, x0  ) 
            H_jac = self.KKT_H_jac( y, M, beta, x0, P0, W  ) 
            h_jac = self.KKT_h_jac( y, beta, self.regu_backoff  ) 

            jac_act = ca.vertcat(g_jac, H_jac, h_jac[idx_active,:])
            if np.linalg.matrix_rank(jac_act) < jac_act.shape[0]:
                print('LICQ does not hold')
            Z = sp.linalg.null_space(jac_act)
            redH = Z.T @ Lag_hess @ Z
            redH = (redH + redH.T) /2       # symmetrize

            eig = np.linalg.eig(redH)[0]
            if np.min(eig) <= tol_act:
                print('SOSC does not hold')
                print('smallest eigenvalue:', np.min(eig))
            else:
                print('SOSC holds')

        return converged


    def reset_iteration_history(self):
        self.it_hist = []

    def write_iteration_history(self, Xbar, Ubar, beta, b, G, K, P, lam, mu, eta):
        
        info = {}
        info['Xbar'] = Xbar
        info['Ubar'] = Ubar
        info['b'] = ca.vertcat(*b).full().flatten()
        info['beta'] = ca.vertcat(*beta).full().flatten()
        # info['beta'] = info['b']**2 - self.regu_backoff
        info['G'] = G
        info['K'] = [Kk.full() for Kk in K[1:]]
        info['P'] = [Pk.full() for Pk in P]
        info['lam'] = ca.vertcat(*lam).full().flatten()
        info['mu'] = ca.vertcat(*mu).full().flatten()
        info['eta'] = ca.vertcat(*eta).full().flatten()
        info['y'] = self.sol['x'].full().flatten()

        # info['time_prep'] = tictoc[0]
        # info['time_ocp'] = tictoc[1]

        info['solver_nom_stats'] = self.solver_nom.stats()
        info['solver_nom_iter'] = info['solver_nom_stats']['iter_count']
        info['solver_nom_t_wall'] = info['solver_nom_stats']['t_wall_total']
        info['solver_nom_t_proc'] = info['solver_nom_stats']['t_proc_total']

        self.it_hist.append(info)
    
    def get_iteration_history(self):
        return self.it_hist.copy()


def riccati_recursion(Q, R, S, A, B):
    """ riccati recursion """

    N = len(Q) - 1
    # cost to go matrix
    P = [None] * (N+1)
    P[-1] = Q[-1]
    # feedback mat
    K = [None] * N
    nu = R[0].shape[0]

    for k in range(N-1, -1, -1):
        # K[k] = - np.linalg.solve( R[k] + B[k].T @ P[k+1] @ B[k], S[k] + B[k].T @ P[k+1] @ A[k] )
        K[k] = - ca.solve( R[k] + B[k].T @ P[k+1] @ B[k], S[k] + B[k].T @ P[k+1] @ A[k] )
        # K[k] = - ca.pinv( R[k] + B[k].T @ P[k+1] @ B[k]) @  (S[k] + B[k].T @ P[k+1] @ A[k] )
        P[k] = Q[k] + A[k].T @ P[k+1] @ A[k] + (S[k].T + A[k].T @ P[k+1] @ B[k]) @ K[k]

    return K, P


def mu_to_eta(mu, b):
    """ convert multiplier to multiplier lambda given backoff b"""
    return .5 * mu / b
