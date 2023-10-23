import casadi as ca
import numpy as np

class solverOpenLoopRMPC:
    def __init__(self, dyn_discr, cost_stage, cost_end, params, 
                    ineq_constr_stage=None, ineq_constr_end=None):

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
        w = ca.SX.sym('u', nw)

        self.f_discr = ca.Function('f_discr', [x, u, w], [dyn_discr(x, u, w, params)])
        self.l_stage = ca.Function('l_stage', [x, u], [cost_stage(x, u, params)])
        self.l_end = ca.Function('l_end', [x], [cost_end(x, params)])

        self.exist_constr_stage = ineq_constr_stage is not None
        self.exist_constr_end = ineq_constr_end is not None

        if self.exist_constr_stage:
            self.h_stage = ca.Function('h_stage', [x ,u], [ca.vertcat(*ineq_constr_stage(x, u, params))])
        if self.exist_constr_end:
            self.h_end = ca.Function('h_end', [x], [ca.vertcat(*ineq_constr_end(x, params))])

        # uncertainty part
        P = ca.SX.sym('P', nx, nx)                          # state deviation elipsoid matrix
        W = ca.SX.sym('W', nw, nw)                          # noise eliopsoid matrix
        A = ca.jacobian( self.f_discr(x, u, w), x )         # dynamics state sensitivity
        Gamma = ca.jacobian(self.f_discr(x, u, w), w)       # dynamics noise sensitivity

        # uncertainty dynamics
        lyap_dyn_ = ca.substitute( A @ P @ A.T + Gamma @ W @ Gamma.T , w, 0)
        self.lyap_dyn = ca.Function('lyap_dyn', [x, u, P, W], [ lyap_dyn_ ]) 

        if self.exist_constr_stage:
            H = ca.jacobian(self.h_stage(x, u), x)
            nh = H.shape[0]
            h_ = [ self.h_stage(x, u)[i] + ca.sqrt( H[i,:] @ P @ H[i,:].T ) for i in range(nh)  ]
            self.h_stage_rob = ca.Function('h_stage_rob', [x, u, P], [ca.vertcat(*h_)])

        if self.exist_constr_end:
            H = ca.jacobian(self.h_end(x), x)
            nh = H.shape[0]
            h_ = [ self.h_end(x)[i] + ca.sqrt( H[i,:] @ P @ H[i,:].T ) for i in range(nh)  ]
            self.h_end_rob = ca.Function('h_end_rob', [x, P], [ca.vertcat(*h_)])

    def create_solver(self):
        
        nx = self.nx
        nu = self.nu
        nw = self.nw
        N = self.N

        opti = ca.Opti()
        # decision vars
        X = opti.variable(nx, N+1)
        U = opti.variable(nu, N)
        P = [opti.variable(nx, nx) for _ in range(N+1)]
    
        # parameters
        x0bar = opti.parameter(nx, 1)
        P0bar = opti.parameter(nx, nx)
        W = opti.parameter(nw, nw)
        
        # objective
        obj = 0
        for k in range(N):
            obj += self.l_stage(X[:,k], U[:,k])
        obj += self.l_end(X[:, -1])
        opti.minimize( obj )

        # constraints
        opti.subject_to( x0bar - X[:,0]  == 0)
        opti.subject_to( P0bar - P[0]  == 0)
        for k in range(N):
            # nominal dynamic
            opti.subject_to( self.f_discr(X[:, k], U[:, k], 0) - X[:, k+1] == 0 )
            # ellipsoid dynamic
            opti.subject_to( self.lyap_dyn(X[:, k], U[:, k], P[k], W) - P[k+1] == 0 )
            # robustified constr
            if self.exist_constr_stage:
                    opti.subject_to(self.h_stage_rob(X[:,k], U[:,k], P[k]) <= 0)

        if self.exist_constr_end:
            opti.subject_to(self.h_end_rob(X[:,-1], P[-1]) <= 0)

        opti.solver('ipopt')

        self.opti = opti
        self.X = X
        self.U = U
        self.P = P
        self.x0bar = x0bar
        self.P0bar = P0bar
        self.W = W

    def set_initial_all(self, X, U, P):
        self.opti.set_initial(self.X, X)
        self.opti.set_initial(self.U, U)

        # if only one value is passed
        if P is not list:
            P = [P] * (self.N + 1)
        for p, p0 in zip(self.P, P):
            self.opti.set_initial(p, p0)

    def set_value_all(self, x0, P0, W):
        self.opti.set_value(self.x0bar, x0)
        self.opti.set_value(self.P0bar, P0)
        self.opti.set_value(self.W, W)

    def solve(self):
        sol = self.opti.solve()
        Xopt = sol.value(self.X)
        Uopt = sol.value(self.U)
        Popt = [sol.value(p) for p in self.P]

        self.sol = sol
        self.Xopt = Xopt
        self.Uopt = Uopt
        self.Popt = Popt

        return Xopt, Uopt, Popt

    def get_sol(self):
        return self.Xopt, self.Uopt, self.Popt
