import casadi as ca
import numpy as np

class solverNominalMPC:
    def __init__(self, dyn_discr, cost_stage, cost_end, params, 
                    ineq_constr_stage=None, ineq_constr_end=None, init_constr_idx=None):

        self.params = params
        self.nx = params['nx']
        self.nu = params['nu']
        self.N = params['N']

        x = ca.SX.sym('x', self.nx)
        u = ca.SX.sym('u', self.nu)

        self.f_discr = ca.Function('f_discr', [x, u], [dyn_discr(x, u, params)])
        self.l_stage = ca.Function('l_stage', [x, u], [cost_stage(x, u, params)])
        self.l_end = ca.Function('l_end', [x], [cost_end(x, params)])

        self.exist_constr_stage = ineq_constr_stage is not None
        self.exist_constr_end = ineq_constr_end is not None
        self.init_constr_idx = init_constr_idx

        if self.exist_constr_stage:
            self.h_stage = ca.Function('h_stage', [x ,u], [ca.vertcat(*ineq_constr_stage(x, u, params))])
            if self.init_constr_idx is None:
                self.init_constr_idx = list(range(self.h_stage.size1_out(0)))     # enforce all constr at k = 0
        if self.exist_constr_end:
            self.h_end = ca.Function('h_end', [x], [ca.vertcat(*ineq_constr_end(x, params))])
        
        self.init_guess = 0

    def create_solver(self):
        
        nx = self.nx
        nu = self.nu
        N = self.N

        # decision vars
        X = ca.SX.sym('X', nx, N+1)
        U = ca.SX.sym('U', nu, N)
        decvars = ca.vertcat(ca.vec(X), ca.vec(U))
        lbw = [- ca.inf] * decvars.shape[0]
        ubw = [ca.inf] * decvars.shape[0]

        obj = 0
        for k in range(N):
            obj += self.l_stage(X[:,k], U[:,k])
        obj += self.l_end(X[:, -1])

        constr_equa = []
        constr_ineq = []

        for k in range(N):
            constr_equa.append( self.f_discr(X[:, k], U[:, k]) - X[:, k+1] )
            if self.exist_constr_stage:
                    constr_ineq.append(self.h_stage(X[:,k], U[:,k]))
        if self.exist_constr_end:
            constr_ineq.append(self.h_end(X[:,-1]))


        constr_equa = ca.vertcat(*constr_equa)
        constr_ineq = ca.vertcat(*constr_ineq)
        g = ca.vertcat(constr_equa, constr_ineq)
        lbg = ca.vertcat( [0] * constr_equa.shape[0], [-np.inf] * constr_ineq.shape[0]  )
        ubg = 0

        nlp = {}
        nlp['x'] = decvars
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

        self.solver = solver
        self.decvars = decvars
        self.X = X
        self.U = U

        self.lbw = lbw
        self.ubw = ubw
        self.lbg = lbg
        self.ubg = ubg
        self.defined_init_vals = False


    def set_initial_all(self, X, U):
        '''
        set initial guess for X and U trajectories
        X: nx by N+1 matrix
        U: ny by N   matrix
        '''
        init_guess = []
        init_guess.append(ca.vec(X))
        init_guess.append(ca.vec(U))
        self.init_guess = ca.vertcat(*init_guess)


    def set_value_x0(self, x0):
        '''
        set value of initial state x0 
        '''

        # set x0 by adding to box constr
        try:
            iter(x0)
        except:
            x0 = [x0]
        self.lbw[:self.nx] = np.array(x0)
        self.ubw[:self.nx] = np.array(x0)

        self.defined_init_vals = True

    
    def get_value_u0(self):
        '''
        get first control of solution (MPC law)
        '''
        return self.Uopt[:,0]


    def get_sol(self):
        '''
        get OCP solution (X and U)
        '''
        return self.Xopt, self.Uopt

    def solve(self):
        '''
        solve OCP
        '''
        assert(self.defined_init_vals) 

        sol = self.solver( \
            x0  = self.init_guess, \
            lbx = self.lbw, \
            ubx = self.ubw, \
            ubg = self.ubg, \
            lbg = self.lbg, \
        )

        Xopt = ca.DM(ca.substitute(self.X, self.decvars, sol['x'])).full().squeeze()
        Uopt = ca.DM(ca.substitute(self.U, self.decvars, sol['x'])).full().squeeze()
        
        self.sol = sol
        self.Xopt = Xopt
        self.Uopt = Uopt

        return Xopt, Uopt