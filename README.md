# SIRO for closed loop robust MPC
An implementation of the SIRO algorithm for the solution of closed loop robust OCP / robust MPC problems, as described in

> Florian Messerer, Moritz Diehl. An Efficient Algorithm for Tube-based Robust Nonlinear Optimal Control with Optimal Linear Feedback. Proceedings of the IEEE Conference on Decision and Control (CDC), 2021.  https://doi.org/10.1109/CDC45484.2021.9683712; https://cdn.syscop.de/publications/Messerer2021.pdf

The code here is not exactly identical with the code used in the paper (which was somewhat messy).
Most notably the kite example from the paper is still missing, but there are two easier toy problems.
Originally the plan was to publish the code only once the kite example is also cleaned up.
However, since I never got around to doing this, and already some people were asking, I am publishing this preliminary version.
The exact code as used for the paper is available upon request.

### A note on the convergence criterion

The code uses the KKT conditions of the robust OCP as convergence criterion.
However, the algorithm solves this robust OCP by alternating between the solution of a nominal OCP and a Riccati recursion, such that the actual robust OCP is never explicitly constructed.
Thus, checking its KKT conditions creates some overhead in the form of additional lines of code which are not straightforward to understand, see `KKT_prepare()` and `KKT_check()`.
When developing the algorithm, this helped to ensure that it actually converges to a KKT point, as opposed to converging to some meaningless stationary point.
However, the theory says that every stationary point of the algorithm has to be a KKT point of the robust OCP (and vice versa).
Thus, a convergence criterion which is both easier to code and easier to understand is the change of the nominal trajectory across two iterations. In the notation of the paper this reads as $\Vert y_{k+1} - y_k\Vert$. 
In the code, cf the convergence criterion as used for the open loop version.

### A note on the open loop version (aka zoRO)

The main point of SIRO is to efficiently optimize the linear feedback matrices (control gains), i.e., to solve a *closed loop* robust OCP.
Nonetheless, with very little overhead the SIRO algorithm can be simplified to solve the corresponding *open loop* robust OCP.
The resulting algorithm is then very similar to zoRO (Zero-Order Robust Optimization). For an efficient implementation of zoRO see https://github.com/acados/acados/tree/master/examples/acados_python/zoRO_example. 

### File overview

Two solvers
* `solverNominalMPC.py`
    * solver for the nominal OCP
* `solverClosedLoopRMPC_SIRO.py`
    * solver for the robust OCP (both closed-loop and open-loop), using the tailored algorithm

Two examples
* simple1Dsystem
    * a simple onedimensional nonlinear system,  $\dot x = x^3 - u + w$
* robot2D_simple
    * a simple 2D "robot" with obstacles (linear system, just a 
    double integrator of the controls)

Examples file structure
* `XXX_run.py`
    * solve and compare nominal, open closed loop OCP (no MPC involved)
* `XXX_runMPC.py`
    * MPC loop
* `XXX_probdef.py`
    * defines system and OCP
* `XXX_plotutils.py`
    * functions for plotting the system

Dependencies
* `requirements.txt`
    * The required Python packages. You can install them by running
    
```
pip install -r requirements.txt
```
