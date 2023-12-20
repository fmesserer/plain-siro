# SIRO for closed loop robust MPC
An implementation of the SIRO algorithm for the solution of closed loop robust OCP / robust MPC problems, as described in

> Florian Messerer, Moritz Diehl. An Efficient Algorithm for Tube-based Robust Nonlinear Optimal Control with Optimal Linear Feedback. Proceedings of the IEEE Conference on Decision and Control (CDC), 2021.  https://doi.org/10.1109/CDC45484.2021.9683712; https://cdn.syscop.de/publications/Messerer2021.pdf

The code here is a cleaned up version of the code used for the paper, and thus has been published with some delay.
The exact results from the paper can still be reproduced.
In addition to the kite problem described in the paper, there are also two simpler problems.

### A note on the convergence criterion

The code uses the KKT conditions of the robust OCP as convergence criterion.
However, the algorithm solves this robust OCP by alternating between the solution of a nominal OCP and a Riccati recursion, such that the actual robust OCP is never explicitly constructed.
Thus, checking its KKT conditions creates some overhead in the form of additional lines of code which are not straightforward to understand, see `KKT_prepare()` and `KKT_check()`.
When developing the algorithm, this helped to ensure that it actually converges to a KKT point, as opposed to converging to some meaningless stationary point.
However, the theory says that every stationary point of the algorithm has to be a KKT point of the robust OCP (and vice versa).
Thus, a convergence criterion which is both easier to code and easier to understand is the change of the nominal trajectory across two iterations. In the notation of the paper this reads as $\Vert y_{k+1} - y_k\Vert$. 
In the code, the latter convergence criterion is used for the open loop version.

### A note on the open loop version (aka zoRO)

The main point of SIRO is to efficiently optimize the linear feedback matrices (control gains), i.e., to solve a *closed loop* robust OCP.
Nonetheless, with very little overhead the SIRO algorithm can be simplified to solve the corresponding *open loop* robust OCP.
The resulting algorithm is then very similar to zoRO (Zero-Order Robust Optimization). For an efficient implementation of zoRO see
> Jonathan Frey, Yunfan Gao, Florian Messerer, Amon Lahr, Melanie Zeilinger, Moritz Diehl. Efficient Zero-Order Robust Optimization for Real-Time Model
Predictive Control with acados, Arxiv 2023. https://arxiv.org/pdf/2311.04557.pdf



## File overview

Solvers
* `solverNominalMPC.py`
    * solver for the nominal OCP
* `solverOpenLoopRMPC`
    * non-tailored solver for the open-loop robust OCP, solved with IPOPT
* `solverClosedLoopRMPC_SIRO.py`
    * solver for the robust OCP (both closed-loop and open-loop), using the tailored algorithm

Examples
* simple1Dsystem
    * a simple onedimensional nonlinear system,  $\dot x = x^3 - u + w$
* robot2D_simple
    * a simple 2D "robot" with obstacles (linear system, just a 
    double integrator of the controls)
* kite
    * the kite problem as described in the paper

Examples file structure
* `XXX_run.py`
    * solve and compare nominal, open closed loop OCP (no MPC involved)
* `XXX_runMPC.py`
    * MPC loop
* `XXX_probdef.py`
    * defines system and OCP
* `XXX_plotutils.py`
    * functions for plotting the system

For the kite problem, there are additionally
* `kite_create_plots_paper.py`
    * creates the plots as used in the paper. This is based on a result file generated from kite_run.py
* `kite_results_paper/`
    * folder containing the result files corresponding to the results and plots published in the paper.

## Dependencies
The required Python packages are listed in `requirements.txt`.
You can install them by running
    
```
pip install -r requirements.txt
```
### Note on the CasADi version
The CasADi version has to be 3.5.5 or older.

This is due to the line 
```H_func = ca.Function('H_func', [y], [H])```
in `solverClodedLoopRMPC_SIRO.py`, in the definition of the function `KKT_prepare`.

The problem is that `H` depends also on symbolic variables other than `y`, which seems to be no longer allowed starting from CasADi 3.6.
Probably by adequately modifying the mentioned line and a few of the following ones, the code will also run with the most recent CasADi version.
