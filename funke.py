import numpy as np
from firedrake import *
from firedrake_adjoint import *
from gradient_descent import minimize_gs

mesh = RectangleMesh(16, 16, 1, 1, -1, -1)
bids = [1, 2, 3, 4]

# we actually want to optimise for q
# but we use q=beta*qc and use qc as the control
# this way we can rescale the gradient dJ/dqc=dJ/dq dq/dqc=beta dJ/dq
# we choose beta such dJ/dq is of order 1 in the initial optimisation iterations
# this is because L-BFGS-B behaves poorly with gradients that are too large or too small
# in the initial iterations when it doesn't have a very accurate Hessian approximation yet
beta = 1e+6

outf = File('funke.pvd', adaptive=True)

def run_model(control, initial_guess=None):
    V = control.function_space()
    mesh = V.mesh()
    u = initial_guess or Function(V, name="Solution")
    v = TestFunction(V)

    q = Function(V, name='Control')

    x1, x2  = SpatialCoordinate(mesh)
    f = 1/((x1-0.5)**2 + (x2-0.5)**2)

    F = (
            dot(grad(u), grad(v)) * dx
            - q*v*dx
            )

    q.assign(beta*control)

    bc = DirichletBC(V, 0, bids)
    solve(F==0, u, bcs=bc, solver_parameters={'snes_converged_reason': None})

    x1, x2  = SpatialCoordinate(mesh)
    eps = np.finfo(float).eps
    ud = exp(-1/(1-x1**2+eps)-1/(1-x2**2+eps))
    udf = Function(V, name='Desired Solution').interpolate(ud)
    alpha = Constant(0) #1e-7
    J = assemble(0.5 * dot(u-ud, u-ud) * dx + alpha/2 * dot(q, q) *dx)

    outf.write(q, u, udf)
    return J, u


ele = FiniteElement("CG", triangle, 1)
bounds = None

if False:
    # fixed mesh with L-BFGS-B
    V = FunctionSpace(mesh, ele)
    q = Function(V)
    c = Control(q)
    J, u = run_model(q)
    rf = ReducedFunctional(J, c)

    qopt = minimize(rf, bounds=bounds, options={"disp": True, "gtol": 1e-10, "ftol": 1e-12,
                                                'maxiter': 200})
    # run model again for output with optimal control
    J, u = run_model(qopt)
else:
    minimize_gs(run_model, mesh, ele, bounds=bounds,
                params={
                        'move_mesh': True,
                        'debug_movement_files': False})
