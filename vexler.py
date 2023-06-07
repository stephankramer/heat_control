from firedrake import *
from firedrake_adjoint import *
from gradient_descent import minimize_gs

mesh = Mesh('square_with_hole.msh')

# we actually want to optimise for q
# but we use q=beta*qc and use qc as the control
# this way we can rescale the gradient dJ/dqc=dJ/dq dq/dqc=beta dJ/dq
# we choose beta such dJ/dq is of order 1 in the initial optimisation iterations
# this is because L-BFGS-B behaves poorly with gradients that are too large or too small
# in the initial iterations when it doesn't have a very accurate Hessian approximation yet
beta = 1e+6

outf = File('qopt.pvd', adaptive=True)


def run_model(control, initial_guess=None):
    V = control.function_space()
    mesh = V.mesh()

    u = initial_guess or Function(V, name="Solution")
    v = TestFunction(V)
    q = Function(V, name='Control')

    x1, x2 = SpatialCoordinate(mesh)
    f = 1/((x1-0.5)**2 + (x2-0.5)**2)

    F = (
            dot(grad(u), grad(v)) * dx
            + 30 * u**3 * v * dx
            + u * v * dx
            - (f + q)*v*dx
            )
    bc = DirichletBC(V, 0, [i+1 for i in range(8)])

    q.assign(beta*control)

    solve(F == 0, u, bcs=bc, solver_parameters={'snes_converged_reason': None})

    x1, x2 = SpatialCoordinate(mesh)
    ud = x1*x2
    udf = Function(V, name='Desired Solution').interpolate(ud)
    alpha = 1e-4
    J = assemble(0.5 * dot(u-ud, u-ud) * dx + alpha/2 * dot(q, q) * dx)

    outf.write(q, u, udf)
    return J, u


ele = FiniteElement("CG", triangle, 1)
# bounds for q are -7<q<20
# but the optimisation algorithm deals with qc=q/beta
bounds = [-7/beta, 20/beta]

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
                params={'max_adapts': 100})
